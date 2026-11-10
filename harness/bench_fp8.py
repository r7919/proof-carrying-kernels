"""FP8 (E4M3) weights: GEMV speed against BF16, and the checker's cost on an FP8 layer.

Part 1 times y = W x per decode shape on a cold L2: cuBLAS (torch.mv, BF16), the BF16
coalesced kernel, and the FP8 kernel (`pck.cuda.gemv_fp8`, one scale per row applied
after the GEMV). The FP8 kernel reads one byte per weight instead of two.

Part 2 runs a fused Llama-3-8B decode layer (qkv, o, gate_up, down) with FP8 weights and
checks every GEMV with one batched checker launch (`check_many`), as
`bench_decode_layer.py` does for BF16. The checker verifies z = Q x against the FP8
weights Q (exact in BF16), so its cost is the BF16 layer's, now against a GEMV that
moves half the bytes.

The quantization error (FP8 against the BF16 weights) is a property of the format, not
of the kernel; it is printed for reference. The checker bounds the kernel's own rounding.

Run: uv run python harness/bench_fp8.py
"""

from __future__ import annotations

import argparse

import torch

from pck.cuda import (
    DEFAULT_BLOCK,
    GpuBlockedChecker,
    check_many,
    gemv_coalesced,
    gemv_fp8,
    quantize_fp8,
)
from pck.cuda.timing import ColdL2Timer, copy_bandwidth_gbps
from pck.fp import tree_height

THREADS = 256
SHAPES = [(896, 896), (896, 4864), (2048, 2048), (4096, 4096), (14336, 4096), (4096, 14336)]
LAYER = [("qkv", 6144, 4096), ("o", 4096, 4096), ("gate_up", 28672, 4096), ("down", 4096, 14336)]


def per_shape(timer: ColdL2Timer, copy_bw: float, gen: torch.Generator) -> None:
    print("part 1: GEMV, median ms on a cold L2; %copy = share of measured copy bandwidth")
    print(
        f"{'m x n':>12s} {'cuBLAS bf16':>12s} {'ours bf16':>10s} {'ours fp8':>9s}"
        f" {'fp8 %copy':>9s} {'fp8 vs bf16':>11s} {'fp8 vs cuBLAS':>13s} {'quant err':>9s}"
    )
    for m, n in SHAPES:
        W = (torch.randn(m, n, generator=gen) * 0.02).to(torch.bfloat16).cuda()
        x = torch.randn(n, generator=gen).to(torch.bfloat16).cuda()
        Q, s = quantize_fp8(W)
        y = W.double() @ x.double()
        err = float((s.double() * gemv_fp8(Q, x, THREADS).double() - y).norm() / y.norm())
        cublas = timer(lambda: torch.mv(W, x)).median_ms  # noqa: B023
        bf16 = timer(lambda: gemv_coalesced(W, x, THREADS)).median_ms  # noqa: B023
        fp8 = timer(lambda: gemv_fp8(Q, x, THREADS)).median_ms  # noqa: B023
        bw = (m * n + 2 * n + 4 * m) / 1e9 / (fp8 / 1e3)
        print(
            f"{f'{m} x {n}':>12s} {cublas:12.4f} {bf16:10.4f} {fp8:9.4f} {bw / copy_bw:9.0%}"
            f" {bf16 / fp8:10.2f}x {cublas / fp8:12.2f}x {err:9.2%}"
        )


def layer(timer: ColdL2Timer, block: int, gen: torch.Generator) -> None:
    mats = []
    for name, m, n in LAYER:
        W = (torch.randn(m, n, generator=gen) * 0.02).to(torch.bfloat16).cuda()
        x = torch.randn(n, generator=gen).to(torch.bfloat16).cuda()
        Q, _ = quantize_fp8(W)
        h = tree_height(n, THREADS, vec=16)
        checker = GpuBlockedChecker(Q.to(torch.bfloat16), block, h, checks=1, generator=gen)
        mats.append((name, W, Q, x, checker))

    def bf16_only() -> None:
        for _, W, _, x, _ in mats:
            gemv_coalesced(W, x, THREADS)

    def fp8_only() -> None:
        for _, _, Q, x, _ in mats:
            gemv_fp8(Q, x, THREADS)

    def fp8_checked() -> None:
        check_many([(c, x, gemv_fp8(Q, x, THREADS)) for _, _, Q, x, c in mats])

    results = check_many([(c, x, gemv_fp8(Q, x, THREADS)) for _, _, Q, x, c in mats])
    ok = all(bool(r.verdict.all()) for r in results)
    base = timer(bf16_only).median_ms
    fp8 = timer(fp8_only).median_ms
    checked = timer(fp8_checked).median_ms
    print(f"\npart 2: fused Llama-3-8B decode layer (4 GEMVs), b = {block}, K = 1, cold L2")
    print(f"all checks pass on FP8 outputs: {ok}")
    print(f"  BF16 GEMVs           {base:.4f} ms")
    print(f"  FP8 GEMVs            {fp8:.4f} ms   ({base / fp8:.2f}x faster than BF16)")
    print(f"  FP8 GEMVs + checks   {checked:.4f} ms   (checker overhead {checked / fp8 - 1:.1%})")


def main() -> None:
    parser = argparse.ArgumentParser(description="FP8 GEMV speed and checker overhead.")
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--block", type=int, default=DEFAULT_BLOCK)
    args = parser.parse_args()
    gen = torch.Generator().manual_seed(0)
    timer = ColdL2Timer(20, args.reps)
    copy_bw = copy_bandwidth_gbps(timer)
    print(f"device: {torch.cuda.get_device_name()}; copy bandwidth {copy_bw:.0f} GB/s")
    per_shape(timer, copy_bw, gen)
    layer(timer, args.block, gen)


if __name__ == "__main__":
    main()
