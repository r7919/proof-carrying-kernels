"""Benchmark the naive BF16 GEMV against cuBLAS (torch.mv) on decode-sized shapes.

Decode reads every weight from DRAM once per token, so each timed run starts with a
cold L2: a buffer twice the L2 size is overwritten before every rep (outside the
timed region). Without this, weights that fit in L2 are served from cache and report
bandwidth above the DRAM peak.

Bandwidth is reported as a percentage of the measured device-to-device copy
bandwidth, a practical ceiling for this GPU. Pass --peak-gbps to also compare
against the spec-sheet number.

Run plugged in, in high-performance mode, and log clocks alongside, e.g.
  nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv -l 1

Run: uv run python harness/bench_gemv.py
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable
from dataclasses import dataclass

import torch

from pck.cuda import gemv_naive

SHAPES = [(896, 896), (1536, 1536), (2048, 2048), (3584, 3584), (4096, 4096), (14336, 4096)]


@dataclass
class Timing:
    median_ms: float
    p10_ms: float
    p90_ms: float


class ColdL2Timer:
    """Times a CUDA function with CUDA events, flushing L2 before every rep."""

    def __init__(self, warmup: int, reps: int) -> None:
        l2_bytes = int(getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 64 << 20))
        self.flush = torch.empty(2 * max(l2_bytes, 1 << 20), dtype=torch.uint8, device="cuda")
        self.warmup, self.reps = warmup, reps

    def __call__(self, fn: Callable[[], object]) -> Timing:
        for _ in range(self.warmup):
            fn()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(self.reps)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(self.reps)]
        for start, end in zip(starts, ends, strict=True):
            self.flush.zero_()
            start.record()
            fn()
            end.record()
        torch.cuda.synchronize()
        times = sorted(float(s.elapsed_time(e)) for s, e in zip(starts, ends, strict=True))
        deciles = statistics.quantiles(times, n=10)
        return Timing(statistics.median(times), deciles[0], deciles[-1])


def copy_bandwidth_gbps(timer: ColdL2Timer, nbytes: int = 1 << 30) -> float:
    """Device-to-device copy bandwidth (read + write), a practical DRAM ceiling."""
    src = torch.empty(nbytes // 2, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    t = timer(lambda: dst.copy_(src))
    return 2 * src.numel() / (t.median_ms / 1e3) / 1e9


def check_correct(W: torch.Tensor, x: torch.Tensor) -> None:
    ref = W.float() @ x.float()
    torch.testing.assert_close(gemv_naive(W, x, 256), ref, rtol=1e-3, atol=1e-3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark BF16 GEMV kernels against cuBLAS.")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--peak-gbps", type=float, help="spec-sheet DRAM bandwidth, GB/s")
    args = parser.parse_args()

    torch.manual_seed(0)
    timer = ColdL2Timer(args.warmup, args.reps)
    props = torch.cuda.get_device_properties(0)
    copy_bw = copy_bandwidth_gbps(timer)
    print(f"device: {props.name}, L2 {timer.flush.numel() // 2 >> 20} MiB")
    print(f"measured copy bandwidth: {copy_bw:.0f} GB/s (100% below)")
    if args.peak_gbps:
        print(f"spec-sheet bandwidth: {args.peak_gbps:.0f} GB/s")
    print("times: median ms [p10, p90], cold L2")
    print(
        f"{'m x n':>12s} {'cuBLAS ms':>22s} {'naive ms':>22s} {'speedup':>8s}"
        f" {'cuBLAS GB/s':>12s} {'naive GB/s':>11s} {'naive %copy':>11s}"
    )
    for m, n in SHAPES:
        W = torch.randn(m, n, device="cuda").to(torch.bfloat16)
        x = torch.randn(n, device="cuda").to(torch.bfloat16)
        check_correct(W, x)
        cublas = timer(lambda: torch.mv(W, x))  # noqa: B023
        naive = timer(lambda: gemv_naive(W, x, 256))  # noqa: B023
        # Bytes each kernel must move: W and x in BF16; y in BF16 (cuBLAS) or FP32 (naive).
        cublas_gb = (2 * m * n + 2 * n + 2 * m) / 1e9
        naive_gb = (2 * m * n + 2 * n + 4 * m) / 1e9
        cublas_bw = cublas_gb / (cublas.median_ms / 1e3)
        naive_bw = naive_gb / (naive.median_ms / 1e3)

        def fmt(t: Timing) -> str:
            return f"{t.median_ms:.4f} [{t.p10_ms:.4f}, {t.p90_ms:.4f}]"

        print(
            f"{f'{m} x {n}':>12s} {fmt(cublas):>22s} {fmt(naive):>22s}"
            f" {cublas.median_ms / naive.median_ms:7.2f}x"
            f" {cublas_bw:12.0f} {naive_bw:11.0f} {naive_bw / copy_bw:11.0%}"
        )


if __name__ == "__main__":
    main()
