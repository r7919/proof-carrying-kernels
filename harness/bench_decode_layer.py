"""Checker overhead on a full decode layer, with checks serial or overlapped.

One transformer decode layer at batch 1 multiplies by seven weight matrices (Llama-3-8B
shapes: q, k, v, o, gate, up, down). Inference engines fuse the ones that share an input,
so the default "fused" layout runs four GEMVs: qkv, o, gate_up, down; "split" runs all
seven. This times the whole layer four ways, on a cold L2:

  gemv only       the seven coalesced GEMVs
  serial checks   each GEMV followed by its checker on the same stream
  overlapped      each checker on a side stream, started once its GEMV finishes, so it
                  runs while the next GEMV does; the layer ends when the last check ends
  batched         all GEMVs, then one checker launch for the whole layer (check_many)

The target is overhead < 5% at the chosen block size b.

Run: uv run python harness/bench_decode_layer.py [--block 64]   (default b = 128)
"""

from __future__ import annotations

import argparse

import torch

from pck.cuda import (
    DEFAULT_BLOCK,
    GpuBlockedChecker,
    PendingCheck,
    SideStreamChecker,
    check_many,
    gemv_coalesced,
)
from pck.cuda.timing import ColdL2Timer, device_name
from pck.fp import tree_height

THREADS = 256
# (name, m, n) for Llama-3-8B: hidden 4096, 8 KV heads of 128, MLP 14336.
LAYOUTS = {
    "split": [
        ("q", 4096, 4096),
        ("k", 1024, 4096),
        ("v", 1024, 4096),
        ("o", 4096, 4096),
        ("gate", 14336, 4096),
        ("up", 14336, 4096),
        ("down", 4096, 14336),
    ],
    "fused": [
        ("qkv", 6144, 4096),
        ("o", 4096, 4096),
        ("gate_up", 28672, 4096),
        ("down", 4096, 14336),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Checker overhead on a decode layer.")
    parser.add_argument("--block", type=int, default=DEFAULT_BLOCK)
    parser.add_argument("--checks", type=int, default=1)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--layout", choices=sorted(LAYOUTS), default="fused")
    args = parser.parse_args()

    gen = torch.Generator().manual_seed(0)
    mats = []
    for name, m, n in LAYOUTS[args.layout]:
        W = (torch.randn(m, n, generator=gen) * 0.02).to(torch.bfloat16).cuda()
        x = torch.randn(n, generator=gen).to(torch.bfloat16).cuda()
        checker = GpuBlockedChecker(
            W, args.block, tree_height(n, THREADS, vec=8), checks=args.checks, generator=gen
        )
        mats.append((name, W, x, checker))
    side = SideStreamChecker()
    pending: list[PendingCheck] = []

    def gemv_only() -> None:
        for _, W, x, _ in mats:
            gemv_coalesced(W, x, THREADS)

    def serial() -> None:
        for _, W, x, checker in mats:
            checker.check(x, gemv_coalesced(W, x, THREADS))

    def overlapped() -> None:
        pending.clear()
        for _, W, x, checker in mats:
            pending.append(side.submit(checker, x, gemv_coalesced(W, x, THREADS)))
        side.wait_all()  # the layer is done only when its last check is

    def batched() -> None:
        check_many([(c, x, gemv_coalesced(W, x, THREADS)) for _, W, x, c in mats])

    # Correctness first: every check must pass on legit outputs, in every mode.
    serial_ok = all(bool(c.verdict(x, gemv_coalesced(W, x, THREADS)).all()) for _, W, x, c in mats)
    overlapped()
    overlap_ok = all(bool(p.wait().verdict.all()) for p in pending)
    outputs = [(c, x, gemv_coalesced(W, x, THREADS)) for _, W, x, c in mats]
    batch_ok = all(bool(r.verdict.all()) for r in check_many(outputs))

    timer = ColdL2Timer(10, args.reps)
    base = timer(gemv_only)
    rows = [
        ("gemv only", base),
        ("serial checks", timer(serial)),
        ("overlapped", timer(overlapped)),
        ("batched", timer(batched)),
    ]
    weight_mib = sum(W.numel() * 2 for _, W, _, _ in mats) / 2**20
    bank_mib = sum(c.c.numel() * 4 + c.a.numel() * 2 for *_, c in mats) / 2**20

    print(
        f"device: {device_name()}; {args.layout} layout ({len(mats)} GEMVs);"
        f" b = {args.block}, K = {args.checks};"
        f" weights {weight_mib:.0f} MiB, checksum bank {bank_mib:.1f} MiB"
        f" ({bank_mib / weight_mib:.1%})"
    )
    print(
        f"all checks pass on legit outputs: serial {serial_ok}, overlapped {overlap_ok},"
        f" batched {batch_ok}"
    )
    print(f"{'mode':>14s} {'ms [p10, p90]':>26s} {'overhead':>9s}")
    for name, t in rows:
        print(
            f"{name:>14s} {t.median_ms:8.4f} [{t.p10_ms:.4f}, {t.p90_ms:.4f}]"
            f" {t.median_ms / base.median_ms - 1:9.1%}"
        )


if __name__ == "__main__":
    main()
