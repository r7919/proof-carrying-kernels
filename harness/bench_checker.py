"""Overhead of the trusted GPU checker on decode-sized GEMV (Phase 1 gate: < 5%).

For each shape and block size b, times the coalesced GEMV alone and the GEMV followed by
the checker (production mode, K checks), each on a cold L2. The checker reads K checksum
rows c and one rounded-up a row (FP32, n values each) per block of b BF16 weight rows, so
its extra DRAM traffic is about (K + 1) * 2 / b of the GEMV's.

Run: uv run python harness/bench_checker.py
"""

from __future__ import annotations

import argparse

import torch

from pck.cuda import GpuBlockedChecker, gemv_coalesced
from pck.cuda.timing import ColdL2Timer
from pck.fp import tree_height

SHAPES = [(4096, 4096), (14336, 4096), (4096, 14336)]
BLOCKS = [16, 32, 64, 128, 256]
THREADS = 256


def main() -> None:
    parser = argparse.ArgumentParser(description="Checker overhead on BF16 GEMV.")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--checks", type=int, default=1, help="K, sign vectors per block")
    args = parser.parse_args()

    timer = ColdL2Timer(args.warmup, args.reps)
    gen = torch.Generator().manual_seed(0)
    print(f"device: {torch.cuda.get_device_name()}; coalesced/{THREADS}; K = {args.checks}")
    print("times: median ms, cold L2; overhead = (GEMV + check) / GEMV - 1")
    print(
        f"{'m x n':>13s} {'b':>4s} {'GEMV':>8s} {'+check':>8s} {'overhead':>9s}"
        f" {'traffic est.':>12s} {'false alarms':>12s}"
    )
    for m, n in SHAPES:
        W = (torch.randn(m, n, device="cuda") * 0.02).to(torch.bfloat16)
        x = torch.randn(n, device="cuda").to(torch.bfloat16)
        height = tree_height(n, THREADS, vec=8)
        gemv = timer(lambda: gemv_coalesced(W, x, THREADS))  # noqa: B023
        for b in BLOCKS:
            checker = GpuBlockedChecker(W, b, height, checks=args.checks, generator=gen)
            y = gemv_coalesced(W, x, THREADS)
            alarms = int((~checker.check(x, y).passed).sum())
            both = timer(lambda: checker.check(x, gemv_coalesced(W, x, THREADS)))  # noqa: B023
            est = (args.checks + 1) * 2 / b
            print(
                f"{f'{m}x{n}':>13s} {b:4d} {gemv.median_ms:8.4f} {both.median_ms:8.4f}"
                f" {both.median_ms / gemv.median_ms - 1:9.1%} {est:12.1%} {alarms:12d}"
            )


if __name__ == "__main__":
    main()
