"""Per-kernel GPU durations for GEMV + checker, to separate kernel time from launch gaps.

The cold-L2 benchmark measures wall time between CUDA events, which includes any idle gap
between the GEMV and the checker. The PyTorch profiler reports how long each kernel
itself ran, so (event overhead - checker kernel time) is the cost of the extra launch.

Run: uv run python harness/profile_checker.py [--m 4096 --n 4096 --block 128]
"""

from __future__ import annotations

import argparse

import torch
from torch.profiler import ProfilerActivity, profile

from pck.cuda import GpuBlockedChecker, gemv_coalesced
from pck.fp import tree_height

THREADS = 256


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile GEMV + checker kernels.")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument("--checks", type=int, default=1)
    parser.add_argument("--segments", type=int, default=None)
    parser.add_argument("--reps", type=int, default=50)
    args = parser.parse_args()

    W = (torch.randn(args.m, args.n, device="cuda") * 0.02).to(torch.bfloat16)
    x = torch.randn(args.n, device="cuda").to(torch.bfloat16)
    checker = GpuBlockedChecker(
        W,
        args.block,
        tree_height(args.n, THREADS, vec=8),
        checks=args.checks,
        generator=torch.Generator().manual_seed(0),
        segments=args.segments,
    )
    l2 = int(getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 64 << 20))
    flush = torch.empty(2 * l2, dtype=torch.uint8, device="cuda")
    for _ in range(10):
        checker.check(x, gemv_coalesced(W, x, THREADS))
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(args.reps):
            flush.zero_()
            checker.check(x, gemv_coalesced(W, x, THREADS))
        torch.cuda.synchronize()

    print(
        f"m={args.m} n={args.n} b={args.block} K={args.checks} segments={checker.segments}"
        f" grid={args.checks * checker.num_blocks}x{checker.segments}"
    )
    print(prof.key_averages().table(sort_by="device_time_total", row_limit=8))


if __name__ == "__main__":
    main()
