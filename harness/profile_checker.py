"""Split the checker's overhead into launch cost and kernel time, using CUDA events only.

The PyTorch profiler needs CUPTI, which is unavailable under WSL2 on some drivers, so this
script isolates the parts with controlled comparisons instead (all on a cold L2):

  gemv              the coalesced GEMV alone
  gemv + trivial    GEMV followed by a tiny kernel: the cost of any second launch
  gemv + check      GEMV followed by the checker
  graph: gemv       the GEMV replayed from a CUDA graph
  graph: gemv+check both kernels in one CUDA graph (launch overhead mostly removed)
  check, warm x50   50 back-to-back checks per timed run, per call (L2 warm)

Run: uv run python harness/profile_checker.py [--m 4096 --n 4096 --block 128]
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import torch

from pck.cuda import GpuBlockedChecker, gemv_coalesced
from pck.cuda.timing import ColdL2Timer
from pck.fp import tree_height

THREADS = 256


def graphed(fn: Callable[[], object]) -> Callable[[], None]:
    """Capture ``fn`` in a CUDA graph and return a function that replays it."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):  # warm up on a side stream, as capture requires
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()

    def replay() -> None:
        graph.replay()

    return replay


def main() -> None:
    parser = argparse.ArgumentParser(description="Split checker overhead into its parts.")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument("--checks", type=int, default=1)
    parser.add_argument("--segments", type=int, default=None)
    parser.add_argument("--reps", type=int, default=200)
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
    y = gemv_coalesced(W, x, THREADS)
    tiny = torch.zeros(32, device="cuda")
    timer = ColdL2Timer(20, args.reps)
    warm = ColdL2Timer(5, 20)
    warm.flush = torch.empty(1, dtype=torch.uint8, device="cuda")  # no flush

    def gemv_check() -> None:
        checker.check(x, gemv_coalesced(W, x, THREADS))

    def checks_x50() -> None:
        for _ in range(50):
            checker.check(x, y)

    rows = {
        "gemv": timer(lambda: gemv_coalesced(W, x, THREADS)).median_ms,
        "gemv + trivial": timer(lambda: (gemv_coalesced(W, x, THREADS), tiny.add_(1))).median_ms,
        "gemv + check": timer(gemv_check).median_ms,
        "graph: gemv": timer(graphed(lambda: gemv_coalesced(W, x, THREADS))).median_ms,
        "graph: gemv+check": timer(graphed(gemv_check)).median_ms,
        "check, warm x50": warm(checks_x50).median_ms / 50,
    }
    print(
        f"m={args.m} n={args.n} b={args.block} K={args.checks} segments={checker.segments}"
        f" grid={args.checks * checker.num_blocks}x{checker.segments}"
    )
    base = rows["gemv"]
    for name, ms in rows.items():
        extra = "" if name == "check, warm x50" else f"  (+{(ms - base) * 1e3:6.1f} us)"
        print(f"{name:>18s} {ms * 1e3:8.1f} us{extra}")


if __name__ == "__main__":
    main()
