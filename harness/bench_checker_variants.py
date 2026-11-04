"""Where does the checker's time go? Times variants of its main loop (diagnostic only).

Every variant uses the production checker's grid, segments and block reductions; only the
per-element arithmetic changes (see harness/csrc/checker_variants.cu). Times are per call:
'warm' runs 50 calls back to back (L2 warm, launch overhead amortized), 'cold' flushes L2
before each call. The production checker (which adds the y terms and the last-block
combine) and the GEMV are listed for reference.

Run: uv run python harness/bench_checker_variants.py
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import torch

from pck.cuda import GpuBlockedChecker, gemv_coalesced
from pck.cuda.timing import ColdL2Timer
from pck.fp import tree_height

THREADS = 256
CASES = [(4096, 4096, 128), (4096, 4096, 64), (14336, 4096, 64), (4096, 14336, 64)]
VARIANTS = {
    0: "loads only",
    1: "FP64 FMA + FP32 ru (prod)",
    2: "FP32 FMA + FP32 ru",
    3: "compensated FP32 + FP32 ru",
    4: "FP64 FMA only",
}


@functools.cache
def variants() -> Any:
    from torch.utils.cpp_extension import load

    csrc = Path(__file__).parent / "csrc"
    return load(
        name="pck_checker_variants",
        sources=[str(csrc / "checker_variants.cpp"), str(csrc / "checker_variants.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--fmad=false"],
    )


def main() -> None:
    cold = ColdL2Timer(10, 100)
    warm = ColdL2Timer(3, 20)
    warm.flush = torch.empty(1, dtype=torch.uint8, device="cuda")  # no flush
    print(f"device: {torch.cuda.get_device_name()}; times in us per call; K = 1")
    for m, n, b in CASES:
        W = (torch.randn(m, n, device="cuda") * 0.02).to(torch.bfloat16)
        x = torch.randn(n, device="cuda").to(torch.bfloat16)
        checker = GpuBlockedChecker(
            W, b, tree_height(n, THREADS, vec=8), generator=torch.Generator().manual_seed(0)
        )
        y = gemv_coalesced(W, x, THREADS)
        out = torch.empty((1, checker.num_blocks, checker.segments), dtype=torch.float64).cuda()
        mib = (checker.c.numel() * 4 + checker.a.numel() * 2) / 2**20
        gemv = cold(lambda: gemv_coalesced(W, x, THREADS)).median_ms  # noqa: B023
        print(
            f"\n{m}x{n} b={b}: grid {checker.num_blocks}x{checker.segments},"
            f" checker data {mib:.2f} MiB, GEMV {gemv * 1e3:.1f} us"
        )
        print(f"  {'variant':>28s} {'warm':>8s} {'cold':>8s}")

        def report(name: str, fn: Any, reps: int = 50) -> None:
            def batch() -> None:
                for _ in range(reps):
                    fn()

            w = warm(batch).median_ms / reps * 1e3
            c = cold(fn).median_ms * 1e3
            print(f"  {name:>28s} {w:8.1f} {c:8.1f}")

        for v, name in VARIANTS.items():
            report(name, lambda v=v: variants().run(v, checker.c, checker.a, x, out))  # noqa: B023
        report("production checker", lambda: checker.check(x, y))  # noqa: B023


if __name__ == "__main__":
    main()
