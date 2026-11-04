"""Benchmark the naive BF16 GEMV against cuBLAS (torch.mv) on decode-sized shapes.

Times with CUDA events after warm-up and reports medians. Run plugged in, in
high-performance mode, and log clocks with `nvidia-smi` alongside.

Run: uv run --extra gpu python harness/bench_gemv.py
"""

from __future__ import annotations

import statistics
from collections.abc import Callable

import torch

from pck.cuda import gemv_naive

SHAPES = [(896, 896), (1536, 1536), (2048, 2048), (3584, 3584), (4096, 4096), (14336, 4096)]
WARMUP, REPS = 20, 200


def median_ms(fn: Callable[[], torch.Tensor]) -> float:
    for _ in range(WARMUP):
        fn()
    times: list[float] = []
    for _ in range(REPS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
    return statistics.median(times)


def main() -> None:
    torch.manual_seed(0)
    print(f"device: {torch.cuda.get_device_name()}")
    print(
        f"{'m x n':>14s} {'cuBLAS ms':>10s} {'naive ms':>10s} {'speedup':>8s} {'naive GB/s':>11s}"
    )
    for m, n in SHAPES:
        W = torch.randn(m, n, device="cuda").to(torch.bfloat16)
        x = torch.randn(n, device="cuda").to(torch.bfloat16)
        cublas = median_ms(lambda: torch.mv(W, x))  # noqa: B023
        naive = median_ms(lambda: gemv_naive(W, x, 256))  # noqa: B023
        gbytes = (W.numel() * 2 + n * 2 + m * 4) / 1e9
        print(
            f"{f'{m} x {n}':>14s} {cublas:10.4f} {naive:10.4f} "
            f"{cublas / naive:7.2f}x {gbytes / (naive / 1e3):11.1f}"
        )


if __name__ == "__main__":
    main()
