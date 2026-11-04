"""Benchmark timing on a cold L2 cache."""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass

import torch


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
    return float(2 * src.numel() / (t.median_ms / 1e3) / 1e9)
