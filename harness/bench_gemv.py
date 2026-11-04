"""Benchmark the BF16 GEMV kernels against cuBLAS (torch.mv) on decode-sized shapes.

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

from pck.cuda import gemv_coalesced, gemv_naive

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
    return float(2 * src.numel() / (t.median_ms / 1e3) / 1e9)


Gemv = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

# name -> (function, bytes written per output element)
KERNELS: dict[str, tuple[Gemv, int]] = {
    "cuBLAS": (torch.mv, 2),  # BF16 output
    "naive/256": (lambda W, x: gemv_naive(W, x, 256), 4),
    "coalesced/128": (lambda W, x: gemv_coalesced(W, x, 128), 4),
    "coalesced/256": (lambda W, x: gemv_coalesced(W, x, 256), 4),
}


def check_correct(fn: Gemv, W: torch.Tensor, x: torch.Tensor) -> None:
    ref = W.float() @ x.float()
    torch.testing.assert_close(fn(W, x).float(), ref, rtol=2e-2, atol=2e-2)


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
    print(f"measured copy bandwidth: {copy_bw:.0f} GB/s (= 100% below)")
    if args.peak_gbps:
        print(f"spec-sheet bandwidth: {args.peak_gbps:.0f} GB/s")
    print("times: median ms [p10, p90], cold L2; 'vs cuBLAS' > 1 means faster")
    print(
        f"{'m x n':>12s} {'kernel':>14s} {'ms [p10, p90]':>24s} {'GB/s':>6s}"
        f" {'%copy':>6s} {'vs cuBLAS':>9s}"
    )
    for m, n in SHAPES:
        W = torch.randn(m, n, device="cuda").to(torch.bfloat16)
        x = torch.randn(n, device="cuda").to(torch.bfloat16)
        cublas_ms = None
        for name, (fn, out_bytes) in KERNELS.items():
            check_correct(fn, W, x)
            t = timer(lambda: fn(W, x))  # noqa: B023
            cublas_ms = cublas_ms or t.median_ms
            gb = (2 * m * n + 2 * n + out_bytes * m) / 1e9
            bw = gb / (t.median_ms / 1e3)
            times = f"{t.median_ms:.4f} [{t.p10_ms:.4f}, {t.p90_ms:.4f}]"
            print(
                f"{f'{m} x {n}':>12s} {name:>14s} {times:>24s} {bw:6.0f}"
                f" {bw / copy_bw:6.0%} {cublas_ms / t.median_ms:8.2f}x"
            )


if __name__ == "__main__":
    main()
