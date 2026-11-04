"""Trusted blocked checksum checker on the GPU (production mode).

The checksum bank (secret signs r, FP32 checksums c = r^T W_beta, and a_beta = sum |W_i|
rounded up to BF16) is built once at weight load by this trusted code. In production the
fault model is non-adversarial, so a bank drawn ahead of time is fine; RL evaluation must
instead draw the signs after the untrusted kernel has finished (commit-then-challenge).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from pck.cuda._ext import extension
from pck.fp import U32, U64, ceil_div, gamma

KERNEL_THREADS = 256  # threads per block in checker_kernel.cu
BLOCKS_PER_SM = 4
MIN_SEGMENT_COLUMNS = 2 * KERNEL_THREADS


def round_up_bf16(v: torch.Tensor) -> torch.Tensor:
    """Round non-negative FP64 values to BF16 without ever rounding down.

    Rounds up to FP32 first, then up to BF16 by carrying into the top 16 bits whenever
    any of the low 16 bits are set. Values past the BF16 range become +Inf.
    """
    if v.dtype != torch.float64:
        raise TypeError(f"round_up_bf16 expects float64, got {v.dtype}")
    if bool((v < 0).any()):
        raise ValueError("round_up_bf16 expects non-negative values")
    f = v.float()
    f = torch.where(f.double() < v, torch.nextafter(f, torch.full_like(f, math.inf)), f)
    bits = f.view(torch.int32)
    bits = torch.where((bits & 0xFFFF) != 0, (bits + 0x10000) & -0x10000, bits)
    return bits.view(torch.float32).to(torch.bfloat16)  # exact: low 16 bits are zero


def _next_up(v: float, steps: int = 1) -> float:
    for _ in range(steps):
        v = math.nextafter(v, math.inf)
    return v


@dataclass(frozen=True)
class CheckResult:
    """Per (check, block) outcome; each tensor has shape (checks, num_blocks)."""

    passed: torch.Tensor
    rho: torch.Tensor
    tau: torch.Tensor

    @property
    def verdict(self) -> torch.Tensor:
        """Per-block verdict: True only if the block passed every check."""
        return self.passed.all(dim=0)


class GpuBlockedChecker:
    """Checks y = W x one block of ``block`` rows at a time, with ``checks`` sign draws.

    ``height`` is the summation-tree height of the kernel being checked, e.g.
    ``tree_height(n, threads, vec)``; the threshold is proven never to false-alarm for any
    FP32 kernel within that height.

    ``segments`` splits each block's n columns across that many CUDA blocks so the grid
    fills the GPU; by default about ``BLOCKS_PER_SM`` blocks per SM. A checker holds
    per-call scratch space, so use it from one stream at a time.
    """

    def __init__(
        self,
        W: torch.Tensor,
        block: int,
        height: int,
        checks: int = 1,
        generator: torch.Generator | None = None,
        segments: int | None = None,
    ) -> None:
        if not (W.is_cuda and W.dtype == torch.bfloat16 and W.dim() == 2):
            raise ValueError("W must be a 2-D bfloat16 CUDA tensor")
        m, n = W.shape
        if block < 1 or m % block:
            raise ValueError(f"block={block} must be positive and divide m={m}")
        if checks < 1:
            raise ValueError(f"checks must be positive, got {checks}")
        self.m, self.n, self.block, self.checks = m, n, block, checks
        self.num_blocks = m // block

        bits = torch.randint(0, 2, (checks, self.num_blocks, block), generator=generator)
        self.r = (2 * bits - 1).to(torch.int8).to(W.device)

        W64 = W.view(self.num_blocks, block, n).double()
        c = torch.empty((checks, self.num_blocks, n), dtype=torch.float32, device=W.device)
        for k in range(checks):  # one check at a time bounds the FP64 temporaries
            c[k] = torch.einsum("bi,bin->bn", self.r[k].double(), W64).float()
        self.c = c

        self.a = round_up_bf16(W64.abs().sum(dim=1))
        if not bool(torch.isfinite(self.a).all()):
            raise ValueError("sum of |W| over a block overflows BF16")

        self.coef = _next_up(gamma(height) + U32, 2)

        if segments is None:
            sms = torch.cuda.get_device_properties(W.device).multi_processor_count
            wanted = ceil_div(BLOCKS_PER_SM * sms, checks * self.num_blocks)
            segments = max(1, min(wanted, n // MIN_SEGMENT_COLUMNS))
        if not 1 <= segments <= min(n, 65535):
            raise ValueError(f"segments must be in [1, {min(n, 65535)}], got {segments}")
        self.segments = segments
        # Terms per thread chain in the compensated FP32 dot product (Dot2): L.
        chain = ceil_div(ceil_div(n, segments), KERNEL_THREADS)
        self.geps = _next_up(gamma(n + block + 1, U64) + gamma(2 * chain, U32) ** 2, 4)
        self.floor = _next_up(n * 2.0**-148, 2)  # TwoProduct inexactness under underflow
        scratch = (checks, self.num_blocks, segments)
        self._partial_cx = torch.empty(scratch, dtype=torch.float64, device=W.device)
        self._partial_s = torch.empty(scratch, dtype=torch.float64, device=W.device)
        self._counters = torch.zeros((checks, self.num_blocks), dtype=torch.int32, device=W.device)

    def check(self, x: torch.Tensor, y: torch.Tensor) -> CheckResult:
        """Run every check on output ``y`` of the GEMV for input ``x``."""
        passed, rho, tau = extension().blocked_check(
            self.c,
            self.a,
            self.r,
            x,
            y,
            self.coef,
            self.geps,
            self.floor,
            self._partial_cx,
            self._partial_s,
            self._counters,
        )
        return CheckResult(passed, rho, tau)

    def verdict(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Per-block pass/fail (True = passed every check)."""
        return self.check(x, y).verdict


class PendingCheck:
    """A check queued on a side stream; :meth:`wait` makes the current stream wait for it."""

    def __init__(self, result: CheckResult, done: torch.cuda.Event) -> None:
        self._result = result
        self._done = done

    def wait(self) -> CheckResult:
        stream = torch.cuda.current_stream()
        stream.wait_event(self._done)
        for t in (self._result.passed, self._result.rho, self._result.tau):
            t.record_stream(stream)
        return self._result


class SideStreamChecker:
    """Runs checks on a dedicated CUDA stream so they overlap later work.

    Each check still starts only after the kernel that produced ``y`` has finished: the side
    stream waits on an event recorded on the caller's stream at submit time. Checks on one
    side stream run in submission order, so a ``GpuBlockedChecker``'s scratch space is never
    used by two checks at once.
    """

    def __init__(self, device: torch.device | None = None) -> None:
        self.stream = torch.cuda.Stream(device)

    def submit(self, checker: GpuBlockedChecker, x: torch.Tensor, y: torch.Tensor) -> PendingCheck:
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(ready)
            x.record_stream(self.stream)  # keep the allocator from reusing x and y early
            y.record_stream(self.stream)
            result = checker.check(x, y)
            done = torch.cuda.Event()
            done.record(self.stream)
        return PendingCheck(result, done)

    def wait_all(self) -> None:
        """Make the current stream wait for every check submitted so far."""
        torch.cuda.current_stream().wait_stream(self.stream)
