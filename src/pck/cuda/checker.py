"""Trusted blocked checksum checker on the GPU (production mode).

The checksum bank (secret signs r, checksums c = r^T W_beta and a_beta = sum |W_i|) is
built once at weight load by this trusted code. In production the fault model is
non-adversarial, so a bank drawn ahead of time is fine; RL evaluation must instead draw
the signs after the untrusted kernel has finished (commit-then-challenge).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from pck.cuda._ext import extension
from pck.fp import U32, U64, gamma


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
    """

    def __init__(
        self,
        W: torch.Tensor,
        block: int,
        height: int,
        checks: int = 1,
        generator: torch.Generator | None = None,
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

        a64 = W64.abs().sum(dim=1)
        a32 = a64.float()
        up = torch.nextafter(a32, torch.full_like(a32, math.inf))
        self.a = torch.where(a32.double() < a64, up, a32)

        self.coef = _next_up(gamma(height) + U32, 2)
        self.geps = _next_up(gamma(n + block + 1, U64), 2)

    def check(self, x: torch.Tensor, y: torch.Tensor) -> CheckResult:
        """Run every check on output ``y`` of the GEMV for input ``x``."""
        passed, rho, tau = extension().blocked_check(
            self.c, self.a, self.r, x, y, self.coef, self.geps
        )
        return CheckResult(passed, rho, tau)

    def verdict(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Per-block pass/fail (True = passed every check)."""
        return self.check(x, y).verdict
