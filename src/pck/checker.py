"""Blocked checksum checker for y = W x, using the arithmetic of the trusted kernel."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pck.fp import U32, U64, gamma, next_up, round_up_f32, sum_up


class BlockedChecker:
    """One randomized checksum per block of ``block`` output rows.

    For each block beta and secret signs r in {+-1}^block:
      rho = (r^T W_beta) . x - r . y_beta              (residual, FP64)
      tau = (gamma_h + u) * (a_beta . |x|) + eps       (threshold)
    where a_beta = sum_{i in beta} |W_i| is stored rounded up and eps covers the
    FP64 rounding of the residual. Every part of tau is rounded up (T3's hypotheses: an
    upper bound on a_beta . |x|, a coefficient and threshold never below their exact
    values). A block passes only if every output in it is finite, tau is finite (an
    overflowing threshold fails closed) and |rho| <= tau, so NaN or Inf always raises
    the alarm.
    """

    def __init__(
        self, W: NDArray[np.floating], x: NDArray[np.floating], block: int, height: int
    ) -> None:
        m, n = W.shape
        if x.shape != (n,):
            raise ValueError(f"x has shape {x.shape}, expected ({n},)")
        if block < 1 or m % block:
            raise ValueError(f"block={block} must be positive and divide m={m}")
        self.m, self.n = m, n
        self.block = block
        self.num_blocks = m // block
        self.W_blocks = W.reshape(self.num_blocks, block, n).astype(np.float64)
        self.x = x.astype(np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            a = sum_up(np.abs(self.W_blocks).sum(axis=1), block)
            a_up = round_up_f32(a)  # past FP32 range: inf, so tau is inf and fails
            self.abs_bound: NDArray[np.float64] = sum_up(a_up @ np.abs(self.x), n)
        """An upper bound on a_beta . |x| for every block."""
        self.gamma_h = gamma(height)
        self.coef = next_up(next_up(self.gamma_h) + U32)
        """gamma_h + u, rounded up."""

    @property
    def tau_base(self) -> NDArray[np.float64]:
        """The data-independent part of tau per block: (gamma_h + u) * (a_beta . |x|),
        rounded up."""
        with np.errstate(over="ignore", invalid="ignore"):
            return np.nextafter(self.coef * self.abs_bound, np.inf)

    def verdict(
        self, y: NDArray[np.floating], rng: np.random.Generator, checks: int = 1
    ) -> NDArray[np.bool_]:
        """Per-block pass/fail for output ``y`` after ``checks`` independent sign draws.

        True means the block passed. Sign vectors come from ``rng``, which must be
        independent of how ``y`` was produced.
        """
        if y.shape != (self.m,):
            raise ValueError(f"y has shape {y.shape}, expected ({self.m},)")
        with np.errstate(invalid="ignore", over="ignore"):
            yb = y.astype(np.float64).reshape(self.num_blocks, self.block)
        ok = np.isfinite(yb).all(axis=1)
        for _ in range(checks):
            r = rng.choice([-1.0, 1.0], (self.num_blocks, self.block))
            c = np.einsum("kb,kbn->kn", r, self.W_blocks).astype(np.float32).astype(np.float64)
            with np.errstate(invalid="ignore", over="ignore"):
                rho = c @ self.x - (r * yb).sum(axis=1)
                mags = sum_up(np.abs(c) @ np.abs(self.x), self.n) + sum_up(
                    np.abs(yb).sum(axis=1), self.block
                )
                eps = np.nextafter(next_up(gamma(self.n + self.block + 1, U64)) * mags, np.inf)
                tau = np.nextafter(self.tau_base + eps, np.inf)
                ok &= np.isfinite(tau) & (np.abs(rho) <= tau)  # NaN compares False
        return ok
