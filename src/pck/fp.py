"""Floating-point helpers: BF16 rounding, unit roundoffs and rounding-error constants."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

U32: float = 2.0**-24
"""Unit roundoff of FP32 (round to nearest)."""

U64: float = 2.0**-53
"""Unit roundoff of FP64 (round to nearest)."""


def bf16(a: ArrayLike) -> NDArray[np.float32]:
    """Round FP32 values to BF16 (round to nearest, ties to even), returned as FP32.

    NaN stays NaN; values past the BF16 range round to +/-Inf, as on hardware.
    """
    f = np.array(a, dtype=np.float32, copy=True)
    nan = np.isnan(f)
    bits = f.view(np.uint32)
    bits += np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    bits &= np.uint32(0xFFFF0000)
    f[nan] = np.nan
    return f


def gamma(k: int, u: float = U32) -> float:
    """Higham's gamma_k = k*u / (1 - k*u): the error constant for k chained roundings."""
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    if k * u >= 1:
        raise ValueError(f"gamma_k needs k*u < 1, got k*u = {k * u}")
    return k * u / (1 - k * u)


def round_up_f32(v: ArrayLike) -> NDArray[np.float64]:
    """Store non-negative FP64 values as FP32 without ever rounding down.

    Returns the FP32 values widened back to FP64, so callers keep FP64 arithmetic.
    """
    v64 = np.asarray(v, dtype=np.float64)
    if np.any(v64 < 0):
        raise ValueError("round_up_f32 expects non-negative values")
    f = v64.astype(np.float32)
    low = f.astype(np.float64) < v64
    f[low] = np.nextafter(f[low], np.float32(np.inf))
    return f.astype(np.float64)


def next_up(v: float, steps: int = 1) -> float:
    """``v`` moved up by ``steps`` representable FP64 values."""
    for _ in range(steps):
        v = math.nextafter(v, math.inf)
    return v


def sum_up_factor(terms: int) -> float:
    """A factor that turns a round-to-nearest FP64 sum of ``terms`` non-negative values
    (any order) into an upper bound on the exact sum: with |s - S| <= gamma_{terms} S,
    S <= s / (1 - gamma) <= s (1 + 2 gamma) for gamma <= 1/2. Rounded up; multiply by it,
    then round the product up once (`sum_up`)."""
    return next_up(1.0 + 2.0 * gamma(max(terms, 1), U64))


def sum_up(s: ArrayLike, terms: int) -> NDArray[np.float64]:
    """An upper bound on the exact sums whose FP64 round-to-nearest values are ``s``."""
    s64 = np.asarray(s, dtype=np.float64)
    with np.errstate(over="ignore"):
        return np.nextafter(s64 * sum_up_factor(terms), np.inf)


def softmax_eps(vocab: int, zmax: float = 0.0) -> float:
    """Bound on |computed - exact| for a mass of an FP64 softmax over ``vocab`` logits: a
    fraction A / S of kept terms, computed either as one quotient or as a sum of per-term
    probabilities, and also covering one subtraction of a probability from a mass.

    The terms are ``e_j = exp(x_j)`` with ``x_j = fl(z_j - max z)``, ``z = fl(l / T)``;
    ``zmax`` bounds |z_j| over finite logits when the division is inexact (0 for T = 1 or
    for logits used directly). Each argument is off by at most ``u |x_j|`` (the subtraction)
    plus ``2 zmax u`` (both divisions); with ``exp`` within 2 ulp (4u relative), each term is
    within ``1.01 (u |x| e^x + (2 zmax u + 4u) e^x) <= 1.01 (u / e + (2 zmax u + 4u) e_j)``
    (1.01 covers ``e^d - 1 <= d (1 + d)`` for d <= 0.01). The largest true term is 1, so the
    true sum S >= 1, and with any summation order S is computed within ``E S``,
    ``E = 1.01 (V u / e + 2 zmax u + 4u) + 1.01 gamma_V(u)`` (E <= 0.01). A mass is then
    within ``(2E + gamma_{V+2}(u) (1 + E)) / (1 - E)``, the gamma covering per-term divisions
    and the sum of probabilities, plus 2u for one subtraction."""
    u = U64
    v = max(vocab, 1)
    e = 1.01 * (v * u / math.e + 2.0 * zmax * u + 4.0 * u) + 1.01 * gamma(v, u)
    if not e <= 0.01:
        raise ValueError(
            f"softmax over {vocab} logits with |logit / T| up to {zmax} is too "
            "large to check in FP64"
        )
    return (2.0 * e + gamma(v + 2, u) * (1.0 + e)) / (1.0 - e) + 2.0 * u


def ceil_div(a: int, b: int) -> int:
    """ceil(a / b) for non-negative a and positive b."""
    return -(-a // b)


def tree_height(n: int, threads: int, vec: int = 1) -> int:
    """Height of the summation tree used by the GEMV kernel models.

    Each of ``threads`` threads sums its terms sequentially, then a binary tree of
    height log2(threads) combines the threads. With ``vec = 1`` a thread owns a
    contiguous chunk of ceil(n / threads) columns (``emulate_gemv``). With ``vec > 1``
    a thread owns every ``threads``-th vector of ``vec`` columns
    (``emulate_gemv_coalesced``), so its chain holds vec * ceil(ceil(n / vec) / threads)
    terms, counting zero padding.
    """
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    if threads < 1 or threads & (threads - 1):
        raise ValueError(f"threads must be a power of two, got {threads}")
    if vec < 1:
        raise ValueError(f"vec must be positive, got {vec}")
    chunk = vec * ceil_div(ceil_div(n, vec), threads)
    return (chunk - 1) + threads.bit_length() - 1
