"""Top-k / top-p sampling: reference samplers and the trusted O(vocabulary) checker.

The checker inspects a sampler's output against the stored FP32 logits:

* top-k: exactly k distinct in-range indices, and every selected logit is >= every
  rejected logit. Comparisons are exact, so no tolerance is needed; ties may be broken
  either way. Any NaN logit fails.
* top-p: the same prefix property, and the nucleus is minimal: mass(S) >= p - tol and
  mass(S without its smallest logit) < p + tol, with masses from an FP64 softmax. ``tol``
  is the sampler's declared probability error, so a correct sampler that rounds its own
  softmax differently near the boundary is not flagged. The checker adds its own FP64
  rounding slack, gamma_{V+2}(u64) for a vocabulary of V, since masses are at most 1.
* the sampled token is a member of the selected set.

Not checked: that the token was drawn from the right distribution.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pck.fp import U64, gamma

Indices = NDArray[np.int64]


def _as_logits(logits: ArrayLike) -> NDArray[np.float32]:
    a = np.asarray(logits, dtype=np.float32)
    if a.ndim != 1 or a.size == 0:
        raise ValueError("logits must be a non-empty 1-D array")
    return a


def topk_reference(logits: ArrayLike, k: int) -> Indices:
    """The k largest logits (ties broken by index), in no particular order."""
    a = _as_logits(logits)
    if not 1 <= k <= a.size:
        raise ValueError(f"k must be in [1, {a.size}], got {k}")
    order = np.lexsort((np.arange(a.size), -a.astype(np.float64)))
    return order[:k].astype(np.int64)


def softmax64(logits: ArrayLike, temperature: float = 1.0) -> NDArray[np.float64]:
    """Softmax of logits / temperature, computed in FP64."""
    if not temperature > 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    z = _as_logits(logits).astype(np.float64) / temperature
    e = np.exp(z - np.max(z))
    p: NDArray[np.float64] = e / e.sum()
    return p


def topp_reference(logits: ArrayLike, p: float, temperature: float = 1.0) -> Indices:
    """The smallest set of largest logits whose probability mass reaches p."""
    if not 0 < p <= 1:
        raise ValueError(f"p must be in (0, 1], got {p}")
    a = _as_logits(logits)
    probs = softmax64(a, temperature)
    order = np.lexsort((np.arange(a.size), -a.astype(np.float64)))
    mass = np.cumsum(probs[order])
    count = int(np.searchsorted(mass, p)) + 1
    return order[: min(count, a.size)].astype(np.int64)


def _selection_mask(size: int, selected: ArrayLike) -> NDArray[np.bool_] | None:
    """Boolean mask of the selection, or None if it is not a set of in-range indices."""
    s = np.asarray(selected)
    if s.ndim != 1 or s.size == 0 or not np.issubdtype(s.dtype, np.integer):
        return None
    if s.min() < 0 or s.max() >= size or np.unique(s).size != s.size:
        return None
    mask = np.zeros(size, dtype=bool)
    mask[s] = True
    return mask


def _is_prefix(a: NDArray[np.float32], mask: NDArray[np.bool_]) -> bool:
    """Every selected logit is >= every rejected logit (NaN fails)."""
    if np.isnan(a).any():
        return False
    if mask.all():
        return True
    return bool(np.min(a[mask]) >= np.max(a[~mask]))


def check_topk(logits: ArrayLike, selected: ArrayLike, k: int) -> bool:
    """True if ``selected`` is a valid top-k set for ``logits``."""
    a = _as_logits(logits)
    mask = _selection_mask(a.size, selected)
    return mask is not None and int(mask.sum()) == k and _is_prefix(a, mask)


def check_topp(
    logits: ArrayLike,
    selected: ArrayLike,
    p: float,
    temperature: float = 1.0,
    tol: float = 1e-6,
) -> bool:
    """True if ``selected`` is a valid (minimal) top-p nucleus for ``logits``."""
    if not 0 < p <= 1:
        raise ValueError(f"p must be in (0, 1], got {p}")
    if tol < 0:
        raise ValueError(f"tol must be non-negative, got {tol}")
    a = _as_logits(logits)
    mask = _selection_mask(a.size, selected)
    if mask is None or not _is_prefix(a, mask):
        return False
    probs = softmax64(a, temperature)
    mass = float(probs[mask].sum())
    sel = np.flatnonzero(mask)
    smallest = sel[np.argmin(a[sel])]
    without_smallest = mass - float(probs[smallest])
    slack = tol + gamma(a.size + 2, U64)  # the sampler's tolerance plus our own rounding
    return mass >= p - slack and without_smallest < p + slack


def check_sampled(token: int, selected: ArrayLike) -> bool:
    """True if the sampled ``token`` is in the selected set."""
    return bool(np.isin(token, np.asarray(selected)))
