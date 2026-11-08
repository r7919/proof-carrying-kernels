"""Membership checks for samplers that return a token, not a set (vLLM's FlashInfer path).

For one row of logits ``l`` (after temperature and penalties), the sampled token ``t``,
and the request's ``top_k`` / ``top_p`` (vLLM's order: top-k first, then top-p over the
survivors, renormalized):

  top-k   accept iff fewer than ``k`` logits are strictly larger than ``l[t]``.
          Exact: `topK_member_iff` (lean/PCK/TopMember.lean) shows this is the same as
          ``t`` belonging to some correct top-k set.
  top-p   K = tokens with logit >= the k-th largest (all if no top-k); r = fraction of
          K's softmax mass strictly above ``t``. Accept iff the computed r is below
          ``p + slack``, slack = ``tol + eps``: `topP_member_sound` (a sample within the
          sampler's own tolerance ``tol`` always passes) and `topP_member_detects` (a
          rejected one has r >= p + tol), with ``eps`` from `member_eps`.
  greedy  accept iff no logit is strictly larger than ``l[t]`` (top-k with k = 1).

``tol`` is the sampler's own allowance (FlashInfer computes its pivots in FP32); it is a
parameter, not something proven about the sampler. NaN logits or a NaN mass reject.

`check_member` is the CPU reference (NumPy, FP64); `check_members` is the batched torch
version that runs inside vLLM (`pck.vllm_plugin`), on the GPU: exact FP32 comparisons, FP32
softmax terms on an exact split argument summed in FP64, with ``eps`` from `member_eps32`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pck.fp import U32, U64, gamma


def member_eps(vocab: int) -> float:
    """Bound on |computed r - r| in FP64, for any summation order (`gamma` of the proven
    tree bound), FP64 ``exp`` within 2 ulp and one rounding per subtraction and division.

    Each term ``exp(fl(l_j - m))`` is off by at most ``u (|x| e^x + 2 e^x) <= u (1/e + 2 e_j)``
    absolutely (x = l_j - m <= 0); the sum has the maximum's term exactly 1, so both sums
    (kept mass Z >= 1, mass above A <= Z) are within ``E Z``, E = V u (1/e + 0.01) + 2u +
    gamma_V(u) (1 + 3u); the quotient is then within ``2E / (1 - E) + u``."""
    u = U64
    e = vocab * u * (1 / math.e + 0.01) + 2 * u + gamma(vocab, u) * (1 + 3 * u)
    return 2 * e / (1 - e) + u


@dataclass(frozen=True)
class MemberVerdict:
    ok: bool
    above: int
    """Logits strictly larger than the sampled token's."""
    mass_above: float
    """Computed fraction of kept mass strictly above the token (0 without top-p)."""


def check_member(
    logits: NDArray[np.floating],
    token: int,
    top_k: int | None = None,
    top_p: float | None = None,
    *,
    greedy: bool = False,
    tol: float = 1e-4,
) -> MemberVerdict:
    """The CPU reference check for one row (see the module docstring)."""
    lv = np.asarray(logits, dtype=np.float64)
    if np.isnan(lv).any():
        return MemberVerdict(False, -1, math.nan)
    lt = lv[token]
    above = int(np.count_nonzero(lv > lt))
    k = 1 if greedy else top_k
    if k is not None and above >= k:
        return MemberVerdict(False, above, math.nan)
    if greedy or top_p is None or top_p >= 1.0:
        return MemberVerdict(True, above, 0.0)
    kept = np.ones(lv.shape, dtype=bool)
    if top_k is not None and top_k < lv.size:
        kth = np.partition(lv, lv.size - top_k)[lv.size - top_k]
        kept = lv >= kth
    m = lv[kept].max()
    e = np.where(kept, np.exp(lv - m), 0.0)
    r = float(e[lv > lt].sum() / e.sum())
    slack = tol + member_eps(lv.size)
    return MemberVerdict(bool(r < top_p + slack), above, r)


def member_eps32(vocab: int) -> float:
    """Bound on |computed r - r| for `check_members`: FP32 terms, FP64 sums.

    Each kept term ``e^(l_j - m)`` (l_j, m FP32, l_j <= m) comes from the exact split
    ``l_j - m = hi + lo`` (TwoSum in FP32, exact without overflow, |lo| <= u32 |hi|) as
    ``fl(expf(hi) + fl(expf(hi) lo))``; terms with ``hi < -128`` are dropped. With FP32
    ``expf`` within 2 ulp (4 u32 relative), ``|e^lo - 1 - lo| <= lo^2 <= (128 u32)^2`` and
    two roundings, each term is within ``6 u32`` relative plus ``2^-124`` absolute (subnormal
    or flushed results, and dropped terms below ``e^-128``). The sums are FP64 over at most
    V terms plus one product (the count of tokens tied at the k-th logit times their term),
    within ``gamma_{V+2}(u64)``. The largest true term is 1, so Z >= 1, and both sums (kept
    mass Z, mass above A <= Z) are within ``E Z`` with ``E = 6 u32 + V 2^-124 +
    gamma_{V+2}(u64) (1 + 6 u32 + V 2^-124)``; the quotient is then within
    ``2E / (1 - E) + u64``, as in `member_eps`."""
    tiny = math.ldexp(vocab, -124)
    e = 6 * U32 + tiny + gamma(vocab + 2, U64) * (1 + 6 * U32 + tiny)
    return 2 * e / (1 - e) + U64


def _terms(x: Any, m: Any) -> Any:  # pragma: no cover (torch)
    """FP32 ``e^(x - m)`` for ``x <= m`` (row maxima ``m``), from the exact split
    ``x - m = hi + lo`` (Knuth's TwoSum); 0 where ``hi < -128`` or NaN (see `member_eps32`)."""
    import torch

    nm = -m
    hi = x + nm
    bv = hi - x
    lo = (x - (hi - bv)) + (nm - bv)
    eh = torch.exp(hi)
    return torch.where(hi >= -128.0, eh + eh * lo, 0.0)


def check_members(  # pragma: no cover (torch: tests/test_member.py, where it is installed)
    logits: Any,
    tokens: Any,
    top_k: Any = None,
    top_p: Any = None,
    greedy: Any = None,
    tol: float = 1e-4,
    top_k_host: Any = None,
) -> dict[str, Any]:
    """`check_member` for a batch on the logits' device (torch): rows (B, V), tokens (B,),
    optional per-row ``top_k`` (B,) ints (V = no top-k), ``top_p`` (B,) floats (1 = none)
    and ``greedy`` (B,) bools. Returns tensors ``ok`` (B,) bool, ``above`` (B,) int and
    ``mass_above`` (B,) float64.

    ``top_k_host`` is the same top-k values on the host (NumPy; any array holding the same
    set of values, e.g. one per request); with it the check never synchronizes the device.
    Without it they are copied from ``top_k`` (one synchronization).

    Comparisons and top-k are exact in FP32. The top-p fraction uses FP32 terms summed in
    FP64 (`member_eps32`). When every row has a top-k, it is computed over the top
    ``max k`` logits only, plus the count of tokens tied at the k-th logit beyond them."""
    import torch

    lv = logits.float()  # exact for FP16 / BF16 / FP32
    rows, vocab = lv.shape
    f64 = torch.float64
    t = tokens.reshape(rows, 1).to(torch.long)
    lt = lv.gather(1, t)
    gt = lv > lt
    above = gt.sum(1)
    finite = ~torch.isnan(lv).any(1)
    k = torch.full((rows,), vocab, dtype=torch.long, device=lv.device)
    k_lim, all_k = 0, False  # largest top-k below V, and whether every row has one
    if top_k is not None:
        k = top_k.to(torch.long).clamp(1, vocab)
        host = top_k_host if top_k_host is not None else top_k.cpu().numpy()
        hk = np.clip(np.asarray(host).astype(np.int64), 1, vocab)
        lim = hk[hk < vocab]
        k_lim, all_k = (int(lim.max()), lim.size == hk.size) if lim.size else (0, False)
    if greedy is not None:
        is_greedy = greedy.to(torch.bool)
        ok_k = torch.where(is_greedy, torch.ones_like(k), k)
    else:
        is_greedy = torch.zeros(rows, dtype=torch.bool, device=lv.device)
        ok_k = k
    ok = finite & (above < ok_k)
    mass_above = torch.zeros(rows, dtype=f64, device=lv.device)
    if top_p is not None:
        p = top_p.to(f64)
        if all_k:
            vals = lv.topk(k_lim, dim=1).values  # descending
            kth = vals.gather(1, (k.clamp(max=k_lim) - 1).reshape(rows, 1))
            in_k = vals >= kth
            m = vals[:, :1]
            e = torch.where(in_k, _terms(vals, m), 0.0)
            ties = (lv >= kth).sum(1) - in_k.sum(1)  # kept tokens at the k-th logit, not in vals
            e_kth = e.gather(1, (k - 1).reshape(rows, 1)).reshape(rows)  # vals[k-1] is kth
            z = e.sum(1, dtype=f64) + ties * e_kth.to(f64)
            a = torch.where(vals > lt, e, 0.0).sum(1, dtype=f64)
        else:
            x = lv
            if k_lim:
                kth = lv.topk(k_lim, dim=1).values.gather(
                    1, (k.clamp(max=k_lim) - 1).reshape(rows, 1)
                )
                kept = (k < vocab).reshape(rows, 1).logical_not() | (lv >= kth)
                x = torch.where(kept, lv, torch.full_like(lv, -math.inf))
            m = x.max(1, keepdim=True).values
            e = _terms(x, m)
            z = e.sum(1, dtype=f64)
            a = torch.where(gt, e, 0.0).sum(1, dtype=f64)
        r = a / z
        mass_above = torch.where(p < 1.0, r, 0.0)
        slack = tol + member_eps32(vocab)
        topp_ok = (p >= 1.0) | is_greedy | (r < p + slack)
        ok = ok & topp_ok
    return {"ok": ok, "above": above, "mass_above": mass_above}
