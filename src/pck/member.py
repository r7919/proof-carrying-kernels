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
version that runs inside vLLM (`pck.vllm_plugin`), on the GPU, also in FP64.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pck.fp import U64, gamma


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


def check_members(
    logits: Any,
    tokens: Any,
    top_k: Any = None,
    top_p: Any = None,
    greedy: Any = None,
    tol: float = 1e-4,
) -> dict[str, Any]:
    """`check_member` for a batch on the logits' device (torch): rows (B, V), tokens (B,),
    optional per-row ``top_k`` (B,) ints (V = no top-k), ``top_p`` (B,) floats (1 = none)
    and ``greedy`` (B,) bools. Returns tensors ``ok`` (B,) bool, ``above`` (B,) int and
    ``mass_above`` (B,) float64, without synchronizing the device."""
    import torch

    lv = logits.to(torch.float64)
    rows, vocab = lv.shape
    t = tokens.reshape(rows, 1).to(torch.long)
    lt = lv.gather(1, t)
    gt = lv > lt
    above = gt.sum(1)
    finite = ~torch.isnan(lv).any(1)
    k = torch.full((rows,), vocab, dtype=torch.long, device=lv.device)
    if top_k is not None:
        k = top_k.to(torch.long).clamp(1, vocab)
    if greedy is not None:
        k = torch.where(greedy.to(torch.bool), torch.ones_like(k), k)
    ok = finite & (above < k)
    mass_above = torch.zeros(rows, dtype=torch.float64, device=lv.device)
    if top_p is not None:
        p = top_p.to(torch.float64)
        kept = torch.ones_like(gt)
        kk = top_k.to(torch.long).clamp(1, vocab) if top_k is not None else None
        if kk is not None and bool((kk < vocab).any()):
            kmax = int(kk.max())
            values = lv.topk(kmax, dim=1).values
            kth = values.gather(1, (kk - 1).reshape(rows, 1))
            kept = torch.where((kk < vocab).reshape(rows, 1), lv >= kth, kept)
        masked = torch.where(kept, lv, torch.full_like(lv, -math.inf))
        m = masked.max(1, keepdim=True).values
        e = torch.exp(masked - m)
        r = (e * gt).sum(1) / e.sum(1)
        mass_above = torch.where(p < 1.0, r, torch.zeros_like(r))
        slack = tol + member_eps(vocab)
        is_greedy = greedy.to(torch.bool) if greedy is not None else torch.zeros_like(ok)
        topp_ok = (p >= 1.0) | is_greedy | (r < p + slack)
        ok = ok & topp_ok
    return {"ok": ok, "above": above, "mass_above": mass_above}
