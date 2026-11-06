"""Membership checks for token-only samplers (pck/member.py)."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pck.member import check_member, member_eps


def vllm_kept(logits: np.ndarray, k: int | None, p: float | None) -> set[int]:
    """vLLM's reference masking (apply_top_k_top_p_pytorch), in NumPy: top-k keeps logits
    >= the k-th largest; top-p sorts ascending and drops tokens whose cumulative mass is
    <= 1 - p, always keeping the largest."""
    lv = logits.astype(np.float64).copy()
    order = np.argsort(lv, kind="stable")
    srt = lv[order]
    if k is not None:
        srt[srt < srt[srt.size - k]] = -np.inf
    if p is not None:
        e = np.exp(srt - np.max(srt))
        cum = np.cumsum(e / e.sum())
        drop = cum <= 1 - p
        drop[-1] = False
        srt[drop] = -np.inf
    return {int(order[i]) for i in range(srt.size) if np.isfinite(srt[i])}


def top_k_sets(logits: np.ndarray, k: int) -> list[set[int]]:
    """Every correct top-k set, by brute force: k tokens of largest total logit."""
    best = max(sum(logits[list(c)]) for c in itertools.combinations(range(logits.size), k))
    return [
        set(c)
        for c in itertools.combinations(range(logits.size), k)
        if math.isclose(sum(logits[list(c)]), best, rel_tol=0, abs_tol=1e-9)
    ]


small = st.lists(st.integers(-3, 3), min_size=2, max_size=7).map(np.array)


@settings(max_examples=300, deadline=None)
@given(small, st.data())
def test_top_k_count_is_exactly_set_membership(logits: np.ndarray, data: st.DataObject) -> None:
    k = data.draw(st.integers(1, logits.size))
    t = data.draw(st.integers(0, logits.size - 1))
    expected = any(t in s for s in top_k_sets(logits.astype(float), k))
    assert check_member(logits, t, top_k=k).ok == expected


@settings(max_examples=300, deadline=None)
@given(
    st.lists(st.floats(-6, 6, allow_nan=False), min_size=2, max_size=40).map(np.array),
    st.data(),
)
def test_every_token_vllm_keeps_is_accepted(logits: np.ndarray, data: st.DataObject) -> None:
    k = data.draw(st.one_of(st.none(), st.integers(1, logits.size)))
    p = data.draw(st.one_of(st.none(), st.floats(0.05, 0.99)))
    for t in vllm_kept(logits, k, p):
        assert check_member(logits, t, top_k=k, top_p=p).ok  # no false alarms


def test_rejects_tokens_outside_top_k_or_the_nucleus() -> None:
    logits = np.array([5.0, 4.0, 3.0, 0.0, -1.0])
    assert check_member(logits, 2, top_k=3).ok and not check_member(logits, 3, top_k=3).ok
    # softmax mass above token 2 is ~0.91 > p = 0.5: far outside the nucleus
    v = check_member(logits, 2, top_p=0.5)
    assert not v.ok and v.mass_above > 0.9
    assert check_member(logits, 0, greedy=True).ok and not check_member(logits, 1, greedy=True).ok
    assert not check_member(np.array([1.0, math.nan]), 0).ok


def test_the_error_bound_is_tiny_at_llm_vocabularies() -> None:
    assert member_eps(151_936) < 1e-10 and member_eps(2) < 1e-14


def test_the_batched_check_matches_the_reference() -> None:
    torch = pytest.importorskip("torch")
    from pck.member import check_members

    rng = np.random.default_rng(0)
    logits = rng.standard_normal((64, 500)) * 3
    logits[:, :5] = 7.0  # ties at the top
    tokens = rng.integers(0, 500, 64)
    tokens[:32] = np.argsort(-logits[:32], axis=1)[np.arange(32), rng.integers(0, 20, 32)]
    top_k = rng.choice([1, 5, 20, 500], 64)
    top_p = rng.choice([0.3, 0.9, 1.0], 64)
    greedy = rng.random(64) < 0.1
    out = check_members(
        torch.tensor(logits, dtype=torch.float32),
        torch.tensor(tokens),
        torch.tensor(top_k),
        torch.tensor(top_p),
        torch.tensor(greedy),
    )
    f32 = logits.astype(np.float32)
    for i in range(64):
        k = None if top_k[i] == 500 else int(top_k[i])
        p = None if top_p[i] == 1.0 else float(top_p[i])
        ref = check_member(f32[i], int(tokens[i]), k, p, greedy=bool(greedy[i]))
        assert bool(out["ok"][i]) == ref.ok, i
        assert int(out["above"][i]) == ref.above
        if ref.ok and p is not None and not greedy[i]:
            assert abs(float(out["mass_above"][i]) - ref.mass_above) < 1e-12
