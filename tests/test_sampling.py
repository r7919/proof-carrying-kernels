import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from numpy.typing import NDArray

from pck.faults import sharded_topk
from pck.fp import bf16
from pck.sampling import (
    check_sampled,
    check_topk,
    check_topp,
    softmax64,
    topk_reference,
    topp_reference,
)


@st.composite
def logit_vectors(draw: st.DrawFn, max_size: int = 300) -> NDArray[np.float32]:
    """Random logits; BF16 rounding of some draws creates ties."""
    size = draw(st.integers(1, max_size))
    seed = draw(st.integers(0, 2**32 - 1))
    scale = draw(st.sampled_from([0.1, 1.0, 5.0, 30.0]))
    rng = np.random.default_rng(seed)
    a = (rng.standard_normal(size) * scale).astype(np.float32)
    if draw(st.booleans()):
        a = bf16(a)
    return a


class TestTopk:
    @given(logit_vectors(), st.data())
    def test_reference_passes(self, a: NDArray[np.float32], data: st.DataObject) -> None:
        k = data.draw(st.integers(1, a.size))
        assert check_topk(a, topk_reference(a, k), k)

    @given(logit_vectors(), st.data())
    def test_swapping_in_a_larger_rejected_logit_fails(
        self, a: NDArray[np.float32], data: st.DataObject
    ) -> None:
        assume(a.size >= 2)
        k = data.draw(st.integers(1, a.size - 1))
        sel = topk_reference(a, k)
        rejected = np.setdiff1d(np.arange(a.size), sel)
        # Replace the largest selected logit by the smallest rejected one.
        worst = rejected[np.argmin(a[rejected])]
        best = sel[np.argmax(a[sel])]
        swapped = np.where(sel == best, worst, sel)
        assert check_topk(a, swapped, k) == (a[worst] == a[best])

    def test_ties_may_go_either_way(self) -> None:
        a = np.array([3.0, 1.0, 2.0, 2.0], dtype=np.float32)
        assert check_topk(a, [0, 2], 2) and check_topk(a, [0, 3], 2)
        assert not check_topk(a, [0, 1], 2)

    @pytest.mark.parametrize(
        "selected",
        [[0, 0], [0, 4], [-1, 0], [0], [0, 1, 2], [], [0.0, 1.0], [[0, 1]]],
    )
    def test_malformed_selection_fails(self, selected: list[object]) -> None:
        a = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32)
        assert not check_topk(a, np.asarray(selected), 2)

    def test_nan_logit_fails(self) -> None:
        a = np.array([4.0, np.nan, 2.0], dtype=np.float32)
        assert not check_topk(a, [0, 2], 2)

    def test_select_all(self) -> None:
        a = np.array([1.0, 2.0], dtype=np.float32)
        assert check_topk(a, [1, 0], 2)

    def test_reference_rejects_bad_k(self) -> None:
        with pytest.raises(ValueError, match="k must"):
            topk_reference([1.0, 2.0], 3)
        with pytest.raises(ValueError, match="non-empty"):
            topk_reference([], 1)


class TestTopp:
    @given(logit_vectors(), st.floats(0.01, 1.0), st.sampled_from([0.5, 1.0, 2.0]))
    def test_reference_passes(self, a: NDArray[np.float32], p: float, temp: float) -> None:
        assert check_topp(a, topp_reference(a, p, temp), p, temp, tol=0.0)

    @given(logit_vectors(), st.floats(0.05, 0.95), st.sampled_from([0.7, 1.0]))
    def test_fp32_softmax_sampler_passes_with_its_tolerance(
        self, a: NDArray[np.float32], p: float, temp: float
    ) -> None:
        """A sampler that rounds its softmax to FP32 may pick a different boundary token."""
        z = a / np.float32(temp)
        e = np.exp(z - np.max(z))
        probs32 = (e / e.sum()).astype(np.float32)
        order = np.lexsort((np.arange(a.size), -a.astype(np.float64)))
        mass = np.cumsum(probs32[order], dtype=np.float32)
        count = min(int(np.searchsorted(mass, np.float32(p))) + 1, a.size)
        assert check_topp(a, order[:count], p, temp, tol=1e-5)

    @given(logit_vectors(), st.floats(0.05, 0.9))
    def test_one_token_too_many_or_too_few_fails(self, a: NDArray[np.float32], p: float) -> None:
        sel = topp_reference(a, p)
        probs = softmax64(a)
        order = np.lexsort((np.arange(a.size), -a.astype(np.float64)))
        slack = 1e-12  # well above the checker's own rounding slack
        if sel.size < a.size:
            extra = order[: sel.size + 1]
            # Valid only if the nucleus without its smallest token still falls short of p.
            if abs(probs[sel].sum() - p) > slack:
                assert check_topp(a, extra, p, tol=0.0) == (probs[sel].sum() < p)
        if sel.size > 1:
            fewer = order[: sel.size - 1]
            if abs(probs[fewer].sum() - p) > slack:
                assert check_topp(a, fewer, p, tol=0.0) == (probs[fewer].sum() >= p)

    def test_non_prefix_fails(self) -> None:
        a = np.array([5.0, 4.0, 0.0], dtype=np.float32)
        assert check_topp(a, [0, 1], 0.99)
        assert not check_topp(a, [0, 2], 0.99)

    def test_rejects_bad_arguments(self) -> None:
        a = np.array([1.0, 2.0], dtype=np.float32)
        with pytest.raises(ValueError, match="p must"):
            check_topp(a, [1], 0.0)
        with pytest.raises(ValueError, match="tol"):
            check_topp(a, [1], 0.5, tol=-1.0)
        with pytest.raises(ValueError, match="p must"):
            topp_reference(a, 1.5)
        with pytest.raises(ValueError, match="temperature"):
            softmax64(a, 0.0)

    def test_malformed_selection_fails(self) -> None:
        assert not check_topp(np.array([1.0, 2.0], dtype=np.float32), [1, 1], 0.5)


def test_sampled_token_membership() -> None:
    assert check_sampled(3, [1, 3, 5])
    assert not check_sampled(2, [1, 3, 5])


class TestShardedTopk:
    @given(logit_vectors(), st.data())
    def test_exact_with_enough_per_shard(self, a: NDArray[np.float32], data: st.DataObject) -> None:
        k = data.draw(st.integers(1, a.size))
        shards = data.draw(st.integers(1, 8))
        assert check_topk(a, sharded_topk(a, k, shards, per_shard=k), k)

    @given(logit_vectors(), st.data())
    def test_checker_catches_every_wrong_set(
        self, a: NDArray[np.float32], data: st.DataObject
    ) -> None:
        k = data.draw(st.integers(1, a.size))
        shards = data.draw(st.integers(1, 8))
        per_shard = data.draw(st.integers(1, k))
        got = sharded_topk(a, k, shards, per_shard)
        wrong = got.size != k or not np.array_equal(np.sort(a[got])[::-1], np.sort(a)[::-1][:k])
        assert check_topk(a, got, k) == (not wrong)

    def test_clustered_top_logits_are_missed(self) -> None:
        a = np.zeros(64, dtype=np.float32)
        a[:4] = [9.0, 8.0, 7.0, 6.0]  # all in the first of 8 shards
        got = sharded_topk(a, 4, shards=8, per_shard=2)
        assert not check_topk(a, got, 4)

    def test_rejects_bad_arguments(self) -> None:
        with pytest.raises(ValueError, match="per_shard"):
            sharded_topk(np.ones(4, np.float32), 2, 2, 0)
