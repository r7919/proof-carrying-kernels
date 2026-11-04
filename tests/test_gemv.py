from fractions import Fraction
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pck.fp import ceil_div, gamma, tree_height
from pck.gemv import abs_row_sums, emulate_gemv, emulate_gemv_coalesced, reference_gemv
from tests.strategies import Problem, gemv_problems


@st.composite
def random_tree_sums(draw: st.DrawFn) -> tuple[list[float], float, int]:
    """Sum random terms with a random binary tree in FP32; return terms, result, height."""
    terms = draw(
        st.lists(
            st.floats(-1e6, 1e6, width=32, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=40,
        )
    )
    nodes: list[tuple[np.float32, int]] = [(np.float32(t), 0) for t in terms]
    while len(nodes) > 1:
        i = draw(st.integers(0, len(nodes) - 2))
        (a, ha), (b, hb) = nodes[i], nodes[i + 1]
        nodes[i : i + 2] = [(np.float32(a + b), max(ha, hb) + 1)]
    total, height = nodes[0]
    return terms, float(total), height


@given(random_tree_sums())
def test_any_summation_tree_obeys_height_bound(case: tuple[list[float], float, int]) -> None:
    """The core lemma behind T1: |fl(sum) - sum| <= gamma_h * sum |terms|."""
    terms, computed, height = case
    exact = sum(map(Fraction, terms))
    bound = Fraction(gamma(height)) * sum(abs(Fraction(t)) for t in terms)
    assert abs(Fraction(computed) - exact) <= bound


@given(gemv_problems())
def test_legit_kernel_within_spec(p: Problem) -> None:
    y = emulate_gemv(p.W, p.x, p.threads)
    h = tree_height(p.x.size, p.threads)
    err = np.abs(y.astype(np.float64) - reference_gemv(p.W, p.x))
    assert y.dtype == np.float32
    assert np.all(err <= gamma(h) * abs_row_sums(p.W, p.x))


def test_exact_on_small_integers() -> None:
    W = np.arange(12, dtype=np.float32).reshape(3, 4)
    x = np.array([1, -2, 3, -4], dtype=np.float32)
    for threads in (1, 2, 4, 8):
        np.testing.assert_array_equal(emulate_gemv(W, x, threads), W @ x)


def test_drop_tail_zeroes_last_columns() -> None:
    W = np.ones((2, 5), dtype=np.float32)
    x = np.ones(5, dtype=np.float32)
    np.testing.assert_array_equal(emulate_gemv(W, x, 2, drop_tail=2), [3.0, 3.0])


@pytest.mark.parametrize("exploit", [{"acc_bf16": True}, {"partial_bf16": True}, {"drop_tail": 4}])
def test_exploits_violate_spec(exploit: dict[str, Any]) -> None:
    rng = np.random.default_rng(0)
    m, n, threads = 64, 1000, 32
    W = rng.standard_normal((m, n)).astype(np.float32) * 0.02
    x = rng.standard_normal(n).astype(np.float32)
    y = emulate_gemv(W, x, threads, **exploit)
    err = np.abs(y.astype(np.float64) - reference_gemv(W, x))
    assert np.any(err > gamma(tree_height(n, threads)) * abs_row_sums(W, x))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"threads": 3}, "power of two"),
        ({"threads": 4, "drop_tail": 9}, "drop_tail"),
    ],
)
def test_rejects_bad_arguments(kwargs: dict[str, Any], match: str) -> None:
    W = np.ones((2, 8), dtype=np.float32)
    with pytest.raises(ValueError, match=match):
        emulate_gemv(W, np.ones(8, dtype=np.float32), **kwargs)


def test_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        emulate_gemv(np.ones((2, 8), np.float32), np.ones(7, np.float32), 4)


def loop_reference_coalesced(W: np.ndarray, x: np.ndarray, threads: int, vec: int) -> np.ndarray:
    """Thread-by-thread scalar model of the coalesced kernel, independent of reshapes."""
    m, n = W.shape
    per_thread = ceil_div(ceil_div(n, vec), threads)
    y = np.empty(m, dtype=np.float32)
    for i in range(m):
        acc = []
        for t in range(threads):
            terms = [
                np.float32(W[i, j] * x[j]) if j < n else np.float32(0)
                for v in range(per_thread)
                for j in range((v * threads + t) * vec, (v * threads + t + 1) * vec)
            ]
            s = terms[0]
            for term in terms[1:]:
                s = np.float32(s + term)
            acc.append(s)
        # Halving within each warp, then across warps.
        warps = []
        for w in range(threads // 32):
            lanes = acc[32 * w : 32 * (w + 1)]
            while len(lanes) > 1:
                h = len(lanes) // 2
                lanes = [np.float32(lanes[k] + lanes[k + h]) for k in range(h)]
            warps.append(lanes[0])
        while len(warps) > 1:
            h = len(warps) // 2
            warps = [np.float32(warps[k] + warps[k + h]) for k in range(h)]
        y[i] = warps[0]
    return y


@given(
    gemv_problems(max_rows=3, max_cols=150), st.sampled_from([32, 64]), st.sampled_from([1, 4, 8])
)
def test_coalesced_matches_loop_reference(p: Problem, threads: int, vec: int) -> None:
    np.testing.assert_array_equal(
        emulate_gemv_coalesced(p.W, p.x, threads, vec).view(np.uint32),
        loop_reference_coalesced(p.W, p.x, threads, vec).view(np.uint32),
    )


@given(gemv_problems(), st.sampled_from([32, 64, 128, 256]), st.sampled_from([1, 2, 4, 8]))
def test_coalesced_within_spec(p: Problem, threads: int, vec: int) -> None:
    y = emulate_gemv_coalesced(p.W, p.x, threads, vec)
    h = tree_height(p.x.size, threads, vec)
    err = np.abs(y.astype(np.float64) - reference_gemv(p.W, p.x))
    assert y.dtype == np.float32
    assert np.all(err <= gamma(h) * abs_row_sums(p.W, p.x))


def test_coalesced_exact_on_small_integers() -> None:
    W = np.arange(3 * 300, dtype=np.float32).reshape(3, 300) % 7
    x = (np.arange(300, dtype=np.float32) % 5) - 2
    for threads in (32, 64, 256):
        np.testing.assert_array_equal(emulate_gemv_coalesced(W, x, threads), W @ x)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"threads": 16}, ">= 32"),
        ({"threads": 48}, "power of two"),
        ({"threads": 32, "vec": 0}, "vec"),
    ],
)
def test_coalesced_rejects_bad_arguments(kwargs: dict[str, Any], match: str) -> None:
    W = np.ones((2, 8), dtype=np.float32)
    with pytest.raises(ValueError, match=match):
        emulate_gemv_coalesced(W, np.ones(8, dtype=np.float32), **kwargs)


def test_coalesced_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        emulate_gemv_coalesced(np.ones((2, 8), np.float32), np.ones(7, np.float32), 32)
