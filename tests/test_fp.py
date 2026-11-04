import math
from fractions import Fraction

import ml_dtypes
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pck.fp import U32, bf16, ceil_div, gamma, round_up_f32, tree_height

f32 = st.floats(width=32, allow_nan=False)
F32_TINY = float(np.finfo(np.float32).tiny)
F32_MAX = float(np.finfo(np.float32).max)
bf16_values = (
    st.floats(width=32, allow_nan=False, allow_infinity=False)
    .map(lambda v: float(bf16(v)))
    .filter(math.isfinite)
)


class TestBf16:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (1.0, 1.0),
            (1 + 2**-8, 1.0),  # tie, rounds to even (down)
            (1 + 3 * 2**-8, 1 + 2**-6),  # tie, rounds to even (up)
            (1 + 2**-8 + 2**-20, 1 + 2**-7),  # just past the tie
            (-1.5, -1.5),
            (0.0, 0.0),
        ],
    )
    def test_known_values(self, value: float, expected: float) -> None:
        assert float(bf16(value)) == expected

    @given(f32)
    def test_matches_ml_dtypes(self, v: float) -> None:
        ours = bf16(v)
        ref = np.float32(np.float32(v).astype(ml_dtypes.bfloat16))
        assert ours.tobytes() == ref.tobytes()

    @given(f32)
    def test_idempotent_and_low_bits_clear(self, v: float) -> None:
        once = bf16(v)
        assert (once.view(np.uint32) & 0xFFFF) == 0
        assert bf16(once).tobytes() == once.tobytes()

    def test_nan_stays_nan(self) -> None:
        # 0x7F800001 is a NaN whose naive rounding would carry into Inf.
        nan = np.array([0x7F800001, 0xFFFFFFFF], dtype=np.uint32).view(np.float32)
        assert np.isnan(bf16(nan)).all()

    def test_overflow_rounds_to_inf(self) -> None:
        assert np.isposinf(bf16(np.finfo(np.float32).max))

    def test_does_not_modify_input(self) -> None:
        a = np.array([1 + 2**-8 + 2**-20], dtype=np.float32)
        before = a.copy()
        bf16(a)
        np.testing.assert_array_equal(a, before)

    @given(bf16_values, bf16_values)
    def test_product_of_two_bf16_is_exact_in_fp32(self, a: float, b: float) -> None:
        """The precision fact the GEMV bound relies on (barring over/underflow)."""
        exact = a * b  # FP64 holds the 16-bit-significand product exactly
        if not (exact == 0 or F32_TINY <= abs(exact) <= F32_MAX):
            return
        assert float(np.float32(a) * np.float32(b)) == exact


class TestGamma:
    def test_zero(self) -> None:
        assert gamma(0) == 0.0

    def test_known_value(self) -> None:
        assert gamma(24) == pytest.approx(24 * U32 / (1 - 24 * U32))

    @given(st.integers(0, 10**6))
    def test_increasing_and_above_ku(self, k: int) -> None:
        assert gamma(k + 1) > gamma(k) >= k * U32

    def test_rejects_ku_at_least_one(self) -> None:
        with pytest.raises(ValueError, match="k\\*u < 1"):
            gamma(2**24)

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            gamma(-1)


class TestRoundUpF32:
    @given(st.floats(0, F32_MAX, allow_nan=False))
    def test_never_rounds_down_and_is_tight(self, v: float) -> None:
        up = float(round_up_f32(np.array([v]))[0])
        assert up >= v
        assert float(np.float32(up)) == up
        below = float(np.nextafter(np.float32(up), np.float32(0)))
        assert below < v or up == 0

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            round_up_f32(np.array([-1.0]))


class TestTreeHeight:
    @pytest.mark.parametrize(
        ("n", "threads", "expected"),
        [(4100, 256, 24), (4096, 256, 23), (1, 1, 0), (5, 1, 4), (3, 8, 3)],
    )
    def test_known_values(self, n: int, threads: int, expected: int) -> None:
        assert tree_height(n, threads) == expected

    @given(st.integers(1, 10**5), st.integers(0, 10))
    def test_formula(self, n: int, log_t: int) -> None:
        t = 2**log_t
        assert tree_height(n, t) == math.ceil(n / t) - 1 + log_t

    @pytest.mark.parametrize("threads", [0, 3, 6, -2])
    def test_rejects_non_power_of_two(self, threads: int) -> None:
        with pytest.raises(ValueError, match="power of two"):
            tree_height(10, threads)

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            tree_height(0, 4)

    @pytest.mark.parametrize(
        ("n", "threads", "vec", "expected"),
        [(4096, 256, 8, 23), (4100, 256, 8, 31), (4096, 128, 8, 38), (1, 32, 8, 12)],
    )
    def test_vectorized_known_values(self, n: int, threads: int, vec: int, expected: int) -> None:
        assert tree_height(n, threads, vec) == expected

    @given(st.integers(1, 10**5), st.integers(0, 10), st.integers(1, 16))
    def test_vectorized_formula(self, n: int, log_t: int, vec: int) -> None:
        t = 2**log_t
        chunk = vec * math.ceil(math.ceil(n / vec) / t)
        assert tree_height(n, t, vec) == chunk - 1 + log_t

    def test_vec_one_is_contiguous_layout(self) -> None:
        assert all(tree_height(n, 64, 1) == tree_height(n, 64) for n in range(1, 500))

    def test_rejects_bad_vec(self) -> None:
        with pytest.raises(ValueError, match="vec"):
            tree_height(10, 4, 0)


@given(st.integers(0, 10**9), st.integers(1, 10**6))
def test_ceil_div(a: int, b: int) -> None:
    assert ceil_div(a, b) == math.ceil(Fraction(a, b))
