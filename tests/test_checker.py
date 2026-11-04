import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pck.checker import BlockedChecker
from pck.faults import flip_bit
from pck.fp import bf16, tree_height
from pck.gemv import emulate_gemv
from tests.strategies import Problem, gemv_problems

BLOCK = 8


def make_checker(p: Problem, block: int = BLOCK) -> BlockedChecker:
    return BlockedChecker(p.W, p.x, block, tree_height(p.x.size, p.threads))


@given(gemv_problems(row_multiple=BLOCK), st.integers(0, 2**32 - 1))
def test_no_false_alarms_on_legit_kernel(p: Problem, seed: int) -> None:
    """Empirical mirror of T1: a kernel inside the spec never trips the checker."""
    y = emulate_gemv(p.W, p.x, p.threads)
    rng = np.random.default_rng(seed)
    assert make_checker(p).verdict(y, rng, checks=4).all()


@given(
    gemv_problems(row_multiple=BLOCK),
    st.integers(0, 2**32 - 1),
    st.data(),
)
def test_large_single_error_always_caught(p: Problem, seed: int, data: st.DataObject) -> None:
    """Empirical mirror of T2: a deviation above 2*tau in one output is caught for any signs."""
    checker = make_checker(p)
    y = emulate_gemv(p.W, p.x, p.threads).astype(np.float64)
    i = data.draw(st.integers(0, y.size - 1))
    blk = i // BLOCK
    sign = data.draw(st.sampled_from([-1.0, 1.0]))
    y[i] += sign * 2.5 * checker.tau_base[blk]
    y32 = y.astype(np.float32)
    if not np.isfinite(y32).all():
        return
    verdict = checker.verdict(y32, np.random.default_rng(seed), checks=1)
    assert not verdict[blk]
    others = np.delete(verdict, blk)
    assert others.all()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_output_fails_its_block_only(bad: float) -> None:
    rng = np.random.default_rng(0)
    W = bf16(rng.standard_normal((32, 64)) * 0.02)
    x = bf16(rng.standard_normal(64))
    y = emulate_gemv(W, x, 8)
    y[13] = bad
    verdict = BlockedChecker(W, x, BLOCK, tree_height(64, 8)).verdict(y, rng, checks=2)
    np.testing.assert_array_equal(verdict, [True, False, True, True])


@pytest.mark.parametrize(
    "exploit",
    ["skip_tail", "acc_bf16", "partial_bf16", "bf16_output"],
)
def test_planted_exploits_caught_at_b64(exploit: str) -> None:
    rng = np.random.default_rng(1)
    m, n, threads = 512, 2048, 128
    W = bf16(rng.standard_normal((m, n)) * 0.02)
    x = bf16(rng.standard_normal(n))
    legit = emulate_gemv(W, x, threads)
    y = {
        "skip_tail": lambda: emulate_gemv(W, x, threads, drop_tail=4),
        "acc_bf16": lambda: emulate_gemv(W, x, threads, acc_bf16=True),
        "partial_bf16": lambda: emulate_gemv(W, x, threads, partial_bf16=True),
        "bf16_output": lambda: bf16(legit),
    }[exploit]()
    checker = BlockedChecker(W, x, 64, tree_height(n, threads))
    assert checker.verdict(legit, rng, checks=8).all()
    assert not checker.verdict(y, rng, checks=8).all()


def test_exponent_flip_to_inf_is_caught() -> None:
    rng = np.random.default_rng(2)
    W = bf16(rng.standard_normal((64, 128)) * 0.02)
    x = bf16(rng.standard_normal(128))
    y = emulate_gemv(W, x, 16)
    y[5] = 1.5  # top exponent bit set gives Inf/NaN
    flipped = flip_bit(y, 5, 30)
    assert not np.isfinite(flipped[5])
    checker = BlockedChecker(W, x, 64, tree_height(128, 16))
    assert not checker.verdict(flipped, rng)[0]


@pytest.mark.parametrize(
    ("block", "x_len", "match"),
    [(5, 4, "divide"), (0, 4, "positive"), (4, 3, "shape")],
)
def test_rejects_bad_configuration(block: int, x_len: int, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        BlockedChecker(np.ones((8, 4)), np.ones(x_len), block, 2)


def test_rejects_wrong_output_shape() -> None:
    checker = BlockedChecker(np.ones((8, 4)), np.ones(4), 4, 2)
    with pytest.raises(ValueError, match="shape"):
        checker.verdict(np.ones(7), np.random.default_rng(0))
