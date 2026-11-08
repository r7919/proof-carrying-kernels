import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pck.checker import BlockedChecker
from pck.faults import flip_bit
from pck.fp import bf16, tree_height
from pck.gemv import emulate_gemv, emulate_gemv_coalesced
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
    st.sampled_from([32, 128, 256]),
    st.integers(0, 2**32 - 1),
)
def test_no_false_alarms_on_coalesced_kernel(p: Problem, threads: int, seed: int) -> None:
    y = emulate_gemv_coalesced(p.W, p.x, threads)
    h = tree_height(p.x.size, threads, vec=8)
    checker = BlockedChecker(p.W, p.x, BLOCK, h)
    assert checker.verdict(y, np.random.default_rng(seed), checks=4).all()


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


def test_crafted_output_passes_at_most_half_of_sign_draws() -> None:
    """Empirical mirror of T4: an adversary that sees one sign vector cannot fool fresh ones.

    One output is off by more than tau; the other outputs are shifted to cancel it exactly
    under a guessed sign vector. With signs drawn after y is fixed, at most half the checks
    pass (T4), so with K = 1 the pass rate must stay near or below 1/2.
    """
    rng = np.random.default_rng(5)
    b, n = 8, 256
    W = bf16(rng.standard_normal((b, n)) * 0.02)
    x = bf16(rng.standard_normal(n))
    checker = BlockedChecker(W, x, b, tree_height(n, 32))
    y = emulate_gemv(W, x, 32).astype(np.float64)
    big = 1.5 * checker.tau_base[0]
    guess = rng.choice([-1.0, 1.0], b)
    y[0] += big
    y[1:] -= guess[0] * big * guess[1:] / (b - 1)  # cancels under the guessed signs
    trials = 4000
    passes = sum(bool(checker.verdict(y.astype(np.float32), rng)[0]) for _ in range(trials))
    assert passes / trials <= 0.5 + 0.03


def test_spread_error_passes_at_most_13_16_of_sign_draws() -> None:
    """Empirical mirror of T4's Paley-Zygmund part: every output is off by less than tau,
    but ||D||_2 >= 2 tau, so a single check passes for at most 13/16 of sign vectors."""
    rng = np.random.default_rng(6)
    b, n = 8, 256
    W = bf16(rng.standard_normal((b, n)) * 0.02)
    x = bf16(rng.standard_normal(n))
    checker = BlockedChecker(W, x, b, tree_height(n, 32))
    y = emulate_gemv(W, x, 32).astype(np.float64)
    tau = checker.tau_base[0]
    D = rng.choice([-1.0, 1.0], b) * (2.05 * tau / np.sqrt(b))  # each |D_i| < tau
    assert np.all(np.abs(D) < tau) and np.linalg.norm(D) >= 2 * tau
    trials = 4000
    passes = sum(bool(checker.verdict((y + D).astype(np.float32), rng)[0]) for _ in range(trials))
    assert passes / trials <= 13 / 16 + 0.03


def test_an_overflowing_threshold_fails_closed() -> None:
    """Weights of +-2^125 make a_beta (a sum over the block of 8 rows) overflow FP32, so tau
    is inf; that must reject, not accept everything (the honest output is exactly 0)."""
    m, n = 8, 64
    signs = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    W = np.tile(signs * 2.0**125, (m, 1)).astype(np.float32)
    x = np.ones(n, dtype=np.float32)
    checker = BlockedChecker(W, x, block=m, height=tree_height(n, 32))
    rng = np.random.default_rng(0)
    assert not checker.verdict(np.full(m, 1e30), rng).any()
    assert not checker.verdict(np.zeros(m), rng).any()  # fails closed, even when honest


def test_a_shared_weight_bank_gives_the_same_verdicts() -> None:
    from pck.checker import WeightBank

    rng = np.random.default_rng(1)
    W = bf16(rng.standard_normal((16, 40)))
    bank = WeightBank(W, 8)
    for seed in range(3):
        x = bf16(np.random.default_rng(seed).standard_normal(40))
        y = emulate_gemv(W, x, 4)
        own, shared = BlockedChecker(W, x, 8, 10), BlockedChecker(W, x, 8, 10, bank)
        np.testing.assert_array_equal(own.abs_bound, shared.abs_bound)
        assert (
            own.verdict(y, np.random.default_rng(0)) == shared.verdict(y, np.random.default_rng(0))
        ).all()
    with pytest.raises(ValueError, match="another W"):
        BlockedChecker(W, x, 16, 10, bank)
