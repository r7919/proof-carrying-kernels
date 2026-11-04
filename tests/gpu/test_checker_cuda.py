import math
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from pck.cuda import GpuBlockedChecker, gemv_coalesced, gemv_naive
from pck.fp import U32, U64, bf16, gamma, tree_height
from pck.gemv import emulate_gemv

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
]

M, N, THREADS = 512, 4100, 256
HEIGHT = tree_height(N, THREADS, vec=8)


def problem(
    m: int = M, n: int = N, seed: int = 0
) -> tuple[NDArray[np.float32], NDArray[np.float32], torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    W, x = bf16(rng.standard_normal((m, n)) * 0.02), bf16(rng.standard_normal(n))
    return (
        W,
        x,
        torch.from_numpy(W).to(torch.bfloat16).cuda(),
        torch.from_numpy(x).cuda().to(torch.bfloat16),
    )


def make(Wt: torch.Tensor, block: int = 64, checks: int = 8, seed: int = 0) -> GpuBlockedChecker:
    gen = torch.Generator().manual_seed(seed)
    return GpuBlockedChecker(Wt, block, HEIGHT, checks=checks, generator=gen)


def as_cuda(y: NDArray[np.float32]) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)).cuda()


@pytest.mark.parametrize("block", [1, 8, 64, 128, 512])
def test_legit_gpu_output_passes(block: int) -> None:
    _, _, Wt, xt = problem()
    checker = make(Wt, block)
    for y in (gemv_coalesced(Wt, xt, THREADS), gemv_naive(Wt, xt, THREADS)):
        result = checker.check(xt, y)
        assert result.passed.all()
        assert (result.rho.abs() <= result.tau).all()


def test_matches_fp64_reference() -> None:
    """rho matches an FP64 reference; tau is never below it (T3) and at most 0.1% above.

    The kernel bounds a . |x| with round-up FP32 arithmetic, which is looser than exact by
    at most about n * 2^-24 relative, and bounds |c| . |x| by (1 + u32) * a . |x|.
    """
    _, x, Wt, xt = problem(m=256, n=1000, seed=1)
    checker = GpuBlockedChecker(
        Wt, 32, HEIGHT, checks=4, generator=torch.Generator().manual_seed(1)
    )
    y = gemv_coalesced(Wt, xt, 128)
    res = checker.check(xt, y)

    c = checker.c.cpu().numpy().astype(np.float64)
    a = checker.a.cpu().numpy().astype(np.float64)
    r = checker.r.cpu().numpy().astype(np.float64)
    x64, yb = x.astype(np.float64), y.cpu().numpy().astype(np.float64).reshape(-1, 32)
    rho = c @ x64 - np.einsum("kbi,bi->kb", r, yb)
    s = a @ np.abs(x64)
    eps = gamma(1000 + 32 + 1, U64) * ((1 + U32) * s + np.abs(yb).sum(1))
    tau = (gamma(HEIGHT) + U32) * s + eps
    rho_tol = gamma(1000 + 32 + 1, U64) * (np.abs(c) @ np.abs(x64) + np.abs(yb).sum(1))

    np.testing.assert_allclose(res.rho.cpu().numpy(), rho, rtol=0, atol=2 * float(rho_tol.max()))
    tau_gpu = res.tau.cpu().numpy()
    assert np.all(tau_gpu >= tau * (1 - 1e-12))  # reference sums are within ~n * 2^-53
    assert np.all(tau_gpu <= tau * (1 + 1e-3))


@pytest.mark.parametrize("exploit", ["skip_tail", "acc_bf16", "partial_bf16", "bf16_output"])
def test_planted_exploits_caught(exploit: str) -> None:
    W, x, Wt, xt = problem(m=512, n=2048, seed=2)
    height = tree_height(2048, 128)
    checker = GpuBlockedChecker(
        Wt, 64, height, checks=8, generator=torch.Generator().manual_seed(2)
    )
    legit = emulate_gemv(W, x, 128)
    y = {
        "skip_tail": lambda: emulate_gemv(W, x, 128, drop_tail=4),
        "acc_bf16": lambda: emulate_gemv(W, x, 128, acc_bf16=True),
        "partial_bf16": lambda: emulate_gemv(W, x, 128, partial_bf16=True),
        "bf16_output": lambda: bf16(legit),
    }[exploit]()
    assert checker.verdict(xt, as_cuda(legit)).all()
    assert not checker.verdict(xt, as_cuda(y)).all()


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_output_fails_its_block_only(bad: float) -> None:
    _, _, Wt, xt = problem()
    checker = make(Wt, 64, checks=2)
    y = gemv_coalesced(Wt, xt, THREADS)
    y[130] = bad
    verdict = checker.verdict(xt, y).cpu().numpy()
    expected = np.ones(M // 64, dtype=bool)
    expected[130 // 64] = False
    np.testing.assert_array_equal(verdict, expected)


def test_large_single_error_always_caught() -> None:
    """GPU mirror of T2: a deviation of 2.5 * tau in one output fails its block for any signs."""
    _, _, Wt, xt = problem(seed=4)
    checker = make(Wt, 64, checks=8, seed=4)
    y = gemv_coalesced(Wt, xt, THREADS)
    tau = checker.check(xt, y).tau.min(dim=0).values
    rows = torch.arange(0, M, 64 + 3) % M
    for i in rows.tolist():
        bad = y.clone()
        bad[i] += 2.5 * float(tau[i // 64])
        verdict = checker.verdict(xt, bad)
        assert not verdict[i // 64]
        assert verdict.sum() == M // 64 - 1


def test_no_false_alarms_over_many_draws() -> None:
    _, _, Wt, xt = problem(seed=5)
    y = gemv_coalesced(Wt, xt, THREADS)
    for seed in range(50):
        assert make(Wt, 64, checks=8, seed=seed).verdict(xt, y).all()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"block": 7}, "divide"), ({"block": 0}, "positive"), ({"block": 64, "checks": 0}, "checks")],
)
def test_rejects_bad_configuration(kwargs: dict[str, Any], match: str) -> None:
    _, _, Wt, _ = problem(m=64, n=16)
    with pytest.raises(ValueError, match=match):
        GpuBlockedChecker(Wt, height=4, **kwargs)


def test_rejects_non_bf16_weights() -> None:
    with pytest.raises(ValueError, match="bfloat16"):
        GpuBlockedChecker(torch.ones((8, 8), device="cuda"), 4, 4)


@pytest.mark.parametrize(
    ("which", "match"),
    [("y_dtype", "y must be"), ("y_shape", "shape"), ("x_dtype", "x must be"), ("y_cpu", "CUDA")],
)
def test_rejects_bad_check_inputs(which: str, match: str) -> None:
    _, _, Wt, xt = problem(m=64, n=16)
    checker = GpuBlockedChecker(Wt, 16, 4)
    y = torch.zeros(64, device="cuda")
    args = {
        "y_dtype": (xt, y.double()),
        "y_shape": (xt, y[:32]),
        "x_dtype": (xt.float(), y),
        "y_cpu": (xt, y.cpu()),
    }[which]
    with pytest.raises(RuntimeError, match=match):
        checker.check(*args)


@pytest.mark.parametrize("segments", [1, 2, 3, 7, 64, 1000])
def test_segment_count_does_not_change_result(segments: int) -> None:
    """Splitting columns changes only summation order: verdicts and rho agree, and tau
    differs only by the FP32 round-up slack in a . |x| (at most ~(n + 1) * 2^-24 relative),
    never dropping below its exact value."""
    _, x, Wt, xt = problem(m=256, n=1000, seed=6)
    y = gemv_coalesced(Wt, xt, 128)

    def checker(segs: int) -> GpuBlockedChecker:
        gen = torch.Generator().manual_seed(6)
        return GpuBlockedChecker(Wt, 32, HEIGHT, checks=4, generator=gen, segments=segs)

    ref = checker(1).check(xt, y)
    split_checker = checker(segments)
    split = split_checker.check(xt, y)
    assert torch.equal(split.passed, ref.passed)
    torch.testing.assert_close(split.rho, ref.rho, rtol=0, atol=1e-12)
    torch.testing.assert_close(split.tau, ref.tau, rtol=(1000 + 1) * 2.0**-24, atol=0)

    a = split_checker.a.cpu().numpy().astype(np.float64)
    s_exact = a @ np.abs(x.astype(np.float64))
    assert np.all(split.tau.cpu().numpy() >= (gamma(HEIGHT) + U32) * s_exact * (1 - 1e-12))


def test_repeated_calls_reuse_scratch_correctly() -> None:
    _, _, Wt, xt = problem(seed=7)
    checker = make(Wt, 64, checks=2)
    assert checker.segments >= 1
    y = gemv_coalesced(Wt, xt, THREADS)
    bad = y.clone()
    bad[5] = math.nan
    first = checker.check(xt, y)
    for _ in range(20):
        assert torch.equal(checker.verdict(xt, bad)[1:], first.verdict[1:])
        assert not checker.verdict(xt, bad)[0]
        again = checker.check(xt, y)
        assert torch.equal(again.passed, first.passed)
        assert torch.equal(again.rho, first.rho)


@pytest.mark.parametrize("segments", [0, 1001])
def test_rejects_bad_segment_count(segments: int) -> None:
    _, _, Wt, _ = problem(m=64, n=1000)
    with pytest.raises(ValueError, match="segments"):
        GpuBlockedChecker(Wt, 16, 4, segments=segments)
