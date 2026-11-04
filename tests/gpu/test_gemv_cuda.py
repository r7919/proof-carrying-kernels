import numpy as np
import pytest
import torch
from hypothesis import given, settings
from numpy.typing import NDArray

from pck.checker import BlockedChecker
from pck.cuda import gemv_naive
from pck.fp import bf16, tree_height
from pck.gemv import emulate_gemv
from tests.strategies import Problem, gemv_problems

pytestmark = pytest.mark.gpu
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def to_cuda(a: NDArray[np.float32]) -> torch.Tensor:
    """BF16-valued FP32 array -> BF16 CUDA tensor (exact, since values are already BF16)."""
    return torch.from_numpy(a).to(torch.bfloat16).cuda()


def run(W: NDArray[np.float32], x: NDArray[np.float32], threads: int) -> NDArray[np.float32]:
    y = gemv_naive(to_cuda(W), to_cuda(x), threads)
    assert y.dtype == torch.float32
    out: NDArray[np.float32] = y.cpu().numpy()
    return out


def random_problem(m: int, n: int, seed: int) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    rng = np.random.default_rng(seed)
    return bf16(rng.standard_normal((m, n)) * 0.02), bf16(rng.standard_normal(n))


@needs_cuda
@pytest.mark.parametrize(
    ("m", "n", "threads"),
    [
        (1, 1, 1),
        (3, 5, 1),
        (7, 100, 4),
        (64, 1000, 32),
        (5, 17, 1024),
        (33, 4096, 128),
        (256, 4100, 256),
    ],
)
def test_bit_exact_with_emulator(m: int, n: int, threads: int) -> None:
    W, x = random_problem(m, n, seed=m * n + threads)
    np.testing.assert_array_equal(
        run(W, x, threads).view(np.uint32), emulate_gemv(W, x, threads).view(np.uint32)
    )


@needs_cuda
@settings(max_examples=50)
@given(gemv_problems(max_rows=32, max_cols=600))
def test_bit_exact_with_emulator_random(p: Problem) -> None:
    np.testing.assert_array_equal(
        run(p.W, p.x, p.threads).view(np.uint32),
        emulate_gemv(p.W, p.x, p.threads).view(np.uint32),
    )


@needs_cuda
def test_signed_zero_matches_emulator() -> None:
    W = np.zeros((2, 8), dtype=np.float32)
    x = -np.ones(8, dtype=np.float32)
    for threads in (1, 2, 4, 16):
        np.testing.assert_array_equal(
            run(W, x, threads).view(np.uint32), emulate_gemv(W, x, threads).view(np.uint32)
        )


@needs_cuda
def test_checker_passes_gpu_output() -> None:
    m, n, threads = 512, 4100, 256
    W, x = random_problem(m, n, seed=7)
    checker = BlockedChecker(W, x, 64, tree_height(n, threads))
    assert checker.verdict(run(W, x, threads), np.random.default_rng(0), checks=8).all()


@needs_cuda
def test_empty_output() -> None:
    W = torch.zeros((0, 8), dtype=torch.bfloat16, device="cuda")
    x = torch.zeros(8, dtype=torch.bfloat16, device="cuda")
    assert gemv_naive(W, x).shape == (0,)


@needs_cuda
@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda W, x: (W.float(), x.float(), 4), "bfloat16"),
        (lambda W, x: (W.t(), x[:4], 4), "contiguous"),
        (lambda W, x: (W, x[:3], 4), "shape"),
        (lambda W, x: (W, x, 3), "power of two"),
        (lambda W, x: (W, x, 2048), "power of two"),
    ],
)
def test_rejects_bad_inputs(make: object, match: str) -> None:
    W = torch.ones((4, 8), dtype=torch.bfloat16, device="cuda")
    x = torch.ones(8, dtype=torch.bfloat16, device="cuda")
    args = make(W, x)  # type: ignore[operator]
    with pytest.raises(RuntimeError, match=match):
        gemv_naive(*args)


def test_rejects_cpu_tensors() -> None:
    W = torch.ones((4, 8), dtype=torch.bfloat16)
    x = torch.ones(8, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="CUDA tensors"):
        gemv_naive(W, x)
