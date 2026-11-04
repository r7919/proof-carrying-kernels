from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from pck.checker import BlockedChecker
from pck.cuda import gemv_coalesced, gemv_naive
from pck.fp import bf16, tree_height
from pck.gemv import emulate_gemv, emulate_gemv_coalesced
from tests.strategies import Problem, gemv_problems

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
]

Array = NDArray[np.float32]


@dataclass(frozen=True)
class Kernel:
    name: str
    run: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
    model: Callable[[Array, Array, int], Array]
    vec: int
    min_threads: int


KERNELS = [
    Kernel("naive", gemv_naive, emulate_gemv, vec=1, min_threads=1),
    Kernel("coalesced", gemv_coalesced, emulate_gemv_coalesced, vec=8, min_threads=32),
]
kernels = pytest.mark.parametrize("kernel", KERNELS, ids=lambda k: k.name)


def to_cuda(a: Array) -> torch.Tensor:
    """BF16-valued FP32 array -> BF16 CUDA tensor (exact, since values are already BF16)."""
    return torch.from_numpy(a).to(torch.bfloat16).cuda()


def run(kernel: Kernel, W: Array, x: Array, threads: int) -> Array:
    y = kernel.run(to_cuda(W), to_cuda(x), threads)
    assert y.dtype == torch.float32
    out: Array = y.cpu().numpy()
    return out


def assert_bit_exact(kernel: Kernel, W: Array, x: Array, threads: int) -> None:
    np.testing.assert_array_equal(
        run(kernel, W, x, threads).view(np.uint32),
        kernel.model(W, x, threads).view(np.uint32),
    )


def random_problem(m: int, n: int, seed: int) -> tuple[Array, Array]:
    rng = np.random.default_rng(seed)
    return bf16(rng.standard_normal((m, n)) * 0.02), bf16(rng.standard_normal(n))


@kernels
@pytest.mark.parametrize(
    ("m", "n", "threads"),
    [
        (1, 1, 32),
        (3, 5, 64),
        (7, 100, 32),
        (64, 1000, 32),
        (5, 17, 1024),
        (33, 4096, 128),
        (256, 4096, 256),
        (256, 4100, 256),
        (16, 14336, 512),
    ],
)
def test_bit_exact_with_model(kernel: Kernel, m: int, n: int, threads: int) -> None:
    W, x = random_problem(m, n, seed=m * n + threads)
    assert_bit_exact(kernel, W, x, threads)


@pytest.mark.parametrize("threads", [1, 2, 4, 16])
def test_naive_small_thread_counts(threads: int) -> None:
    W, x = random_problem(9, 77, seed=threads)
    assert_bit_exact(KERNELS[0], W, x, threads)


@kernels
@settings(max_examples=50)
@given(gemv_problems(max_rows=32, max_cols=600), st.sampled_from([32, 64, 128, 256]))
def test_bit_exact_with_model_random(kernel: Kernel, p: Problem, threads: int) -> None:
    assert_bit_exact(kernel, p.W, p.x, threads)


def test_coalesced_unaligned_pointer_uses_scalar_path() -> None:
    """A contiguous view at an odd offset can't take 16-byte loads; results must not change."""
    m, n, threads = 8, 4096, 128
    W, x = random_problem(m, n, seed=3)
    flat = torch.zeros(m * n + 1, dtype=torch.bfloat16, device="cuda")
    flat[1:] = to_cuda(W).flatten()
    W_view = flat[1:].view(m, n)
    assert W_view.is_contiguous() and W_view.data_ptr() % 16 != 0
    y = gemv_coalesced(W_view, to_cuda(x), threads).cpu().numpy()
    np.testing.assert_array_equal(
        y.view(np.uint32), emulate_gemv_coalesced(W, x, threads).view(np.uint32)
    )


@kernels
def test_signed_zero_matches_model(kernel: Kernel) -> None:
    W = np.zeros((2, 64), dtype=np.float32)
    x = -np.ones(64, dtype=np.float32)
    for threads in (32, 64):
        assert_bit_exact(kernel, W, x, threads)


@kernels
def test_checker_passes_gpu_output(kernel: Kernel) -> None:
    m, n, threads = 512, 4100, 256
    W, x = random_problem(m, n, seed=7)
    checker = BlockedChecker(W, x, 64, tree_height(n, threads, kernel.vec))
    y = run(kernel, W, x, threads)
    assert checker.verdict(y, np.random.default_rng(0), checks=8).all()


@kernels
def test_empty_output(kernel: Kernel) -> None:
    W = torch.zeros((0, 8), dtype=torch.bfloat16, device="cuda")
    x = torch.zeros(8, dtype=torch.bfloat16, device="cuda")
    assert kernel.run(W, x, 32).shape == (0,)


@kernels
@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda W, x, k: (W.float(), x.float(), 32), "bfloat16"),
        (lambda W, x, k: (W.t(), x[:4], 32), "contiguous"),
        (lambda W, x, k: (W, x[:3], 32), "shape"),
        (lambda W, x, k: (W, x, 48), "power of two"),
        (lambda W, x, k: (W, x, 2048), "power of two"),
        (lambda W, x, k: (W.cpu(), x.cpu(), 32), "CUDA tensors"),
    ],
)
def test_rejects_bad_inputs(kernel: Kernel, make: Callable[..., Any], match: str) -> None:
    W = torch.ones((4, 8), dtype=torch.bfloat16, device="cuda")
    x = torch.ones(8, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match=match):
        kernel.run(*make(W, x, kernel))


def test_coalesced_rejects_fewer_than_a_warp() -> None:
    W = torch.ones((4, 8), dtype=torch.bfloat16, device="cuda")
    x = torch.ones(8, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match=r"\[32, 1024\]"):
        gemv_coalesced(W, x, 16)
