"""The FP8 (E4M3) GEMV: bit-exact with the coalesced CPU model at vec = 16, and checked
by the trusted checker against its FP8 weights (exact in BF16)."""

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from pck.cuda import GpuBlockedChecker, gemv_fp8, quantize_fp8
from pck.fp import tree_height
from pck.gemv import emulate_gemv_coalesced

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
]

VEC = 16


def problem(m: int, n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    W = (torch.randn(m, n, generator=g) * 0.02).bfloat16().cuda()
    x = torch.randn(n, generator=g).bfloat16().cuda()
    return W, x


def assert_bit_exact(Q: torch.Tensor, x: torch.Tensor, threads: int) -> None:
    z = gemv_fp8(Q, x, threads).cpu().numpy()
    model = emulate_gemv_coalesced(
        Q.float().cpu().numpy(), x.float().cpu().numpy(), threads, vec=VEC
    )
    np.testing.assert_array_equal(z.view(np.uint32), model.view(np.uint32))


@pytest.mark.parametrize(
    ("m", "n", "threads"),
    [(1, 1, 32), (3, 5, 64), (7, 100, 32), (5, 17, 1024), (33, 4096, 128), (256, 4100, 256)],
)
def test_bit_exact_with_model(m: int, n: int, threads: int) -> None:
    W, x = problem(m, n, seed=m * n + threads)
    assert_bit_exact(quantize_fp8(W)[0], x, threads)


@settings(max_examples=40, deadline=None)
@given(
    st.integers(1, 32),
    st.integers(1, 600),
    st.sampled_from([32, 64, 128, 256]),
    st.integers(0, 2**31 - 1),
)
def test_every_fp8_value_bit_exact(m: int, n: int, threads: int, seed: int) -> None:
    """Raw E4M3 bit patterns (every finite value, subnormals included)."""
    g = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 256, (m, n), generator=g, dtype=torch.uint8)
    bits[(bits & 0x7F) == 0x7F] = 0  # E4M3 NaN
    Q = bits.view(torch.float8_e4m3fn).cuda()
    x = torch.randn(n, generator=g) * 2.0 ** torch.randint(-8, 8, (n,), generator=g)
    assert_bit_exact(Q, x.bfloat16().cuda(), threads)


def test_unaligned_pointer_uses_scalar_path() -> None:
    m, n, threads = 8, 4096, 128
    W, x = problem(m, n, seed=3)
    Q = quantize_fp8(W)[0]
    flat = torch.zeros(m * n + 1, dtype=torch.uint8, device="cuda")
    flat[1:] = Q.view(torch.uint8).flatten()
    view = flat[1:].view(torch.float8_e4m3fn).view(m, n)
    assert view.is_contiguous() and view.data_ptr() % 16 != 0
    assert torch.equal(gemv_fp8(view, x, threads), gemv_fp8(Q, x, threads))


def test_quantize_fp8_scales_rows() -> None:
    W, _ = problem(16, 300, seed=1)
    W[3] = 0
    Q, s = quantize_fp8(W)
    assert Q.dtype == torch.float8_e4m3fn and s.shape == (16,)
    assert s[3] == 1 and not Q[3].float().any()
    assert torch.allclose(Q.float().abs().amax(dim=1)[s != 1], torch.tensor(448.0).cuda())
    rel = (s[:, None] * Q.float() - W.float()).norm() / W.float().norm()
    assert rel < 0.05


def test_rejects_bf16_weights() -> None:
    W, x = problem(4, 32, seed=0)
    with pytest.raises(RuntimeError, match="Float8_e4m3fn"):
        gemv_fp8(W, x)


def checker_for(Q: torch.Tensor, threads: int) -> GpuBlockedChecker:
    h = tree_height(Q.shape[1], threads, VEC)
    g = torch.Generator().manual_seed(0)
    return GpuBlockedChecker(Q.to(torch.bfloat16), 128, h, checks=8, generator=g)


def test_checker_passes_fp8_output() -> None:
    W, x = problem(1024, 4096, seed=7)
    Q = quantize_fp8(W)[0]
    checker = checker_for(Q, 256)
    for seed in range(5):
        x = problem(1, 4096, seed=100 + seed)[1]
        assert checker.verdict(x, gemv_fp8(Q, x, 256)).all()


def test_checker_catches_faults_in_fp8_output() -> None:
    W, x = problem(1024, 4096, seed=8)
    Q = quantize_fp8(W)[0]
    checker = checker_for(Q, 256)
    z = gemv_fp8(Q, x, 256)
    rounded = z.bfloat16().float()  # BF16 output, a planted shortcut
    assert not checker.verdict(x, rounded).all()
    flipped = z.clone()
    flipped.view(torch.int32)[5] ^= 1 << 30  # exponent bit flip
    assert not checker.verdict(x, flipped)[0]
