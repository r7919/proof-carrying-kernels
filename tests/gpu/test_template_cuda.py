import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from pck.agent.column import ColumnConfig, column_space, emulate_column
from pck.agent.config import KernelConfig, emulate, row_space
from pck.cuda import gemv_column, gemv_template
from pck.fp import bf16

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
]

SPACE = row_space()


def to_cuda(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a).to(torch.bfloat16).cuda()


@settings(max_examples=300, deadline=None)
@given(
    st.sampled_from(SPACE),
    st.integers(1, 40),
    st.integers(1, 3000),
    st.integers(0, 2**32 - 1),
)
def test_template_matches_its_cpu_model(cfg: KernelConfig, m: int, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    W = bf16(rng.standard_normal((m, n)) * 0.02)
    x = bf16(rng.standard_normal(n) * 2.0 ** rng.integers(-8, 9, n))
    got = gemv_template(to_cuda(W), to_cuda(x), cfg).cpu().numpy()
    np.testing.assert_array_equal(got, emulate(cfg, W, x))


@pytest.mark.parametrize(
    "cfg", [KernelConfig(vec=v, unroll=u) for v in (1, 2, 4, 8) for u in (1, 2, 4)]
)
def test_unaligned_input_takes_the_scalar_path(cfg: KernelConfig) -> None:
    rng = np.random.default_rng(0)
    W = bf16(rng.standard_normal((8, 1025)) * 0.02)
    x = bf16(rng.standard_normal(1025))
    Wc, xc = to_cuda(W), to_cuda(np.concatenate([[0.0], x]).astype(np.float32))[1:]
    np.testing.assert_array_equal(gemv_template(Wc, xc, cfg).cpu().numpy(), emulate(cfg, W, x))


def test_bad_rows_per_block_is_rejected() -> None:
    W = torch.zeros(4, 64, dtype=torch.bfloat16, device="cuda")
    x = torch.zeros(64, dtype=torch.bfloat16, device="cuda")
    from pck.cuda._ext import extension

    with pytest.raises(RuntimeError, match="launch failed"):
        extension().gemv_template(W, x, 64, 4, 8, 1, 0)  # 16 threads per row


@settings(max_examples=300, deadline=None)
@given(
    st.sampled_from(column_space()),
    st.integers(1, 80),
    st.integers(1, 3000),
    st.integers(0, 2**32 - 1),
)
def test_column_template_matches_its_cpu_model(
    cfg: ColumnConfig, rows: int, n: int, seed: int
) -> None:
    m = rows * cfg.vec  # the kernel needs m % vec == 0
    rng = np.random.default_rng(seed)
    W = bf16(rng.standard_normal((m, n)) * 0.02)
    x = bf16(rng.standard_normal(n) * 2.0 ** rng.integers(-8, 3, n))
    WT = to_cuda(np.ascontiguousarray(W.T))
    got = gemv_column(WT, to_cuda(x), cfg).cpu().numpy()
    np.testing.assert_array_equal(got, emulate_column(cfg, W, x))


def test_column_template_rejects_rows_not_divisible_by_vec() -> None:
    WT = torch.zeros(16, 12, dtype=torch.bfloat16, device="cuda")
    x = torch.zeros(16, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="launch failed"):
        gemv_column(WT, x, ColumnConfig(vec=8))
