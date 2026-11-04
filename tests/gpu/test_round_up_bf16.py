"""round_up_bf16 runs on CPU tensors, so these tests need torch but not a GPU."""

import math

import ml_dtypes
import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from pck.cuda.checker import round_up_bf16

BF16_MAX = float(ml_dtypes.finfo(ml_dtypes.bfloat16).max)

pytestmark = pytest.mark.gpu


@given(st.lists(st.floats(0, BF16_MAX, allow_nan=False), min_size=1, max_size=50))
def test_never_rounds_down_and_is_tight(values: list[float]) -> None:
    v = torch.tensor(values, dtype=torch.float64)
    up = round_up_bf16(v)
    assert up.dtype == torch.bfloat16
    up64 = up.double()
    assert bool((up64 >= v).all())
    # The next BF16 value below the result is already below the input (or it is zero).
    below = np.array(
        [np.nextafter(ml_dtypes.bfloat16(u), ml_dtypes.bfloat16(0)) for u in up64.tolist()],
        dtype=np.float64,
    )
    assert np.all((below < np.array(values)) | (up64.numpy() == 0))


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, 0.0), (1.0, 1.0), (1 + 2**-30, 1 + 2**-7), (1 + 2**-8, 1 + 2**-7), (3.0, 3.0)],
)
def test_known_values(value: float, expected: float) -> None:
    assert float(round_up_bf16(torch.tensor([value], dtype=torch.float64))[0]) == expected


def test_overflow_becomes_inf() -> None:
    assert math.isinf(
        float(round_up_bf16(torch.tensor([BF16_MAX * 1.001], dtype=torch.float64))[0])
    )


def test_rejects_negative_and_wrong_dtype() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        round_up_bf16(torch.tensor([-1.0], dtype=torch.float64))
    with pytest.raises(TypeError, match="float64"):
        round_up_bf16(torch.tensor([1.0]))
