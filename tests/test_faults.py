import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pck.faults import flip_bit


@given(
    st.lists(st.floats(width=32, allow_nan=False), min_size=1, max_size=10),
    st.data(),
    st.integers(0, 31),
)
def test_double_flip_is_identity(values: list[float], data: st.DataObject, bit: int) -> None:
    y = np.array(values, dtype=np.float32)
    i = data.draw(st.integers(0, y.size - 1))
    once = flip_bit(y, i, bit)
    assert once.tobytes() != y.tobytes()
    assert flip_bit(once, i, bit).tobytes() == y.tobytes()


def test_sign_bit_negates_and_leaves_input() -> None:
    y = np.array([1.5, -2.0], dtype=np.float32)
    np.testing.assert_array_equal(flip_bit(y, 0, 31), [-1.5, -2.0])
    np.testing.assert_array_equal(y, [1.5, -2.0])


def test_rejects_bad_bit_and_dtype() -> None:
    with pytest.raises(ValueError, match="bit"):
        flip_bit(np.zeros(1, np.float32), 0, 32)
    with pytest.raises(TypeError, match="float32"):
        flip_bit(np.zeros(1, np.float64), 0, 1)
