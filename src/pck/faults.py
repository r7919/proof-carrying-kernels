"""Fault injection on kernel outputs."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def flip_bit(y: NDArray[np.float32], index: int, bit: int) -> NDArray[np.float32]:
    """Return a copy of FP32 array ``y`` with bit ``bit`` of element ``index`` flipped.

    Bits 0-22 are the mantissa, 23-30 the exponent and 31 the sign.
    """
    if y.dtype != np.float32:
        raise TypeError(f"flip_bit expects float32, got {y.dtype}")
    if not 0 <= bit < 32:
        raise ValueError(f"bit must be in [0, 32), got {bit}")
    f = y.copy()
    f.view(np.uint32)[index] ^= np.uint32(1 << bit)
    return f
