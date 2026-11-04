"""Shared Hypothesis strategies for GEMV problems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from hypothesis import strategies as st
from numpy.typing import NDArray

from pck.fp import bf16


@dataclass
class Problem:
    W: NDArray[np.float32]
    x: NDArray[np.float32]
    threads: int


@st.composite
def gemv_problems(
    draw: st.DrawFn, max_rows: int = 64, max_cols: int = 300, row_multiple: int = 1
) -> Problem:
    """Random BF16 GEMV problems with Gaussian or heavy-tailed activations."""
    m = row_multiple * draw(st.integers(1, max(1, max_rows // row_multiple)))
    n = draw(st.integers(1, max_cols))
    threads = 2 ** draw(st.integers(0, 6))
    seed = draw(st.integers(0, 2**32 - 1))
    dist = draw(st.sampled_from(["gauss", "student-t", "mixed-scale"]))
    rng = np.random.default_rng(seed)
    W = bf16(rng.standard_normal((m, n)) * 0.02)
    if dist == "gauss":
        x = rng.standard_normal(n)
    elif dist == "student-t":
        x = rng.standard_t(3, n)
    else:  # magnitudes spread over many binades, with cancellation
        x = rng.standard_normal(n) * 2.0 ** rng.integers(-20, 20, n)
    return Problem(W=W, x=bf16(x), threads=threads)
