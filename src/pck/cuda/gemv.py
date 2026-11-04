"""Python entry points for the CUDA GEMV kernels (compiled on first use)."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import torch

_CSRC = Path(__file__).parent / "csrc"


@functools.cache
def _extension() -> Any:
    from torch.utils.cpp_extension import load

    return load(
        name="pck_gemv",
        sources=[str(_CSRC / "gemv.cpp"), str(_CSRC / "gemv_kernel.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--fmad=false", "-lineinfo"],
    )


def gemv_naive(W: torch.Tensor, x: torch.Tensor, threads: int = 256) -> torch.Tensor:
    """y = W x for contiguous BF16 CUDA tensors, FP32 accumulation and FP32 output.

    Uses the same summation tree as :func:`pck.gemv.emulate_gemv` with the same
    ``threads``, so the result matches it bit for bit.
    """
    y: torch.Tensor = _extension().gemv_naive(W, x, threads)
    return y
