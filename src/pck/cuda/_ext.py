"""Loads the CUDA extension, compiling it on first use."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

_CSRC = Path(__file__).parent / "csrc"
SOURCES = ("bindings.cpp", "gemv_kernel.cu", "checker_kernel.cu", "sampling_kernel.cu")


@functools.cache
def extension() -> Any:
    from torch.utils.cpp_extension import load

    return load(
        name="pck_kernels",
        sources=[str(_CSRC / s) for s in SOURCES],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--fmad=false", "-lineinfo"],
    )
