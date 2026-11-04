"""Python entry points for the CUDA GEMV kernels (compiled on first use)."""

from __future__ import annotations

import torch

from pck.cuda._ext import extension


def gemv_naive(W: torch.Tensor, x: torch.Tensor, threads: int = 256) -> torch.Tensor:
    """y = W x for contiguous BF16 CUDA tensors, FP32 accumulation and FP32 output.

    Uses the same summation tree as :func:`pck.gemv.emulate_gemv` with the same
    ``threads``, so the result matches it bit for bit.
    """
    y: torch.Tensor = extension().gemv_naive(W, x, threads)
    return y


def gemv_coalesced(W: torch.Tensor, x: torch.Tensor, threads: int = 256) -> torch.Tensor:
    """y = W x with coalesced 16-byte loads and a warp-shuffle reduction.

    Same contract as :func:`gemv_naive`, with ``threads`` a power of two in [32, 1024].
    Matches :func:`pck.gemv.emulate_gemv_coalesced` bit for bit.
    """
    y: torch.Tensor = extension().gemv_coalesced(W, x, threads)
    return y
