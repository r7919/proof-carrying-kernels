"""Python entry points for the CUDA GEMV kernels (compiled on first use)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from pck.cuda._ext import extension

if TYPE_CHECKING:
    from pck.agent.column import ColumnConfig
    from pck.agent.config import KernelConfig


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


def template_flags(cfg: KernelConfig) -> int:
    """The template kernel's exploit flags for ``cfg``."""
    return (
        (cfg.acc == "bf16")
        | (cfg.partials == "bf16") << 1
        | (cfg.out == "bf16") << 2
        | int(cfg.skip_tail) << 3
    )


def gemv_template(W: torch.Tensor, x: torch.Tensor, cfg: KernelConfig) -> torch.Tensor:
    """y = W x with the agent's template instantiated with ``cfg`` (default slot).

    Matches :func:`pck.agent.config.emulate` bit for bit.
    """
    y: torch.Tensor = extension().gemv_template(
        W, x, cfg.threads, cfg.rows_per_block, cfg.vec, cfg.unroll, template_flags(cfg)
    )
    return y


def gemv_column(WT: torch.Tensor, x: torch.Tensor, cfg: ColumnConfig) -> torch.Tensor:
    """y = W x with the column-major template, from ``WT = W.t().contiguous()``.

    Matches :func:`pck.agent.column.emulate_column` bit for bit.
    """
    y: torch.Tensor = extension().gemv_column(
        WT, x, cfg.threads, cfg.vec, cfg.splits, cfg.skip_below
    )
    return y
