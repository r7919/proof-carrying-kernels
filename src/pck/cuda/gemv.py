"""Python entry points for the CUDA GEMV kernels (compiled on first use)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
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


def gemv_template(
    W: torch.Tensor,
    x: torch.Tensor,
    cfg: KernelConfig,
    q8: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """y = W x with the agent's template instantiated with ``cfg`` (default slot); with
    ``cfg.weights == "int8"``, from ``q8 = (Wq, S)``, the INT8 copy on the GPU (`row_q8`).

    Matches :func:`pck.agent.config.emulate` bit for bit.
    """
    if cfg.weights == "int8":
        if q8 is None:
            raise ValueError("an INT8 config needs the INT8 copy of W (q8)")
        y: torch.Tensor = extension().gemv_template_q8(
            *q8, x, cfg.threads, cfg.rows_per_block, cfg.vec, cfg.unroll
        )
        return y
    y = extension().gemv_template(
        W, x, cfg.threads, cfg.rows_per_block, cfg.vec, cfg.unroll, template_flags(cfg)
    )
    return y


def row_q8(W: np.ndarray, device: torch.device | str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """The INT8 copy (Wq, S) of W for `gemv_template`, quantized on the CPU by
    :func:`pck.agent.config.quantize_rows`, so the GPU reads exactly what the CPU model
    dequantizes."""
    from pck.agent.config import quantize_rows

    Wq, S = quantize_rows(np.asarray(W, dtype=np.float32))
    return torch.from_numpy(Wq).to(device), torch.from_numpy(S).to(device)


_TICKETS: dict[torch.device, torch.Tensor] = {}


def _tickets(device: torch.device, count: int) -> torch.Tensor:
    """Zeroed merge tickets, reused across calls (the kernel leaves them zero)."""
    t = _TICKETS.get(device)
    if t is None or t.numel() < count:
        t = torch.zeros(max(count, 1024), dtype=torch.int32, device=device)
        _TICKETS[device] = t
    return t


def gemv_column(WT: torch.Tensor, x: torch.Tensor, cfg: ColumnConfig) -> torch.Tensor:
    """y = W x with the column-major template, from ``WT = W.t().contiguous()``.

    Matches :func:`pck.agent.column.emulate_column` bit for bit. Calls on one device must
    not overlap (they share the merge tickets), so use a single stream.
    """
    tiles = -(-WT.shape[1] // (32 * cfg.vec))
    y: torch.Tensor = extension().gemv_column(
        WT, x, _tickets(WT.device, tiles), cfg.threads, cfg.vec, cfg.splits, cfg.skip_below
    )
    return y
