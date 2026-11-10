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


FP8_MAX = 448.0
"""Largest finite FP8 E4M3 value."""


def quantize_fp8(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 weights for :func:`gemv_fp8`: (Q, s) with Q in float8_e4m3fn, shape (m, n), and
    s FP32 of shape (m,), so W ~ s[:, None] * Q (one scale per row, max |w| / 448; an
    all-zero row gets scale 1)."""
    amax = W.abs().amax(dim=1).float()
    s = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
    Q = (W.float() / s[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return Q.contiguous(), s


def gemv_fp8(Q: torch.Tensor, x: torch.Tensor, threads: int = 256) -> torch.Tensor:
    """z = Q x for FP8 E4M3 weights ``Q`` (m, n) and BF16 ``x``: FP32 accumulation and
    output, 16 weights per 16-byte load. Each FP8 x BF16 product is exact in FP32, so z
    matches :func:`pck.gemv.emulate_gemv_coalesced` with ``vec=16`` bit for bit, and the
    trusted checker verifies it against Q (FP8 values are exact in BF16). The row
    scales of `quantize_fp8` are applied afterwards: y = s * z."""
    z: torch.Tensor = extension().gemv_fp8(Q, x, threads)
    return z


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
