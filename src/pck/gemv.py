"""Bit-faithful CPU model of a BF16 GEMV kernel, with switches for planted exploits."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pck.fp import bf16, ceil_div

WARP = 32
"""Threads per warp on NVIDIA GPUs."""


def emulate_gemv(
    W: NDArray[np.float32],
    x: NDArray[np.float32],
    threads: int,
    *,
    acc_bf16: bool = False,
    partial_bf16: bool = False,
    drop_tail: int = 0,
) -> NDArray[np.float32]:
    """Compute y = W x the way the kernel does, in FP32.

    ``W`` and ``x`` hold BF16 values (stored as FP32), so every product is exact in
    FP32. Thread t sums a contiguous chunk of ceil(n / threads) products, then a
    pairwise tree reduces across threads.

    Exploits (all violate the accuracy spec):
      acc_bf16:      round the running per-thread accumulator to BF16 after each add.
      partial_bf16:  round per-thread partial sums to BF16 before the tree.
      drop_tail:     skip the last ``drop_tail`` columns.
    """
    m, n = W.shape
    if x.shape != (n,):
        raise ValueError(f"x has shape {x.shape}, expected ({n},)")
    if threads < 1 or threads & (threads - 1):
        raise ValueError(f"threads must be a power of two, got {threads}")
    if not 0 <= drop_tail <= n:
        raise ValueError(f"drop_tail must be in [0, {n}], got {drop_tail}")
    chunk = -(-n // threads)

    P = (W * x).astype(np.float32)
    if drop_tail:
        P[:, n - drop_tail :] = 0
    P = np.pad(P, ((0, 0), (0, chunk * threads - n))).reshape(m, threads, chunk)

    acc = P[:, :, 0].copy()
    for j in range(1, chunk):
        acc = (acc + P[:, :, j]).astype(np.float32)
        if acc_bf16:
            acc = bf16(acc)
    if partial_bf16:
        acc = bf16(acc)
    while acc.shape[1] > 1:
        acc = (acc[:, 0::2] + acc[:, 1::2]).astype(np.float32)
    return acc[:, 0]


def _halving_sum(a: NDArray[np.float32]) -> NDArray[np.float32]:
    """Reduce the last axis (a power of two) by repeated halving: a[i] += a[i + len/2].

    This is the order of a ``__shfl_down_sync`` reduction with offsets len/2, ..., 1.
    """
    while a.shape[-1] > 1:
        half = a.shape[-1] // 2
        a = (a[..., :half] + a[..., half:]).astype(np.float32)
    return a[..., 0]


def emulate_gemv_coalesced(
    W: NDArray[np.float32], x: NDArray[np.float32], threads: int, vec: int = 8
) -> NDArray[np.float32]:
    """Compute y = W x in the order of the coalesced CUDA kernel, in FP32.

    Columns are grouped into vectors of ``vec`` (one 16-byte load for BF16 at
    vec = 8). Thread t owns vectors t, t + threads, t + 2*threads, ... and sums their
    products sequentially, vector by vector, with columns past n as +0. Each warp then
    reduces its 32 lanes by halving, and the warp sums are reduced by halving again.
    The tree height is ``tree_height(n, threads, vec)``.
    """
    m, n = W.shape
    if x.shape != (n,):
        raise ValueError(f"x has shape {x.shape}, expected ({n},)")
    if threads < WARP or threads & (threads - 1):
        raise ValueError(f"threads must be a power of two >= {WARP}, got {threads}")
    if vec < 1:
        raise ValueError(f"vec must be positive, got {vec}")
    per_thread = ceil_div(ceil_div(n, vec), threads)

    P = (W * x).astype(np.float32)
    P = np.pad(P, ((0, 0), (0, per_thread * threads * vec - n)))
    # Column ((v * threads) + t) * vec + k is term v * vec + k of thread t.
    by_vector = P.reshape(m, per_thread, threads, vec).transpose(0, 2, 1, 3)
    terms = by_vector.reshape(m, threads, per_thread * vec)

    acc = terms[:, :, 0].copy()
    for j in range(1, terms.shape[2]):
        acc = (acc + terms[:, :, j]).astype(np.float32)
    warp_sums = _halving_sum(acc.reshape(m, threads // WARP, WARP))
    return _halving_sum(warp_sums)


def reference_gemv(W: NDArray[np.floating], x: NDArray[np.floating]) -> NDArray[np.float64]:
    """FP64 reference for y = W x (error ~n * 2^-53, negligible against FP32 bounds)."""
    return W.astype(np.float64) @ x.astype(np.float64)


def abs_row_sums(W: NDArray[np.floating], x: NDArray[np.floating]) -> NDArray[np.float64]:
    """Per-row sum_j |W_ij x_j|: the quantity the rounding-error bound scales with."""
    return np.abs(W).astype(np.float64) @ np.abs(x).astype(np.float64)
