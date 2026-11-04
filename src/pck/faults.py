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


def sharded_topk(
    logits: NDArray[np.float32], k: int, shards: int, per_shard: int
) -> NDArray[np.int64]:
    """An approximate top-k of the kind that silently returns the wrong set.

    Splits the vocabulary into ``shards`` contiguous shards, keeps the ``per_shard`` largest
    of each, then takes the top k of the survivors. Exact when ``per_shard >= k``; when
    ``per_shard < k`` it misses true top-k logits that cluster in one shard.
    """
    a = np.asarray(logits, dtype=np.float32)
    if not 1 <= k <= a.size or shards < 1 or per_shard < 1:
        raise ValueError("need 1 <= k <= len(logits), shards >= 1 and per_shard >= 1")
    candidates = []
    for idx in np.array_split(np.arange(a.size), shards):
        if idx.size:
            order = np.lexsort((idx, -a[idx].astype(np.float64)))
            candidates.append(idx[order[:per_shard]])
    pool = np.concatenate(candidates)
    order = np.lexsort((pool, -a[pool].astype(np.float64)))
    return pool[order[:k]].astype(np.int64)
