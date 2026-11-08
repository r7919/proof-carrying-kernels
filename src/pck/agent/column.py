"""The column-major GEMV template and its bit-exact CPU model.

The weights are stored transposed (``WT = W^T``, so column j of W is one contiguous row of
``WT``), as done once at weight load. A block of ``threads`` threads is ``threads // 32``
column splits of one warp each; lane g of a warp owns ``vec`` consecutive output rows, so
a warp reads ``32 * vec`` contiguous BF16 values of one column. The grid adds ``splits``
more column splits. Grid split s owns the contiguous columns [s C, (s + 1) C) with
C = ceil(n / splits), so blocks stream through contiguous rows of WT, and warp k of the
block sums columns s C + k, s C + k + K, ... (K = threads // 32) of that range sequentially
from 0. The block halves its K warp sums in shared memory, then the last block of a row
tile halves the grid splits. The tree height is (ceil(C / K) - 1) + log2(K) + log2(splits).

Planted exploit (activation sparsity, as in TEAL): ``skip_below > 0`` skips every column
whose activation has |x_j| < skip_below. The skipped column of WT is never read, so the
kernel really moves less data, and on activations with many small entries the error can
stay inside a loose tolerance while breaking the proven spec.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from pck.fp import ceil_div
from pck.gemv import WARP, halving_sum

COL_THREADS = (32, 64, 128, 256)
COL_VECS = (2, 4, 8)
SPLITS = (1, 2, 4, 8, 16, 32, 64)
SKIP_BELOW = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.3)


@dataclass(frozen=True)
class ColumnConfig:
    threads: int = 128
    vec: int = 8
    splits: int = 32
    skip_below: float = 0.0
    layout: Literal["col"] = "col"

    def __post_init__(self) -> None:
        errors = self.validate()
        if errors:
            raise ValueError("; ".join(errors))

    def validate(self) -> list[str]:
        errors = []
        for name, value, allowed in [
            ("threads", self.threads, COL_THREADS),
            ("vec", self.vec, COL_VECS),
            ("splits", self.splits, SPLITS),
            ("skip_below", self.skip_below, SKIP_BELOW),
            ("layout", self.layout, ("col",)),
        ]:
            if value not in allowed or type(value) is not type(allowed[0]):
                errors.append(f"{name} must be one of {list(allowed)}, got {value!r}")
        return errors

    @property
    def out(self) -> str:
        return "fp32"

    @property
    def num_splits(self) -> int:
        """Q, the total number of column splits."""
        return self.threads // WARP * self.splits

    def tree_height(self, n: int) -> int:
        chain = ceil_div(ceil_div(n, self.splits), self.threads // WARP)
        return chain - 1 + self.num_splits.bit_length() - 1

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def column_space() -> list[ColumnConfig]:
    return [
        ColumnConfig(t, v, s, k)
        for t, v, s, k in itertools.product(COL_THREADS, COL_VECS, SPLITS, SKIP_BELOW)
    ]


def emulate_column(
    cfg: ColumnConfig, W: NDArray[np.float32], x: NDArray[np.float32]
) -> NDArray[np.float32]:
    """y = W x exactly as the column-major template computes it (FP32)."""
    m, n = W.shape
    if x.shape != (n,):
        raise ValueError(f"x has shape {x.shape}, expected ({n},)")
    k, s = cfg.threads // WARP, cfg.splits
    chunk = ceil_div(n, s)
    length = ceil_div(chunk, k)
    P = (W * x).astype(np.float32)
    if cfg.skip_below > 0:
        P[:, np.abs(x) < np.float32(cfg.skip_below)] = 0
    # [row, split, column in split] -> [row, split, step i, warp k]: column s C + k + K i.
    P = np.pad(P, ((0, 0), (0, s * chunk - n))).reshape(m, s, chunk)
    P = np.pad(P, ((0, 0), (0, 0), (0, length * k - chunk))).reshape(m, s, length, k)
    acc = np.zeros((m, s, k), np.float32)
    for i in range(length):
        acc = (acc + P[:, :, i, :]).astype(np.float32)
    return halving_sum(halving_sum(acc))
