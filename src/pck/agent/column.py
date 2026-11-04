"""The column-major GEMV template and its bit-exact CPU model.

The weights are stored transposed (``WT = W^T``, so column j of W is one contiguous row of
``WT``), as done once at weight load. A block of ``threads`` threads is ``threads // 32``
column splits of one warp each; lane g of a warp owns ``vec`` consecutive output rows, so
a warp reads ``32 * vec`` contiguous BF16 values of one column. The grid adds ``splits``
more column splits, so there are Q = (threads // 32) * splits splits in all, and split
q = s * (threads // 32) + k sums columns q, q + Q, q + 2Q, ... sequentially from 0. The
block halves its splits in shared memory, and a second pass halves the grid splits.
The tree height is (ceil(n / Q) - 1) + log2(Q).

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

from pck.fp import ceil_div, tree_height
from pck.gemv import WARP

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
        return tree_height(n, self.num_splits)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def column_space() -> list[ColumnConfig]:
    return [
        ColumnConfig(t, v, s, k)
        for t, v, s, k in itertools.product(COL_THREADS, COL_VECS, SPLITS, SKIP_BELOW)
    ]


def _halving_sum(a: NDArray[np.float32]) -> NDArray[np.float32]:
    while a.shape[-1] > 1:
        half = a.shape[-1] // 2
        a = (a[..., :half] + a[..., half:]).astype(np.float32)
    return a[..., 0]


def emulate_column(
    cfg: ColumnConfig, W: NDArray[np.float32], x: NDArray[np.float32]
) -> NDArray[np.float32]:
    """y = W x exactly as the column-major template computes it (FP32)."""
    m, n = W.shape
    if x.shape != (n,):
        raise ValueError(f"x has shape {x.shape}, expected ({n},)")
    k, s = cfg.threads // WARP, cfg.splits
    q = k * s
    length = ceil_div(n, q)
    P = (W * x).astype(np.float32)
    if cfg.skip_below > 0:
        P[:, np.abs(x) < np.float32(cfg.skip_below)] = 0
    P = np.pad(P, ((0, 0), (0, length * q - n))).reshape(m, length, q)
    acc = np.zeros((m, q), np.float32)
    for i in range(length):
        acc = (acc + P[:, i, :]).astype(np.float32)
    return _halving_sum(_halving_sum(acc.reshape(m, s, k)))
