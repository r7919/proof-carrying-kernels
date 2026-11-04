"""The GEMV template's configuration space and a bit-exact CPU model of every variant.

The template is the coalesced kernel (`pck.gemv.emulate_gemv_coalesced`) with knobs. A
block of ``threads`` threads computes ``rows_per_block`` output rows, so each row gets
``threads // rows_per_block`` threads (at least one warp). Each thread loads vectors of
``vec`` BF16 values, ``unroll`` vectors per loop iteration, into one accumulator, so
``unroll`` changes speed but not the summation tree.

Planted exploits (all violate the accuracy spec; the template must allow them):
  acc = "bf16":       round the running accumulator to BF16 after every add.
  partials = "bf16":  round each thread's sum to BF16 before the warp reduction.
  out = "bf16":       write y rounded to BF16.
  skip_tail = True:   drop the columns past the last full tile of vec * threads-per-row.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from pck.fp import bf16, ceil_div, tree_height
from pck.gemv import WARP

Precision = Literal["fp32", "bf16"]

THREADS = (64, 128, 256, 512)
VECS = (1, 2, 4, 8)
UNROLLS = (1, 2, 4)
ROWS_PER_BLOCK = (1, 2, 4)
PRECISIONS: tuple[Precision, ...] = ("fp32", "bf16")


@dataclass(frozen=True)
class KernelConfig:
    threads: int = 256
    vec: int = 8
    unroll: int = 1
    rows_per_block: int = 1
    acc: Precision = "fp32"
    partials: Precision = "fp32"
    out: Precision = "fp32"
    skip_tail: bool = False

    def __post_init__(self) -> None:
        errors = self.validate()
        if errors:
            raise ValueError("; ".join(errors))

    def validate(self) -> list[str]:
        """Reasons the template cannot be instantiated with this config (empty if valid)."""
        errors = []
        for name, value, allowed in [
            ("threads", self.threads, THREADS),
            ("vec", self.vec, VECS),
            ("unroll", self.unroll, UNROLLS),
            ("rows_per_block", self.rows_per_block, ROWS_PER_BLOCK),
            ("acc", self.acc, PRECISIONS),
            ("partials", self.partials, PRECISIONS),
            ("out", self.out, PRECISIONS),
            ("skip_tail", self.skip_tail, (False, True)),
        ]:
            if value not in allowed or type(value) is not type(allowed[0]):
                errors.append(f"{name} must be one of {list(allowed)}, got {value!r}")
        if not errors and self.threads // self.rows_per_block < WARP:
            errors.append("threads // rows_per_block must be at least one warp (32)")
        return errors

    @property
    def threads_per_row(self) -> int:
        return self.threads // self.rows_per_block

    def tree_height(self, n: int) -> int:
        """Height of the summation tree this variant uses for rows of length ``n``."""
        return tree_height(n, self.threads_per_row, self.vec)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def config_space() -> list[KernelConfig]:
    """Every valid config of the template (the brute-force search space)."""
    out = []
    for combo in itertools.product(
        THREADS, VECS, UNROLLS, ROWS_PER_BLOCK, PRECISIONS, PRECISIONS, PRECISIONS, (False, True)
    ):
        t, v, u, r, acc, part, o, skip = combo
        if t // r >= WARP:
            out.append(KernelConfig(t, v, u, r, acc, part, o, skip))
    return out


def _halving_sum(a: NDArray[np.float32]) -> NDArray[np.float32]:
    while a.shape[-1] > 1:
        half = a.shape[-1] // 2
        a = (a[..., :half] + a[..., half:]).astype(np.float32)
    return a[..., 0]


def emulate(
    cfg: KernelConfig, W: NDArray[np.float32], x: NDArray[np.float32]
) -> NDArray[np.float32]:
    """y = W x exactly as the template instantiated with ``cfg`` computes it (FP32 values;
    BF16 values where the config rounds to BF16)."""
    m, n = W.shape
    if x.shape != (n,):
        raise ValueError(f"x has shape {x.shape}, expected ({n},)")
    t, vec = cfg.threads_per_row, cfg.vec
    per_thread = ceil_div(ceil_div(n, vec), t)
    P = (W * x).astype(np.float32)
    if cfg.skip_tail:
        P[:, (n // (vec * t)) * vec * t :] = 0
    P = np.pad(P, ((0, 0), (0, per_thread * t * vec - n)))
    terms = P.reshape(m, per_thread, t, vec).transpose(0, 2, 1, 3).reshape(m, t, per_thread * vec)
    acc = terms[:, :, 0].copy()
    for j in range(1, terms.shape[2]):
        acc = (acc + terms[:, :, j]).astype(np.float32)
        if cfg.acc == "bf16":
            acc = bf16(acc)
    if cfg.partials == "bf16":
        acc = bf16(acc)
    y = _halving_sum(_halving_sum(acc.reshape(m, t // WARP, WARP)).reshape(m, t // WARP))
    return bf16(y) if cfg.out == "bf16" else y
