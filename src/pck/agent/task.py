"""GEMV tasks: a weight matrix (fixed per task, as if loaded) and an activation distribution."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from pck.fp import bf16

Dist = Literal["gauss", "student-t", "outliers", "mixed-scale"]


def activations(dist: Dist, n: int, rng: np.random.Generator) -> NDArray[np.float32]:
    """BF16 activations: Gaussian, Student-t (3 dof), LLM-like outliers (1% of channels
    100x larger) or magnitudes spread over 40 binades."""
    if dist == "gauss":
        x = rng.standard_normal(n)
    elif dist == "student-t":
        x = rng.standard_t(3, n)
    elif dist == "outliers":
        x = rng.standard_normal(n)
        x[rng.choice(n, max(1, n // 100), replace=False)] *= 100.0
    else:
        x = rng.standard_normal(n) * 2.0 ** rng.integers(-20, 21, n)
    return bf16(x)


@dataclass(frozen=True)
class Task:
    m: int
    n: int
    dist: Dist = "gauss"
    seed: int = 0

    @cached_property
    def W(self) -> NDArray[np.float32]:
        """The task's BF16 weights, N(0, 0.02^2)."""
        rng = np.random.default_rng([self.seed, self.m, self.n])
        return bf16(rng.standard_normal((self.m, self.n)) * 0.02)

    @cached_property
    def fixed_x(self) -> NDArray[np.float32]:
        """The one input the weak-test regime always uses."""
        return activations(self.dist, self.n, np.random.default_rng([self.seed, 1]))

    def fresh_x(self, rng: np.random.Generator) -> NDArray[np.float32]:
        return activations(self.dist, self.n, rng)
