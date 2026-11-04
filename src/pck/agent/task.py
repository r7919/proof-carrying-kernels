"""GEMV tasks: a weight matrix (fixed per task, as if loaded) and an activation distribution."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from pck.fp import bf16

Dist = Literal["gauss", "student-t", "outliers", "mixed-scale", "real"]


def activations(dist: Dist, n: int, rng: np.random.Generator) -> NDArray[np.float32]:
    """BF16 activations: Gaussian, Student-t (3 dof), LLM-like outliers (1% of channels
    100x larger) or magnitudes spread over 40 binades."""
    if dist == "real":
        raise ValueError("real activations come from a captured layer (Task.source)")
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
    source: str | None = None
    """For ``dist="real"``: "<capture name>/<layer>", e.g. "qwen2.5-0.5b-l12/down_proj"."""

    def __post_init__(self) -> None:
        if (self.dist == "real") != (self.source is not None):
            raise ValueError("a task has a source exactly when its dist is 'real'")

    @property
    def key(self) -> str:
        """Identifies the task's weights and input distribution (timing-table key)."""
        return f"{self.m}x{self.n}-{self.source or self.dist}"

    def _real(self) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int64]]:
        from pck.agent.real import load

        assert self.source is not None
        name, layer = self.source.split("/")
        W, X, prompt = load(name, layer)
        if W.shape != (self.m, self.n):
            raise ValueError(f"{self.source} has shape {W.shape}, task says {(self.m, self.n)}")
        return W, X, prompt

    @cached_property
    def W(self) -> NDArray[np.float32]:
        """The task's BF16 weights: the captured layer's, or N(0, 0.02^2)."""
        if self.source is not None:
            return self._real()[0]
        rng = np.random.default_rng([self.seed, self.m, self.n])
        return bf16(rng.standard_normal((self.m, self.n)) * 0.02)

    @cached_property
    def fixed_x(self) -> NDArray[np.float32]:
        """The one input the weak-test regime always uses."""
        if self.source is not None:
            return self.fresh_x(np.random.default_rng([self.seed, 1]))
        return activations(self.dist, self.n, np.random.default_rng([self.seed, 1]))

    def fresh_x(self, rng: np.random.Generator) -> NDArray[np.float32]:
        """A new input: a random captured activation (of the prompts episodes may see) or a
        draw from the distribution."""
        if self.source is not None:
            from pck.agent.real import split

            _, X, prompt = self._real()
            seen, _ = split(prompt)
            x: NDArray[np.float32] = X[int(rng.choice(seen))]
            return x
        return activations(self.dist, self.n, rng)

    def held_out_x(self) -> list[NDArray[np.float32]]:
        """Captured activations no episode sees (empty for synthetic tasks)."""
        if self.source is None:
            return []
        from pck.agent.real import split

        _, X, prompt = self._real()
        return [X[i] for i in split(prompt)[1]]
