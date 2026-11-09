"""Real tasks: one decoder layer of an open model, with its captured decode activations.

`harness/capture_layer.py` runs the model, records the input of every linear layer of one
decoder layer at each decode step, and saves the weights and inputs (as BF16 bits) to
``$PCK_CACHE/<name>.npz`` (default ``~/.cache/pck``). The file is regenerated, not
committed (about 30 MB). Activations of the first prompts are the ones episodes see; those
of the last prompts are kept for the hidden adversarial suite.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

LAYERS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
HELD_OUT_PROMPTS = 2
"""The last prompts, whose activations only the hidden suite uses."""


def cache_dir() -> Path:
    return Path(os.environ.get("PCK_CACHE", Path.home() / ".cache" / "pck"))


def to_bits(a: NDArray[np.float32]) -> NDArray[np.uint16]:
    """BF16 values stored as FP32 -> their 16-bit patterns (exact)."""
    return (np.ascontiguousarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def from_bits(b: NDArray[np.uint16]) -> NDArray[np.float32]:
    return (b.astype(np.uint32) << 16).view(np.float32)


@cache
def load(
    name: str, layer: str
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int64]]:
    """(W, X, prompt) for one linear layer: weights (m, n), inputs (N, n), prompt index per
    input. Raises FileNotFoundError with the command that creates the file."""
    path = cache_dir() / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; create it with: uv run python -m harness.capture_layer"
        )
    with np.load(path) as data:
        return (
            from_bits(data[f"W_{layer}"]),
            from_bits(data[f"X_{layer}"]),
            data[f"P_{layer}"].astype(np.int64),
        )


def split(prompt: NDArray[np.int64]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Indices of the inputs episodes may see, and of the held-out ones."""
    cut = int(np.max(prompt)) + 1 - HELD_OUT_PROMPTS
    return np.flatnonzero(prompt < cut), np.flatnonzero(prompt >= cut)
