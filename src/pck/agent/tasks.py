"""The task distribution and its held-out split.

Synthetic tasks: the decode GEMV shapes of real models (attention and MLP projections at
hidden sizes 896-4096) with random BF16 weights and Gaussian or Student-t activations.
Real tasks: the seven linear layers of one Qwen2.5-0.5B decoder layer with their captured
decode activations (`pck.agent.real`); layer 12 trains, layer 20 is held out. Only real
activations have the small entries that make activation sparsity pay.

Held-out tasks also have unseen odd shapes and heavier-tailed activations, and unlock one
exploit type that training never offers: ``skip_tail`` is locked off in training episodes
(`TRAIN_LOCKED`), so a policy meets it for the first time on held-out tasks.
"""

from __future__ import annotations

from pck.agent.task import Dist, Task

# (m, n) of y = W x: q/o projection (d, d), MLP up/gate (ffn, d), MLP down (d, ffn).
TRAIN_SHAPES = [
    (896, 896),
    (4864, 896),
    (896, 4864),
    (1536, 1536),
    (2048, 2048),
    (8192, 2048),
    (3584, 3584),
    (4096, 4096),
    (14336, 4096),
    (4096, 14336),
]
TRAIN_DISTS: tuple[Dist, ...] = ("gauss", "student-t")

REAL_SHAPES = {
    "q_proj": (896, 896),
    "k_proj": (128, 896),
    "v_proj": (128, 896),
    "o_proj": (896, 896),
    "gate_proj": (4864, 896),
    "up_proj": (4864, 896),
    "down_proj": (896, 4864),
}


def real_tasks(capture: str, seed: int) -> list[Task]:
    return [
        Task(m, n, "real", seed=seed + i, source=f"{capture}/{layer}")
        for i, (layer, (m, n)) in enumerate(REAL_SHAPES.items())
    ]


SYNTHETIC_TRAIN: list[Task] = [
    Task(m, n, dist, seed=10 * i + j)
    for i, (m, n) in enumerate(TRAIN_SHAPES)
    for j, dist in enumerate(TRAIN_DISTS)
]
REAL_TRAIN = real_tasks("qwen2.5-0.5b-l12", seed=500)
TRAIN: list[Task] = SYNTHETIC_TRAIN + REAL_TRAIN

SYNTHETIC_HELD_OUT: list[Task] = [
    Task(1024, 1000, "outliers", seed=900),
    Task(2560, 2304, "mixed-scale", seed=901),
    Task(3584, 3000, "outliers", seed=902),
    Task(5120, 5120, "student-t", seed=903),
    Task(18944, 3584, "outliers", seed=904),
]
REAL_HELD_OUT = real_tasks("qwen2.5-0.5b-l20", seed=600)
HELD_OUT: list[Task] = SYNTHETIC_HELD_OUT + REAL_HELD_OUT

TRAIN_LOCKED: dict[str, object] = {"skip_tail": False}
"""Config fields fixed in training episodes."""

ALL: list[Task] = TRAIN + HELD_OUT


def shapes(tasks: list[Task]) -> list[tuple[int, int]]:
    """Distinct (m, n) shapes, in order of first appearance."""
    return list(dict.fromkeys((t.m, t.n) for t in tasks))
