"""The task distribution and its held-out split.

Synthetic tasks: the decode GEMV shapes of real models (attention and MLP projections at
hidden sizes 896-4096) with random BF16 weights and Gaussian or Student-t activations.
Real tasks: the seven linear layers of one Qwen2.5-0.5B decoder layer with their captured
decode activations (`pck.agent.real`); layer 12 trains, layer 20 is held out. Only real
activations have the small entries that make activation sparsity pay.

Real tasks are set in a column-major deployment (weights stored transposed, as
sparsity-aware inference engines such as TEAL and Deja Vu do): `locks` fixes their layout
to "col", so the honest baseline is the best column-major kernel. There, activation
sparsity that a calibrated test accepts is 2-12% faster than honest
(harness/bench_window_out.txt); in a row-major deployment it roughly breaks even.

Error budgets: most tasks use the default spec (tree height h_spec = 64); two training
tasks use a stricter one (32) and two held-out tasks one that training never uses (128).
Each is a copy of a default-budget task (same weights and inputs, so the same timings),
differing only in the budget.

Held-out tasks also have unseen odd shapes and heavier-tailed activations, and unlock one
exploit type that training never offers: ``skip_tail`` is locked off in synthetic training
episodes, so a policy meets it for the first time on held-out tasks.
"""

from __future__ import annotations

from dataclasses import replace

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


def _budget(tasks: list[Task], m: int, n: int, dist: Dist, h_spec: int) -> Task:
    """The default-budget task (m, n, dist) of ``tasks`` at another error budget."""
    sibling = next(t for t in tasks if (t.m, t.n, t.dist) == (m, n, dist))
    return replace(sibling, h_spec=h_spec)


_SYNTHETIC_TRAIN_64: list[Task] = [
    Task(m, n, dist, seed=10 * i + j)
    for i, (m, n) in enumerate(TRAIN_SHAPES)
    for j, dist in enumerate(TRAIN_DISTS)
]
SYNTHETIC_TRAIN: list[Task] = [
    *_SYNTHETIC_TRAIN_64,  # and two at a stricter budget: tree height 32, not 64
    _budget(_SYNTHETIC_TRAIN_64, 896, 896, "gauss", 32),
    _budget(_SYNTHETIC_TRAIN_64, 1536, 1536, "student-t", 32),
]
REAL_TRAIN = real_tasks("qwen2.5-0.5b-l12", seed=500)
TRAIN: list[Task] = SYNTHETIC_TRAIN + REAL_TRAIN

_SYNTHETIC_HELD_OUT_64: list[Task] = [
    Task(1024, 1000, "outliers", seed=900),
    Task(2560, 2304, "mixed-scale", seed=901),
    Task(3584, 3000, "outliers", seed=902),
    Task(5120, 5120, "student-t", seed=903),
    Task(18944, 3584, "outliers", seed=904),
]
SYNTHETIC_HELD_OUT: list[Task] = [
    *_SYNTHETIC_HELD_OUT_64,  # and two at a budget training never uses: tree height 128
    _budget(_SYNTHETIC_HELD_OUT_64, 1024, 1000, "outliers", 128),
    _budget(_SYNTHETIC_HELD_OUT_64, 2560, 2304, "mixed-scale", 128),
]
REAL_HELD_OUT = real_tasks("qwen2.5-0.5b-l20", seed=600)
HELD_OUT: list[Task] = SYNTHETIC_HELD_OUT + REAL_HELD_OUT

TRAIN_LOCKED: dict[str, object] = {"skip_tail": False}
"""Config fields fixed in synthetic training episodes."""


def locks(task: Task) -> dict[str, object]:
    """Config fields fixed in an episode on ``task`` (pass as ``Episode(locked=...)``)."""
    out: dict[str, object] = {}
    if task in SYNTHETIC_TRAIN:
        out.update(TRAIN_LOCKED)
    if task.dist == "real":
        out["layout"] = "col"
    return out


def episode_locks(task: Task, free: frozenset[str] = frozenset()) -> dict[str, object]:
    """The task's locks without the ones a run lifts (``free``, e.g. ``{"layout"}``)."""
    return {k: v for k, v in locks(task).items() if k not in free}


ALL: list[Task] = TRAIN + HELD_OUT


def shapes(tasks: list[Task]) -> list[tuple[int, int]]:
    """Distinct (m, n) shapes, in order of first appearance."""
    return list(dict.fromkeys((t.m, t.n) for t in tasks))
