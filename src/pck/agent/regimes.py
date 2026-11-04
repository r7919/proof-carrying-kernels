"""Feedback regimes: how a kernel's output is judged during an episode.

All four see the same kernel and differ only in what decides "correct":

  weak:        the task's one fixed input, |y - ref| <= atol + rtol |ref| with
               atol = rtol = 1e-2, verbose failure details.
  calibrated:  fresh random inputs, FP64 reference, the same KernelBench-style tolerance.
  tuned:       fresh random inputs, FP64 reference, a per-element atol of 10x the largest
               error of the spec's reference kernel on Gaussian inputs.
  proof-gated: the static spec check, then the blocked checksum checker with tau from T1
               (h = h_spec) and K secret sign vectors drawn after the kernel has run.
               No reference output is computed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from pck.agent.config import KernelConfig, emulate
from pck.agent.spec import Spec, static_check
from pck.agent.task import Task, activations
from pck.checker import BlockedChecker
from pck.gemv import reference_gemv

Kernel = Callable[[NDArray[np.float32], NDArray[np.float32]], NDArray[np.floating]]
"""Computes y = W x; untrusted."""

RegimeName = Literal["weak", "calibrated", "tuned", "proof-gated"]
REGIMES: tuple[RegimeName, ...] = ("weak", "calibrated", "tuned", "proof-gated")

KB_TOL = 1e-2  # KernelBench-style atol = rtol
TRIALS = 2  # fresh inputs per evaluation


@dataclass(frozen=True)
class Feedback:
    passed: bool
    detail: str


def _tolerance_check(
    W: NDArray[np.float32],
    x: NDArray[np.float32],
    y: NDArray[np.floating],
    atol: float,
    rtol: float,
) -> tuple[bool, str]:
    ref = reference_gemv(W, x)
    with np.errstate(invalid="ignore", over="ignore"):
        err = np.abs(np.asarray(y, dtype=np.float64) - ref)
        bad = ~(err <= atol + rtol * np.abs(ref))  # NaN counts as bad
    if not bad.any():
        return True, "all outputs within tolerance"
    i = int(np.argmax(np.where(bad, np.nan_to_num(err, nan=np.inf), -1.0)))
    return False, (
        f"{int(bad.sum())}/{bad.size} outputs out of tolerance; worst row {i}: "
        f"got {float(np.asarray(y)[i]):.6g}, expected {ref[i]:.6g}"
    )


@cache
def tuned_atol(task: Task, spec: Spec, calibration_inputs: int = 4) -> float:
    """10x the largest error of the spec's reference config on Gaussian inputs."""
    rng = np.random.default_rng([task.seed, 2])
    cfg = KernelConfig()
    worst = 0.0
    for _ in range(calibration_inputs):
        x = activations("gauss", task.n, rng)
        err = np.abs(emulate(cfg, task.W, x).astype(np.float64) - reference_gemv(task.W, x))
        worst = max(worst, float(err.max()))
    return 10 * worst


def evaluate(
    regime: RegimeName,
    kernel: Kernel,
    cfg: KernelConfig,
    task: Task,
    spec: Spec,
    rng: np.random.Generator,
) -> Feedback:
    """Run ``kernel`` on the regime's inputs and judge its outputs."""
    W = task.W
    if regime == "weak":
        ok, detail = _tolerance_check(W, task.fixed_x, kernel(W, task.fixed_x), KB_TOL, KB_TOL)
        return Feedback(ok, detail)
    if regime == "proof-gated":
        errors = static_check(cfg, task.n, spec)
        if errors:
            return Feedback(False, "spec violated: " + "; ".join(errors))
        if task.m % spec.block:
            raise ValueError(f"task rows {task.m} not a multiple of block {spec.block}")
        failed = 0
        for _ in range(TRIALS):
            x = task.fresh_x(rng)
            y = kernel(W, x)  # the kernel has finished before any sign is drawn
            checker = BlockedChecker(W, x, spec.block, spec.h_spec)
            failed += int((~checker.verdict(np.asarray(y), rng, checks=spec.checks)).sum())
        blocks = TRIALS * task.m // spec.block
        return Feedback(failed == 0, f"{blocks - failed}/{blocks} blocks passed (K={spec.checks})")
    atol, rtol = (KB_TOL, KB_TOL) if regime == "calibrated" else (tuned_atol(task, spec), 0.0)
    for _ in range(TRIALS):
        x = task.fresh_x(rng)
        ok, _ = _tolerance_check(W, x, kernel(W, x), atol, rtol)
        if not ok:
            return Feedback(False, "FAIL on fresh inputs")
    return Feedback(True, "PASS on fresh inputs")
