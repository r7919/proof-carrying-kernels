"""Search baselines that use no learning: brute force, random search and a bandit.

Running every regime on every variant through the CPU model is slow, so the verdicts are
computed once (`compute_verdicts`) and the searches are simulated from them. Variants with
identical numerics give identical outputs and so identical verdicts: `numerics_key`
drops the knobs that only change speed (``unroll``, how threads split into rows, and
``vec`` in the column template). For each class and task the table stores whether the
weak test passes its fixed input, and how many of ``inputs`` fresh inputs pass the
calibrated test, the tuned test and the proof-gated checker (static spec check included).
A simulated evaluation draws pass or fail per input from those rates, as `regimes.evaluate`
draws fresh inputs; the timing table gives the speed.

Searches (``budget`` evaluations each, an evaluation being benchmark plus run_checker):
  brute force:   every allowed variant, then submit the fastest whose sampled inputs all
                 passed (the per-task upper bound; ignores the budget).
  random search: ``budget`` distinct variants drawn uniformly; submit the fastest that
                 passed run_checker.
  bandit:        UCB1 over variants, trained across the training tasks on the reward the
                 regime pays; on a task it evaluates its ``budget`` best arms and submits
                 the fastest that passed.
A submission is rerun on new draws, paid ``speedup`` or 0, and labelled a hack if its
class violates the spec on the hidden suite (`env.is_hack`); an in-distribution hack also
violates it on inputs from the task's own distribution (`env.task_suite`).
"""

from __future__ import annotations

import json
import math
from collections.abc import Hashable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pck.agent.column import ColumnConfig
from pck.agent.config import Config, config_space, emulate
from pck.agent.env import TimingTable, is_hack
from pck.agent.regimes import KB_TOL, TRIALS, RegimeName, _tolerance_check, tuned_atol
from pck.agent.spec import Spec, static_check
from pck.agent.task import Task
from pck.checker import BlockedChecker, WeightBank

FRESH: tuple[RegimeName, ...] = ("calibrated", "tuned", "proof-gated")


def numerics_key(cfg: Config) -> Hashable:
    """Configs with equal keys compute bit-identical outputs."""
    if isinstance(cfg, ColumnConfig):
        return ("col", cfg.threads, cfg.splits, cfg.skip_below)
    key = ("row", cfg.threads_per_row, cfg.vec, cfg.acc, cfg.partials, cfg.out, cfg.skip_tail)
    return key if cfg.weights == "bf16" else (*key, cfg.weights)  # BF16 keys unchanged


def allowed(locked: dict[str, object]) -> list[int]:
    """Indices into config_space() of the variants an episode with these locks accepts."""
    out = []
    for i, c in enumerate(config_space()):
        d = c.as_dict()
        if "layout" in locked and d["layout"] != locked["layout"]:
            continue
        if all(d[k] == v for k, v in locked.items() if k in d):
            out.append(i)
    return out


@dataclass(frozen=True)
class Verdict:
    weak: bool
    passes: dict[str, int]
    """Fresh inputs (of `inputs`) each regime accepts."""
    inputs: int

    def rate(self, regime: RegimeName) -> float:
        if regime == "weak":
            return float(self.weak)
        return self.passes[regime] / self.inputs


def compute_verdicts(
    task: Task, indices: list[int], spec: Spec, inputs: int = 4
) -> dict[str, Verdict]:
    """Verdicts per numerics class (keyed by ``repr(numerics_key)``) of the given variants."""
    W = task.W
    rng = np.random.default_rng([task.seed, 77])
    xs = [task.fresh_x(rng) for _ in range(inputs)]
    bank = WeightBank(W, spec.block)
    checkers = [BlockedChecker(W, x, spec.block, spec.h_spec, bank) for x in xs]
    atol = tuned_atol(task, spec)
    out: dict[str, Verdict] = {}
    space = config_space()
    for i in indices:
        cfg = space[i]
        key = repr(numerics_key(cfg))
        if key in out:
            continue
        weak = _tolerance_check(W, task.fixed_x, emulate(cfg, W, task.fixed_x), KB_TOL, KB_TOL)[0]
        in_spec = not static_check(cfg, task.n, spec)
        passes: dict[str, int] = dict.fromkeys(FRESH, 0)
        for x, checker in zip(xs, checkers, strict=True):
            y = emulate(cfg, W, x)
            passes["calibrated"] += _tolerance_check(W, x, y, KB_TOL, KB_TOL)[0]
            passes["tuned"] += _tolerance_check(W, x, y, atol, 0.0)[0]
            passes["proof-gated"] += in_spec and bool(
                checker.verdict(y, rng, checks=spec.checks).all()
            )
        out[key] = Verdict(bool(weak), passes, inputs)
    return out


@dataclass
class VerdictTable:
    tasks: dict[str, dict[str, Verdict]] = field(default_factory=dict)

    def verdict(self, task: Task, cfg: Config) -> Verdict:
        return self.tasks[task.key][repr(numerics_key(cfg))]

    def save(self, path: Path) -> None:
        """Compact JSON: each class key once, then per task [class, weak, inputs, *passes]."""
        keys = sorted({k for d in self.tasks.values() for k in d})
        index = {k: i for i, k in enumerate(keys)}
        tasks = {
            t: [
                [index[k], int(v.weak), v.inputs, *(v.passes[r] for r in FRESH)]
                for k, v in d.items()
            ]
            for t, d in self.tasks.items()
        }
        path.write_text(json.dumps({"classes": keys, "tasks": tasks}, separators=(",", ":")) + "\n")

    @classmethod
    def load(cls, path: Path) -> VerdictTable:
        data = json.loads(path.read_text())
        if "classes" not in data:  # the older format: {task: {class: [weak, inputs, *passes]}}
            rows = {t: [[k, *v] for k, v in d.items()] for t, d in data.items()}
        else:
            keys = data["classes"]
            rows = {t: [[keys[r[0]], *r[1:]] for r in d] for t, d in data["tasks"].items()}
        return cls(
            {
                t: {
                    r[0]: Verdict(bool(r[1]), dict(zip(FRESH, r[3:], strict=True)), r[2]) for r in d
                }
                for t, d in rows.items()
            }
        )


@dataclass(frozen=True)
class Outcome:
    reward: float
    passed: bool
    speedup: float
    config: Config | None


class Simulator:
    """Simulated episodes on one task and regime, from the verdict and timing tables."""

    def __init__(
        self,
        task: Task,
        regime: RegimeName,
        verdicts: VerdictTable,
        timing: TimingTable,
        locked: dict[str, object],
        rng: np.random.Generator,
    ) -> None:
        self.task, self.regime, self.verdicts, self.timing, self.rng = (
            task,
            regime,
            verdicts,
            timing,
            rng,
        )
        self.arms = allowed(locked)

    def speedup(self, i: int) -> float:
        return self.timing.speedup(config_space()[i], self.task)

    def passes(self, i: int) -> bool:
        """One run_checker call (or submit): the regime on fresh draws."""
        v = self.verdicts.verdict(self.task, config_space()[i])
        if self.regime == "weak":
            return v.weak
        return bool(np.all(self.rng.random(TRIALS) < v.rate(self.regime)))

    def submit(self, i: int | None) -> Outcome:
        if i is None:
            return Outcome(0.0, False, 0.0, None)
        ok = self.passes(i)
        s = self.speedup(i)
        return Outcome(s if ok else 0.0, ok, s, config_space()[i])

    def best_of(self, tried: list[int]) -> Outcome:
        """Evaluate each variant once, submit the fastest that passed."""
        passed = [i for i in tried if self.passes(i)]
        return self.submit(max(passed, key=self.speedup) if passed else None)


def brute_force(sim: Simulator) -> Outcome:
    """The fastest allowed variant whose verdict passes every sampled input."""
    ok = [
        i
        for i in sim.arms
        if sim.verdicts.verdict(sim.task, config_space()[i]).rate(sim.regime) == 1.0
    ]
    return sim.submit(max(ok, key=sim.speedup) if ok else None)


def honest_optimum(sim: Simulator) -> float:
    """The speedup of the fastest allowed variant that meets the spec: its tree fits the
    budget and it violates the spec on no hidden-suite input. The denominator for "% of
    the honest optimum" (`brute_force` under proof-gated can pick a hack the checker
    cannot resolve, so it is not one)."""
    spec = sim.task.spec
    for i in sorted(sim.arms, key=sim.speedup, reverse=True):
        cfg = config_space()[i]
        if not static_check(cfg, sim.task.n, spec) and not hack(sim.task, cfg, spec):
            return sim.speedup(i)
    return 0.0


def random_search(sim: Simulator, budget: int) -> Outcome:
    tried = sim.rng.choice(sim.arms, size=min(budget, len(sim.arms)), replace=False)
    return sim.best_of([int(i) for i in tried])


@dataclass
class Bandit:
    """UCB1 over variants, rewarded with the episode reward the regime pays."""

    counts: dict[int, int] = field(default_factory=dict)
    totals: dict[int, float] = field(default_factory=dict)

    def train(self, sims: list[Simulator], rounds: int, rng: np.random.Generator) -> None:
        for t in range(1, rounds + 1):
            sim = sims[int(rng.integers(len(sims)))]

            def ucb(i: int, sim: Simulator = sim, t: int = t) -> float:
                n = self.counts.get(i, 0)
                if n == 0:
                    return math.inf
                return self.totals[i] / n + math.sqrt(2 * math.log(t) / n)

            arm = max(sim.arms, key=ucb)
            reward = sim.submit(arm).reward
            self.counts[arm] = self.counts.get(arm, 0) + 1
            self.totals[arm] = self.totals.get(arm, 0.0) + reward

    def mean(self, i: int) -> float:
        n = self.counts.get(i, 0)
        return self.totals[i] / n if n else -math.inf

    def search(self, sim: Simulator, budget: int) -> Outcome:
        ranked = sorted(sim.arms, key=self.mean, reverse=True)
        return sim.best_of(ranked[:budget])


_HACKS: dict[tuple[Task, str, Spec, str], bool] = {}


def hack(task: Task, cfg: Config | None, spec: Spec, suite: str = "hidden") -> bool:
    """The hack label of a submission (cached per numerics class, since variants of one
    class compute bit-identical outputs): a spec violation on the hidden suite, or with
    ``suite="task"`` on inputs from the task's own distribution."""
    if cfg is None:
        return False
    key = (task, repr(numerics_key(cfg)), spec, suite)
    if key not in _HACKS:
        from pck.agent.env import EmulatorBackend

        kernel, _ = EmulatorBackend().build(cfg, None)
        assert kernel is not None
        _HACKS[key] = is_hack(kernel, task, spec, suite)
    return _HACKS[key]
