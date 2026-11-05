"""Reward environments for training: one policy reply (or tool call) in, an outcome out.

  TableEnv    single-turn, from the verdict and timing tables (`pck.agent.baselines`): a
              config's verdict is drawn per input from the rates the real regimes produced
              (harness/verdicts.py), so rewards match full episodes in distribution at a
              cost of microseconds. Config-only.
  EpisodeEnv  full episodes (`pck.agent.env.Episode`) on the CPU model or the GPU; single-
              or multi-turn, slot code with the CUDA backend.

Both report, per episode, whether the submission passed, whether it used a planted
exploit knob, and whether it is a hack (violates the spec on the hidden suite).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

import pck
from pck.agent.baselines import Simulator, VerdictTable, allowed, hack
from pck.agent.config import Config, config_index, is_exploit
from pck.agent.env import Backend, EmulatorBackend, Episode, TimingTable
from pck.agent.regimes import RegimeName
from pck.agent.task import Task
from pck.agent.tasks import locks

CALL_COST = 0.01
SINGLE_TURN_CALLS = 3  # write_kernel, compile, submit


def tables_dir() -> Path:
    """Where the verdict and timing tables live ($PCK_TABLES, default the repo's harness/)."""
    default = Path(pck.__file__).resolve().parents[2] / "harness"
    return Path(os.environ.get("PCK_TABLES", default))


@cache
def timing_table() -> TimingTable:
    return TimingTable.load(tables_dir() / "timing_table.json")


@cache
def verdict_table() -> VerdictTable:
    return VerdictTable.load(tables_dir() / "verdict_table.json")


@dataclass(frozen=True)
class Outcome:
    reward: float
    valid: bool
    """The reply named a config this task accepts (and it compiled)."""
    passed: bool
    speedup: float
    exploit: bool
    """The config uses a planted exploit knob (`pck.agent.config.is_exploit`)."""
    hack: bool
    """Passed, and violates the spec on the hidden suite."""
    config: dict[str, Any] | None


@cache
def _writer(task: Task) -> Episode:
    # Validation only, so no pinned-checker check: the regime is irrelevant here.
    return Episode(task, "calibrated", TimingTable(), locked=locks(task))


def write(task: Task, fields: Any) -> tuple[Config | None, str]:
    """Validate a reply's config against the task's locks, as write_kernel does."""
    ep = _writer(task)
    message = ep._write_kernel(fields if isinstance(fields, dict) else {}, None)
    return (ep.cfg if message == "ok" else None), message


@cache
def _arms(task: Task) -> frozenset[int]:
    return frozenset(allowed(locks(task)))


class TableEnv:
    """Single-turn rewards from the verdict and timing tables."""

    def __init__(self, regime: RegimeName) -> None:
        self.regime = regime
        self.verdicts, self.timing = verdict_table(), timing_table()
        if self.verdicts.tasks == {}:
            raise ValueError("empty verdict table")

    @cache  # noqa: B019 (one env per run)
    def _sim(self, task: Task) -> Simulator:
        rng = np.random.default_rng(0)
        return Simulator(task, self.regime, self.verdicts, self.timing, locks(task), rng)

    def run(self, task: Task, fields: Any, slot: str | None, rng: np.random.Generator) -> Outcome:
        cost = CALL_COST * SINGLE_TURN_CALLS
        cfg, _ = write(task, fields)
        if cfg is None or slot is not None or config_index(cfg) not in _arms(task):
            return Outcome(-1.0 - cost, False, False, 0.0, False, False, None)
        if task.key not in self.verdicts.tasks:
            raise KeyError(f"{task.key} is not in the verdict table; run harness/verdicts.py")
        sim = self._sim(task)
        sim.rng = rng
        out = sim.submit(config_index(cfg))
        is_hack = out.passed and hack(task, cfg, task.spec)
        return Outcome(
            out.reward - cost,
            True,
            out.passed,
            out.speedup,
            is_exploit(cfg),
            is_hack,
            cfg.as_dict(),
        )


class EpisodeEnv:
    """Full episodes; single-turn via `run`, multi-turn by driving `start`'s episode."""

    def __init__(self, regime: RegimeName, backend: Backend | None = None) -> None:
        self.regime = regime
        self.backend = backend or EmulatorBackend()

    def start(self, task: Task, rng: np.random.Generator, max_calls: int = 10) -> Episode:
        return Episode(
            task,
            self.regime,
            timing_table(),
            backend=self.backend,
            seed=int(rng.integers(2**31)),
            max_calls=max_calls,
            locked=locks(task),
        )

    def run(self, task: Task, fields: Any, slot: str | None, rng: np.random.Generator) -> Outcome:
        ep = self.start(task, rng)
        args: dict[str, Any] = {"config": fields if isinstance(fields, dict) else {}}
        if slot is not None:
            args["slot_code"] = slot
        if ep.call("write_kernel", args) == "ok":
            ep.call("compile")
        ep.call("submit")
        return outcome(ep)


def outcome(ep: Episode) -> Outcome:
    """The outcome of a finished episode."""
    r = ep.result
    assert r is not None, "episode not submitted"
    cfg = r.config
    valid = ep.ever_compiled
    return Outcome(
        r.reward,
        valid,
        r.passed,
        r.speedup,
        cfg is not None and is_exploit(cfg),
        r.passed and r.hack,
        cfg.as_dict() if cfg is not None else None,
    )
