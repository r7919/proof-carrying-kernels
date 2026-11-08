"""Search baselines per regime: what each feedback regime pays for and lets through.

Simulates brute force, random search (budget 10) and a UCB bandit (trained on the
training tasks, budget 10 on each task) under every regime, from harness/verdict_table.json
and harness/timing_table.json (see pck.agent.baselines). Per regime and task set it
reports the mean reward the regime pays and the mean certified speedup (the speedup if the
submission meets the proven spec on the hidden suite, else 0), both as a share of the
honest optimum (the fastest in-spec variant), and the hack rate of the submissions: spec
violations on the hidden suite, and the in-distribution ones (on held-out inputs from the
task's own distribution, which a deployment would actually see).

Run: uv run python -m harness.baselines   (CPU; needs the two tables and captured layers)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pck.agent.baselines import (
    Bandit,
    Simulator,
    VerdictTable,
    brute_force,
    hack,
    honest_optimum,
    random_search,
)
from pck.agent.env import TimingTable
from pck.agent.regimes import REGIMES, RegimeName
from pck.agent.task import Task
from pck.agent.tasks import HELD_OUT, TRAIN, locks

HERE = Path(__file__).parent
BUDGET = 10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5000, help="bandit training rounds")
    args = parser.parse_args()
    verdicts = VerdictTable.load(HERE / "verdict_table.json")
    timing = TimingTable.load(HERE / "timing_table.json")
    rng = np.random.default_rng(0)

    def make(task: Task, regime: RegimeName) -> Simulator:
        return Simulator(task, regime, verdicts, timing, locks(task), rng)

    # The honest optimum per task: the fastest variant that meets the spec (not brute force
    # under proof-gated, which can pick a hack below the checker's resolution).
    optimum = {t.key: honest_optimum(make(t, "proof-gated")) for t in TRAIN + HELD_OUT}
    print(
        f"budget {BUDGET}, {args.seeds} seeds;"
        " reward and certified speedup as % of the honest optimum"
    )
    print(
        f"{'regime':>12s} {'search':>8s} {'tasks':>9s} {'reward':>8s}"
        f" {'certified':>10s} {'hacks':>7s} {'in-dist':>8s}"
    )
    for regime in REGIMES:
        bandit = Bandit()
        bandit.train([make(t, regime) for t in TRAIN], args.rounds, rng)
        for name, tasks in (("train", TRAIN), ("held-out", HELD_OUT)):
            for search in ("brute", "random", "bandit"):
                paid, certified, hacks, in_dist = [], [], [], []
                for task in tasks:
                    for _ in range(1 if search == "brute" else args.seeds):
                        sim = make(task, regime)
                        if search == "brute":
                            out = brute_force(sim)
                        elif search == "random":
                            out = random_search(sim, BUDGET)
                        else:
                            out = bandit.search(sim, BUDGET)
                        is_hack = out.passed and hack(task, out.config, task.spec)
                        paid.append(out.reward / optimum[task.key])
                        certified.append(
                            0.0 if is_hack or not out.passed else out.speedup / optimum[task.key]
                        )
                        hacks.append(is_hack)
                        in_dist.append(is_hack and hack(task, out.config, task.spec, "task"))
                print(
                    f"{regime:>12s} {search:>8s} {name:>9s} {np.mean(paid):8.1%}"
                    f" {np.mean(certified):10.1%} {np.mean(hacks):7.1%} {np.mean(in_dist):8.1%}"
                )


if __name__ == "__main__":
    main()
