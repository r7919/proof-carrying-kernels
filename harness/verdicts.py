"""Compute the verdict table the search baselines are simulated from (CPU, parallel).

For every task and every numerics class of its allowed variants, run the CPU model on the
weak test's fixed input and on fresh inputs, and record which regimes accept it
(`pck.agent.baselines.compute_verdicts`). Writes harness/verdict_table.json after every
task and skips tasks already in it, so an interrupted run resumes. Real tasks
need the captured layers (uv run python -m harness.capture_layer).

Run: uv run python -m harness.verdicts [--jobs 4]   (several CPU-hours on 4 cores; the
largest shapes take over an hour each, so let it run in the background and resume)
With --add-missing it instead adds, for every task already in the table, the classes of
every variant a free-layout run can submit (`RunConfig.free_layout`) that it lacks, e.g.
after new variants were added to the config space; existing entries are kept.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from pck.agent.baselines import Verdict, VerdictTable, allowed, compute_verdicts, numerics_key
from pck.agent.config import config_space
from pck.agent.tasks import ALL, episode_locks, locks

OUT = Path(__file__).parent / "verdict_table.json"


def one(i: int) -> tuple[str, dict[str, Verdict], float]:
    task = ALL[i]
    start = time.time()
    verdicts = compute_verdicts(task, allowed(locks(task)), task.spec)
    return task.key, verdicts, time.time() - start


FREE = frozenset({"layout"})


def missing(i: int, have: frozenset[str]) -> tuple[str, dict[str, Verdict], float]:
    """Verdicts of the classes task i's free-layout variants have and ``have`` lacks."""
    task, space = ALL[i], config_space()
    start = time.time()
    todo = [
        j for j in allowed(episode_locks(task, FREE)) if repr(numerics_key(space[j])) not in have
    ]
    verdicts = compute_verdicts(task, todo, task.spec) if todo else {}
    return task.key, verdicts, time.time() - start


def add_missing(table: VerdictTable, out: Path, jobs: int) -> None:
    order = sorted(
        (i for i in range(len(ALL)) if ALL[i].key in table.tasks),
        key=lambda i: ALL[i].m * ALL[i].n,
    )
    haves = [frozenset(table.tasks[ALL[i].key]) for i in order]
    with ProcessPoolExecutor(jobs) as pool:
        for key, verdicts, secs in pool.map(missing, order, haves):
            table.tasks[key] |= verdicts
            print(f"{key:>36s}: {len(verdicts)} classes added in {secs:.0f} s", flush=True)
            table.save(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--add-missing", action="store_true", help="add classes the table lacks (see above)"
    )
    args = parser.parse_args()
    table = VerdictTable.load(args.out) if args.out.exists() else VerdictTable()
    if args.add_missing:
        add_missing(table, args.out, args.jobs)
        print(f"wrote {os.path.relpath(args.out)}")
        return
    todo = [i for i in range(len(ALL)) if ALL[i].key not in table.tasks]  # resumable
    order = sorted(todo, key=lambda i: ALL[i].m * ALL[i].n)  # small ones first
    with ProcessPoolExecutor(args.jobs) as pool:
        for key, verdicts, secs in pool.map(one, order):  # in order, so saves are prefixes
            table.tasks[key] = verdicts
            print(f"{key:>36s}: {len(verdicts)} classes in {secs:.0f} s", flush=True)
            table.save(args.out)
    print(f"wrote {os.path.relpath(args.out)}")


if __name__ == "__main__":
    main()
