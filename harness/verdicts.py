"""Compute the verdict table the search baselines are simulated from (CPU, parallel).

For every task and every numerics class of its allowed variants, run the CPU model on the
weak test's fixed input and on fresh inputs, and record which regimes accept it
(`pck.agent.baselines.compute_verdicts`). Writes harness/verdict_table.json after every
task and skips tasks already in it, so an interrupted run resumes. Real tasks
need the captured layers (uv run python -m harness.capture_layer).

Run: uv run python -m harness.verdicts [--jobs 4]   (about half an hour on 4 cores)
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from pck.agent.baselines import Verdict, VerdictTable, allowed, compute_verdicts
from pck.agent.spec import Spec
from pck.agent.tasks import ALL, locks

OUT = Path(__file__).parent / "verdict_table.json"


def one(i: int) -> tuple[str, dict[str, Verdict], float]:
    task = ALL[i]
    start = time.time()
    verdicts = compute_verdicts(task, allowed(locks(task)), Spec())
    return task.key, verdicts, time.time() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    table = VerdictTable.load(args.out) if args.out.exists() else VerdictTable()
    todo = [i for i in range(len(ALL)) if ALL[i].key not in table.tasks]  # resumable
    order = sorted(todo, key=lambda i: ALL[i].m * ALL[i].n)  # small ones first
    with ProcessPoolExecutor(args.jobs) as pool:
        for key, verdicts, secs in pool.map(one, order):  # in order, so saves are prefixes
            table.tasks[key] = verdicts
            print(f"{key:>36s}: {len(verdicts)} classes in {secs:.0f} s", flush=True)
            table.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
