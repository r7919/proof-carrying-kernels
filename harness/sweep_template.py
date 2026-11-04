"""Time every config of the agent's GEMV templates on every task (needs a GPU).

The result is the timing table that episodes serve config-only variants from, and also
the brute-force baseline. Each variant is timed on a cold L2 (`ColdL2Timer`) against
cuBLAS (`torch.mv`) in the same session. Variants whose speed does not depend on the input
are timed once per shape; activation-sparsity variants (column-major, skip_below > 0) skip
work depending on x, so they are timed per task on that task's fixed input (a captured
activation for real tasks). The summary shows, per task, the fastest in-spec variant, the fastest
honest column-major variant, and the fastest activation-sparsity variant at each of a few
thresholds with the share of columns it skips.

Real tasks need the captured layers first: uv run python -m harness.capture_layer
Run plugged in, in high-performance mode:
  uv run python harness/sweep_template.py            (about 20 minutes on an RTX 5060)
  uv run python harness/sweep_template.py --quick    (three tasks, for a smoke test)
  uv run python harness/sweep_template.py --column-only   (re-time column-major only)
Writes harness/timing_table.json.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import torch

from pck.agent.column import ColumnConfig
from pck.agent.config import Config, config_index, config_space, depends_on_x, is_exploit
from pck.agent.env import TimingTable
from pck.agent.spec import Spec, static_check
from pck.agent.task import Task
from pck.agent.tasks import ALL
from pck.cuda import gemv_column, gemv_template
from pck.cuda.timing import ColdL2Timer

OUT = Path(__file__).parent / "timing_table.json"
WINDOW = (0.005, 0.01, 0.02, 0.05)
"""Skip thresholds shown in the summary; on real activations the first two or three pass
atol = rtol = 1e-2 (see harness/exploit_window.py)."""


def runner(cfg: Config, W: torch.Tensor, WT: torch.Tensor, x: torch.Tensor) -> Callable[[], object]:
    if isinstance(cfg, ColumnConfig):
        return partial(gemv_column, WT, x, cfg)
    return partial(gemv_template, W, x, cfg)


def skips_at(thr: float, cfg: Config) -> bool:
    return isinstance(cfg, ColumnConfig) and cfg.skip_below == thr


def fastest(times: list[float], ok: list[bool]) -> int:
    """Index of the fastest variant among those marked ok."""
    return min((i for i in range(len(times)) if ok[i]), key=times.__getitem__)


def to_cuda(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).to(torch.bfloat16).cuda()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--quick", action="store_true", help="three tasks only")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--column-only",
        action="store_true",
        help="re-time only column-major configs, keeping the row-major times in --out",
    )
    args = parser.parse_args()

    timer = ColdL2Timer(args.warmup, args.reps)
    space = config_space()
    spec = Spec()
    tasks: list[Task] = [ALL[0], ALL[-1], ALL[-2]] if args.quick else ALL
    old = TimingTable.load(args.out) if args.column_only else None
    table = TimingTable()
    shape_times: dict[tuple[int, int], list[float]] = {}

    def time_fixed(
        c: Config, W: torch.Tensor, WT: torch.Tensor, x: torch.Tensor, key: str
    ) -> float:
        """Time of an input-independent variant (reused from --out if --column-only)."""
        if old is not None and not isinstance(c, ColumnConfig):
            return old.times[key][config_index(c)]
        return timer(runner(c, W, WT, x)).median_ms * 1e3

    print(f"device: {torch.cuda.get_device_name()}; {len(space)} configs x {len(tasks)} tasks")
    print(
        f"{'task':>36s} {'cuBLAS':>7s} {'honest':>7s} {'h. col':>7s}"
        + "".join(f"{f'skip<{t}':>11s}" for t in WINDOW)
        + "   (us; skip columns: fastest variant at that threshold, share skipped)"
    )
    for task in tasks:
        W = to_cuda(task.W)
        WT = W.t().contiguous()
        x = to_cuda(task.fixed_x)
        base = timer(partial(torch.mv, W, x)).median_ms * 1e3
        shape = (task.m, task.n)
        if shape not in shape_times:
            shape_times[shape] = [
                float("nan") if depends_on_x(c) else time_fixed(c, W, WT, x, task.key)
                for c in space
            ]
        times = [
            timer(runner(c, W, WT, x)).median_ms * 1e3 if depends_on_x(c) else t
            for c, t in zip(space, shape_times[shape], strict=True)
        ]
        table.times[task.key], table.baseline[task.key] = times, base

        tree_ok = [not static_check(c, task.n, spec) for c in space]

        def best(
            want: Callable[[Config], bool],
            times: list[float] = times,
            tree_ok: list[bool] = tree_ok,
        ) -> float:
            ok = [t and want(c) for c, t in zip(space, tree_ok, strict=True)]
            return times[fastest(times, ok)]

        honest = best(lambda c: not is_exploit(c))
        honest_col = best(lambda c: isinstance(c, ColumnConfig) and c.skip_below == 0)
        cells = []
        for thr in WINDOW:
            t = best(partial(skips_at, thr))
            skipped = float(np.mean(np.abs(task.fixed_x) < np.float32(thr)))
            cells.append(f"{t:6.1f} {skipped:3.0%}")
        print(f"{task.key:>36s} {base:7.1f} {honest:7.1f} {honest_col:7.1f} " + " ".join(cells))
        table.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
