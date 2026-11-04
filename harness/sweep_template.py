"""Time every config of the agent's GEMV templates on every task (needs a GPU).

The result is the timing table that episodes serve config-only variants from, and also
the brute-force baseline. Each variant is timed on a cold L2 (`ColdL2Timer`) against
cuBLAS (`torch.mv`) in the same session. Variants whose speed does not depend on the input
are timed once per shape; activation-sparsity variants (column-major, skip_below > 0) skip
work depending on x, so they are timed per task on that task's fixed input (a captured
activation for real tasks). The summary shows, per task, the best in-spec variant and the
best activation-sparsity variant with the share of columns it skips.

Real tasks need the captured layers first: uv run python -m harness.capture_layer
Run plugged in, in high-performance mode:
  uv run python harness/sweep_template.py            (about 20 minutes on an RTX 5060)
  uv run python harness/sweep_template.py --quick    (three tasks, for a smoke test)
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
from pck.agent.config import Config, config_space, is_exploit
from pck.agent.env import TimingTable
from pck.agent.spec import Spec, static_check
from pck.agent.task import Task
from pck.agent.tasks import ALL
from pck.cuda import gemv_column, gemv_template
from pck.cuda.timing import ColdL2Timer

OUT = Path(__file__).parent / "timing_table.json"


def runner(cfg: Config, W: torch.Tensor, WT: torch.Tensor, x: torch.Tensor) -> Callable[[], object]:
    if isinstance(cfg, ColumnConfig):
        return partial(gemv_column, WT, x, cfg)
    return partial(gemv_template, W, x, cfg)


def depends_on_x(cfg: Config) -> bool:
    return isinstance(cfg, ColumnConfig) and cfg.skip_below > 0


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
    args = parser.parse_args()

    timer = ColdL2Timer(args.warmup, args.reps)
    space = config_space()
    spec = Spec()
    tasks: list[Task] = [ALL[0], ALL[-1], ALL[-2]] if args.quick else ALL
    table = TimingTable()
    shape_times: dict[tuple[int, int], list[float]] = {}
    print(f"device: {torch.cuda.get_device_name()}; {len(space)} configs x {len(tasks)} tasks")
    print(f"{'task':>32s} {'cuBLAS us':>9s} {'best in-spec':>13s} {'best sparsity':>14s}  skips")
    for task in tasks:
        W = to_cuda(task.W)
        WT = W.t().contiguous()
        x = to_cuda(task.fixed_x)
        base = timer(partial(torch.mv, W, x)).median_ms * 1e3
        shape = (task.m, task.n)
        if shape not in shape_times:
            shape_times[shape] = [
                float("nan") if depends_on_x(c) else timer(runner(c, W, WT, x)).median_ms * 1e3
                for c in space
            ]
        times = [
            timer(runner(c, W, WT, x)).median_ms * 1e3 if depends_on_x(c) else t
            for c, t in zip(space, shape_times[shape], strict=True)
        ]
        table.times[task.key], table.baseline[task.key] = times, base

        tree_ok = [not static_check(c, task.n, spec) for c in space]
        i_ok = fastest(
            times, [t and not is_exploit(c) for c, t in zip(space, tree_ok, strict=True)]
        )
        i_sp = fastest(times, [t and depends_on_x(c) for c, t in zip(space, tree_ok, strict=True)])
        sp = space[i_sp]
        assert isinstance(sp, ColumnConfig)
        skipped = float(np.mean(np.abs(task.fixed_x) < np.float32(sp.skip_below)))
        print(
            f"{task.key:>32s} {base:9.1f} {base / times[i_ok]:7.2f}x {times[i_ok]:4.0f}"
            f" {base / times[i_sp]:8.2f}x {times[i_sp]:4.0f}  {skipped:5.1%} at {sp.skip_below}"
            f"  (in-spec: {space[i_ok].as_dict()['layout']})"
        )
        table.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
