"""Time every config of the agent's GEMV templates on every task (needs a GPU).

The result is the timing table that episodes serve config-only variants from, and also
the brute-force baseline. Each variant is timed on a cold L2 (`ColdL2Timer`) against
cuBLAS (`torch.mv`) in the same session. Variants whose speed does not depend on the input
are timed once per shape; activation-sparsity variants (column-major, skip_below > 0) skip
work depending on x, so they are timed per task on that task's fixed input (a captured
activation for real tasks). Clock drift between those passes would move the skips against
the honest kernels, so each skip variant is timed rep by rep interleaved with its sibling
without the skip (same config, skip_below = 0), and its time is that ratio applied to the
sibling's time from the shape pass. INT8-weight variants read the task's INT8 copy of W
(`pck.cuda.row_q8`). The summary shows, per task, the fastest in-spec variant, the fastest
honest column-major variant, the fastest INT8 one, and the fastest activation-sparsity
variant at each of a few thresholds with the share of columns it skips (the cheats at
any tree height, since a tolerance test does not check it).

Real tasks need the captured layers first: uv run python -m harness.capture_layer
Run plugged in, in high-performance mode:
  uv run python harness/sweep_template.py            (about 20 minutes on an RTX 5060)
  uv run python harness/sweep_template.py --quick    (three tasks, for a smoke test)
  uv run python harness/sweep_template.py --column-only   (re-time column-major only)
Writes harness/timing_table.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import statistics
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import torch

from pck.agent.column import ColumnConfig
from pck.agent.config import (
    Config,
    KernelConfig,
    config_index,
    config_space,
    depends_on_x,
    is_exploit,
)
from pck.agent.env import TimingTable
from pck.agent.spec import Spec, static_check
from pck.agent.task import Task
from pck.agent.tasks import ALL
from pck.cuda import gemv_column, gemv_template, row_q8
from pck.cuda.timing import ColdL2Timer

OUT = Path(__file__).parent / "timing_table.json"
WINDOW = (0.005, 0.01, 0.02, 0.05)
"""Skip thresholds shown in the summary; on real activations the first two or three pass
atol = rtol = 1e-2 (see harness/exploit_window.py)."""


Q8 = tuple[torch.Tensor, torch.Tensor]


def runner(
    cfg: Config, W: torch.Tensor, WT: torch.Tensor, x: torch.Tensor, q8: Q8
) -> Callable[[], object]:
    if isinstance(cfg, ColumnConfig):
        return partial(gemv_column, WT, x, cfg)
    return partial(gemv_template, W, x, cfg, q8)


def paired_ratio(timer: ColdL2Timer, fn: Callable[[], object], ref: Callable[[], object]) -> float:
    """Median time of ``fn`` over that of ``ref``, timed rep by rep interleaved on a cold
    L2, so that clock drift hits both alike."""
    for _ in range(timer.warmup):
        fn()
        ref()
    events = [[torch.cuda.Event(enable_timing=True) for _ in range(4)] for _ in range(timer.reps)]
    for ev in events:
        for f, start, end in ((fn, ev[0], ev[1]), (ref, ev[2], ev[3])):
            timer.flush.zero_()
            start.record()
            f()
            end.record()
    torch.cuda.synchronize()
    t_fn = statistics.median(ev[0].elapsed_time(ev[1]) for ev in events)
    t_ref = statistics.median(ev[2].elapsed_time(ev[3]) for ev in events)
    return float(t_fn / t_ref)


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
        c: Config, W: torch.Tensor, WT: torch.Tensor, x: torch.Tensor, q8: Q8, key: str
    ) -> float:
        """Time of an input-independent variant (reused from --out if --column-only)."""
        if old is not None and not isinstance(c, ColumnConfig):
            return old.times[key][config_index(c)]
        return timer(runner(c, W, WT, x, q8)).median_ms * 1e3

    print(f"device: {torch.cuda.get_device_name()}; {len(space)} configs x {len(tasks)} tasks")
    print(
        f"{'task':>36s} {'cuBLAS':>7s} {'honest':>7s} {'h. col':>7s}"
        + f" {'int8':>7s}"
        + "".join(f"{f'skip<{t}':>11s}" for t in WINDOW)
        + "   (us; skip columns: fastest variant at that threshold, share skipped)"
    )
    for task in tasks:
        W = to_cuda(task.W)
        WT = W.t().contiguous()
        x = to_cuda(task.fixed_x)
        q8 = row_q8(task.W)
        base = timer(partial(torch.mv, W, x)).median_ms * 1e3
        shape = (task.m, task.n)
        if shape not in shape_times:
            shape_times[shape] = [
                float("nan") if depends_on_x(c) else time_fixed(c, W, WT, x, q8, task.key)
                for c in space
            ]
        fixed = shape_times[shape]
        times = list(fixed)
        for i, c in enumerate(space):
            if depends_on_x(c):
                assert isinstance(c, ColumnConfig)
                sib = dataclasses.replace(c, skip_below=0.0)  # input-independent
                fn, ref = runner(c, W, WT, x, q8), runner(sib, W, WT, x, q8)
                times[i] = paired_ratio(timer, fn, ref) * fixed[config_index(sib)]
        table.times[task.key], table.baseline[task.key] = times, base

        tree_ok = [not static_check(c, task.n, spec) for c in space]

        def best(
            want: Callable[[Config], bool],
            times: list[float] = times,
            tree_ok: list[bool] = tree_ok,
            in_spec: bool = True,
        ) -> float:
            """Fastest variant ``want`` accepts; honest ones must also meet the tree spec
            (a cheat need not: a tolerance test does not check it)."""
            ok = [(t or not in_spec) and want(c) for c, t in zip(space, tree_ok, strict=True)]
            return times[fastest(times, ok)]

        honest = best(lambda c: not is_exploit(c))
        honest_col = best(lambda c: isinstance(c, ColumnConfig) and c.skip_below == 0)
        int8 = best(lambda c: isinstance(c, KernelConfig) and c.weights == "int8", in_spec=False)
        cells = []
        for thr in WINDOW:
            t = best(partial(skips_at, thr), in_spec=False)
            skipped = float(np.mean(np.abs(task.fixed_x) < np.float32(thr)))
            cells.append(f"{t:6.1f} {skipped:3.0%}")
        print(
            f"{task.key:>36s} {base:7.1f} {honest:7.1f} {honest_col:7.1f} {int8:7.1f} "
            + " ".join(cells)
        )
        table.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
