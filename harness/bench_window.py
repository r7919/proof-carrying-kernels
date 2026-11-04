"""Re-time the exploit window's key variants on the real tasks with many reps (needs a GPU).

The sweep (`harness/sweep_template.py`) times thousands of variants at 30 reps and keeps
the fastest per category, which favours lucky measurements. This script takes, per real
task, the fastest variant of each category from the timing table and re-times just those
at 200 reps, interleaving the variants rep by rep so that clock drift hits all of them
alike. It adds cuBLAS on the column-major weights (`torch.mv` on the transposed view) as a
reference for what that layout allows. Categories: honest row-major, honest column-major,
and activation sparsity at each threshold of the window.

Run after the sweep: uv run python harness/bench_window.py
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import torch

from pck.agent.column import ColumnConfig
from pck.agent.config import Config, KernelConfig, config_space, is_exploit
from pck.agent.env import TimingTable
from pck.agent.spec import Spec, static_check
from pck.agent.tasks import REAL_HELD_OUT, REAL_TRAIN
from pck.cuda import gemv_column, gemv_template

TABLE = Path(__file__).parent / "timing_table.json"
THRESHOLDS = (0.005, 0.01, 0.02)


def interleaved_medians(
    fns: list[Callable[[], object]], reps: int, flush: torch.Tensor
) -> list[float]:
    """Median time (us) of each function, timing all of them once per rep, cold L2."""
    for fn in fns:
        fn()
    times: list[list[float]] = [[] for _ in fns]
    for _ in range(reps):
        for i, fn in enumerate(fns):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            flush.zero_()
            start.record()
            fn()
            end.record()
            end.synchronize()
            times[i].append(start.elapsed_time(end) * 1e3)
    return [statistics.median(t) for t in times]


def to_cuda(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).to(torch.bfloat16).cuda()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=int, default=200)
    args = parser.parse_args()

    table = TimingTable.load(TABLE)
    space = config_space()
    spec = Spec()
    l2 = int(getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 64 << 20))
    flush = torch.empty(2 * max(l2, 1 << 20), dtype=torch.uint8, device="cuda")
    print(f"device: {torch.cuda.get_device_name()}; {args.reps} interleaved reps, cold L2, us")
    head = "".join(f"{f'skip<{t}':>14s}" for t in THRESHOLDS)
    print(f"{'task':>16s} {'cuBLAS':>7s} {'cuBLAS col':>10s} {'row':>7s} {'col':>7s}{head}")
    for task in REAL_TRAIN + REAL_HELD_OUT:
        times = table.times[task.key]
        ok = [not static_check(c, task.n, spec) for c in space]

        def best(
            want: Callable[[Config], bool], times: list[float] = times, ok: list[bool] = ok
        ) -> Config:
            i = min((i for i, c in enumerate(space) if ok[i] and want(c)), key=times.__getitem__)
            return space[i]

        row = best(lambda c: isinstance(c, KernelConfig) and not is_exploit(c))
        col = best(lambda c: isinstance(c, ColumnConfig) and c.skip_below == 0)
        cheats = [best(partial(skips_at, t)) for t in THRESHOLDS]
        W = to_cuda(task.W)
        WT = W.t().contiguous()
        x = to_cuda(task.fixed_x)
        fns: list[Callable[[], object]] = [partial(torch.mv, W, x), partial(torch.mv, WT.t(), x)]
        fns += [run(c, W, WT, x) for c in [row, col, *cheats]]
        med = interleaved_medians(fns, args.reps, flush)
        skipped = [float(np.mean(np.abs(task.fixed_x) < np.float32(t))) for t in THRESHOLDS]
        cells = "".join(f"{t:7.1f} ({s:3.0%})" for t, s in zip(med[4:], skipped, strict=True))
        name = (task.source or "").split("/", 1)[0][-3:] + "/" + (task.source or "").split("/")[-1]
        print(f"{name:>16s} {med[0]:7.1f} {med[1]:10.1f} {med[2]:7.1f} {med[3]:7.1f}{cells}")


def skips_at(thr: float, cfg: Config) -> bool:
    return isinstance(cfg, ColumnConfig) and cfg.skip_below == thr


def run(cfg: Config, W: torch.Tensor, WT: torch.Tensor, x: torch.Tensor) -> Callable[[], object]:
    if isinstance(cfg, ColumnConfig):
        return partial(gemv_column, WT, x, cfg)
    return partial(gemv_template, W, x, cfg)


if __name__ == "__main__":
    main()
