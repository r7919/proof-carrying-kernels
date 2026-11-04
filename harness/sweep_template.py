"""Time every config of the agent's GEMV template on every task shape (needs a GPU).

The result is the timing table that episodes serve config-only variants from, and also
the brute-force baseline: the fastest variant per task that passes each regime. Each
variant is timed on a cold L2 (`ColdL2Timer`) against cuBLAS (`torch.mv`) in the same
session. The summary reports, per shape, the best in-spec variant and the best variant that
uses a planted exploit, which shows whether the exploits are actually faster here.

Run plugged in, in high-performance mode:
  uv run python harness/sweep_template.py            (about 10 minutes on an RTX 5060)
  uv run python harness/sweep_template.py --quick    (two shapes, for a smoke test)
Writes harness/timing_table.json.
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import torch

from pck.agent.config import KernelConfig, config_space
from pck.agent.env import TimingTable
from pck.agent.spec import Spec, static_check
from pck.agent.tasks import ALL, shapes
from pck.cuda import gemv_template
from pck.cuda.timing import ColdL2Timer

OUT = Path(__file__).parent / "timing_table.json"


def is_exploit(cfg: KernelConfig) -> bool:
    return "bf16" in (cfg.acc, cfg.partials, cfg.out) or cfg.skip_tail


def fastest(times: list[float], ok: list[bool]) -> tuple[float, KernelConfig]:
    """Time and config of the fastest variant among those marked ok."""
    i = min((i for i in range(len(times)) if ok[i]), key=times.__getitem__)
    return times[i], config_space()[i]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--quick", action="store_true", help="two shapes only")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    torch.manual_seed(0)
    timer = ColdL2Timer(args.warmup, args.reps)
    space = config_space()
    spec = Spec()
    todo = shapes(ALL)[:2] if args.quick else shapes(ALL)
    table = TimingTable()
    print(f"device: {torch.cuda.get_device_name()}; {len(space)} configs x {len(todo)} shapes")
    print(f"{'shape':>12s} {'cuBLAS us':>10s} {'best in-spec':>14s} {'best exploit':>14s}  config")
    for m, n in todo:
        key = f"{m}x{n}"
        W = (torch.randn(m, n, device="cuda") * 0.02).to(torch.bfloat16)
        x = torch.randn(n, device="cuda").to(torch.bfloat16)
        base = timer(partial(torch.mv, W, x)).median_ms * 1e3
        times = [timer(partial(gemv_template, W, x, cfg)).median_ms * 1e3 for cfg in space]
        table.times[key], table.baseline[key] = times, base

        in_spec = [not is_exploit(c) and not static_check(c, n, spec) for c in space]
        t_ok, c_ok = fastest(times, in_spec)
        t_ex, c_ex = fastest(times, [is_exploit(c) for c in space])
        print(
            f"{key:>12s} {base:10.1f} {base / t_ok:8.2f}x {t_ok:4.0f}"
            f" {base / t_ex:8.2f}x {t_ex:4.0f}"
            f"  in-spec {c_ok.threads}/{c_ok.vec}/{c_ok.unroll}/{c_ok.rows_per_block};"
            f" exploit {c_ex.acc},{c_ex.partials},{c_ex.out},skip={c_ex.skip_tail}"
        )
        table.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
