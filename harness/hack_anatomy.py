"""How wrong are the hacks the proof-gated checker lets through? (CPU)

For every task, take the proof-gated brute-force submission (the fastest variant whose
sampled inputs all passed the checker) and, if it violates the spec on the hidden suite,
report its worst error on two input sets: held-out inputs from the task's own distribution
(`env.task_suite`) and the rest of the hidden suite. Errors are given as a multiple of the
spec's per-row bound gamma_64 sum|W x| and as a fraction of sum|W x|, with the share of
those inputs the checker accepts.

Run: uv run python -m harness.hack_anatomy   (needs the verdict and timing tables)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pck.agent.baselines import Simulator, VerdictTable, brute_force, hack
from pck.agent.config import Config, emulate
from pck.agent.env import TimingTable, hidden_suite, task_suite
from pck.agent.spec import Spec
from pck.agent.tasks import HELD_OUT, TRAIN, locks
from pck.checker import BlockedChecker
from pck.fp import gamma
from pck.gemv import abs_row_sums, reference_gemv

HERE = Path(__file__).parent


def worst(
    cfg: Config, W: NDArray[np.float32], xs: list[NDArray[np.float32]], spec: Spec
) -> tuple[float, float, float]:
    """Max error over the inputs (x spec bound, x sum|Wx|) and the checker's accept rate."""
    rng = np.random.default_rng(0)
    over_spec, over_sum, accepted = 0.0, 0.0, 0
    for x in xs:
        y = emulate(cfg, W, x)
        err = np.abs(y.astype(np.float64) - reference_gemv(W, x))
        a = np.maximum(abs_row_sums(W, x), np.finfo(np.float64).tiny)
        over_spec = max(over_spec, float(np.max(err / (gamma(spec.h_spec) * a))))
        over_sum = max(over_sum, float(np.max(err / a)))
        checker = BlockedChecker(W, x, spec.block, spec.h_spec)
        accepted += bool(checker.verdict(y, rng, checks=spec.checks).all())
    return over_spec, over_sum, accepted / len(xs)


def main() -> None:
    verdicts = VerdictTable.load(HERE / "verdict_table.json")
    timing = TimingTable.load(HERE / "timing_table.json")
    print("proof-gated brute-force submissions that violate the spec on the hidden suite")
    print(f"{'task':>36s} {'variant':>28s}   own distribution       rest of hidden suite")
    print(f"{'':>36s} {'':>28s}   xspec  /sum|Wx|  acc    xspec  /sum|Wx|  acc")
    for task in TRAIN + HELD_OUT:
        spec = task.spec
        rng = np.random.default_rng(0)
        sim = Simulator(task, "proof-gated", verdicts, timing, locks(task), rng)
        out = brute_force(sim)
        if out.config is None or not hack(task, out.config, spec):
            continue
        own = task_suite(task)
        hidden = hidden_suite(task)
        rest = hidden[: len(hidden) - len(task.held_out_x())]  # without the held-out captures
        cfg, W = out.config, task.W
        name = ",".join(f"{k}={v}" for k, v in cfg.as_dict().items() if k != "layout")
        cells = [worst(cfg, W, xs, spec) for xs in (own, rest)]
        row = "   ".join(f"{s:6.3g}  {f:8.1e} {a:4.0%}" for s, f, a in cells)
        print(f"{task.key:>36s} {name[-28:]:>28s}   {row}")


if __name__ == "__main__":
    main()
