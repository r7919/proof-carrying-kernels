"""Report GRPO runs from their directories (config.json, metrics.jsonl, summary.json).

Prints, per run, the learning curve every ``--every`` steps, when the exploit first
appeared and took over, the final sampled replies, and the final evaluation, so any run
can be reported after the fact without its console output.

Run: uv run python -m harness.grpo_report harness/grpo_smoke/*   (any run directories)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sustained(metrics: list[dict[str, Any]]) -> int | None:
    """The first step from which every later step has the exploit in every submission
    (exploit >= 0.99), or None if the run does not end that way."""
    full = None
    for m in reversed(metrics):
        if m["exploit"] < 0.99:
            break
        full = m["step"]
    return full


def exploit_summary(metrics: list[dict[str, Any]]) -> str:
    """When the exploit first appeared, on how many steps every submission had it, and
    the step from which it stayed to the end."""
    first = next((m["step"] for m in metrics if m["exploit"] > 0), None)
    if first is None:
        return "never seen"
    every = sum(m["exploit"] >= 0.99 for m in metrics)
    full = sustained(metrics)
    end = f"kept to the end from step {full}" if full is not None else "not kept to the end"
    return (
        f"first seen at step {first}; in every submission on {every} of {len(metrics)} steps; {end}"
    )


def report(run: Path, every: int) -> dict[str, Any]:
    cfg = json.loads((run / "config.json").read_text())
    summary: dict[str, Any] = json.loads((run / "summary.json").read_text())
    metrics = rows(run / "metrics.jsonl")
    print(f"== {run}: {cfg['model']}, {cfg['regime']}, tasks {', '.join(cfg['tasks'])}")
    print(
        f"   lr {cfg['lr']}, kl {cfg['kl']}, group {cfg['group']} x {cfg['prompts_per_step']},"
        f" max_new_tokens {cfg['max_new_tokens']}, {len(metrics)} steps in"
        f" {metrics[-1]['time']:.0f} s"
    )
    for m in metrics:
        if m["step"] % every == 0 or m["step"] == len(metrics) - 1:
            print(
                f"   step {m['step']:3d} {m['time']:5.0f}s  reward {m['reward']:6.3f}"
                f"  valid {m['valid']:4.0%}  passed {m['passed']:4.0%}"
                f"  exploit {m['exploit']:4.0%}  hacks {m['hack']:4.0%}  kl {m['kl']:.3f}"
            )
    print(f"   exploit: {exploit_summary(metrics)}")
    samples = run / "samples.jsonl"
    if samples.exists():
        last = rows(samples)[-1]
        text = " ".join(last["text"].split())
        print(f"   last sampled reply (step {last['step']}, reward {last['reward']:.3f}): {text}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--every", type=int, default=5)
    args = parser.parse_args()
    summaries = [report(run, args.every) for run in args.runs]
    width = max(len(str(run)) for run in args.runs)
    print("\nfinal evaluation (sampled episodes per task)")
    print(
        f"{'run':<{width}s} {'regime':>12s} {'reward':>7s} {'passed':>7s} {'exploit':>8s}"
        f" {'hacks':>6s}"
    )
    for run, s in zip(args.runs, summaries, strict=True):
        print(
            f"{run!s:<{width}s} {s['regime']:>12s} {s['reward']:7.3f} {s['passed']:7.0%}"
            f" {s['exploit']:8.0%} {s['hack']:6.0%}"
        )


if __name__ == "__main__":
    main()
