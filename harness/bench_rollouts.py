"""Rollout throughput: Hugging Face generate() vs vLLM, at a GRPO step's shape (needs a GPU).

Generates the same groups both ways: ``--prompts`` distinct task prompts with ``--group``
samples each, ``--max-new-tokens`` tokens at most, temperature 1. HF runs first in this
process; then a vLLM server is started (``--vllm-command``, its own environment; see
pck.rl.vllm_backend) with the untrained LoRA adapter loaded, as in training. Reports
seconds per step's rollouts and generated tokens per second, median of ``--reps``.

Run: uv run python -m harness.bench_rollouts [--model Qwen/Qwen3-0.6B]
"""

from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import torch

from pck.agent.tasks import ALL, locks
from pck.rl.config import RunConfig
from pck.rl.prompt import single_turn
from pck.rl.train import Policy
from pck.rl.vllm_backend import Generation

TASKS = ("896x896-qwen2.5-0.5b-l12/o_proj", "896x4864-qwen2.5-0.5b-l12/down_proj")


def timed(fn: Callable[[], list[Generation]], reps: int) -> tuple[float, float]:
    """(median seconds, median generated tokens per second) of ``fn()``."""
    secs, rates = [], []
    for _ in range(reps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        gens = fn()
        torch.cuda.synchronize()
        dt = time.perf_counter() - start
        secs.append(dt)
        rates.append(sum(len(g.ids) for g in gens) / dt)
    return statistics.median(secs), statistics.median(rates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--group", type=int, default=8)
    parser.add_argument("--prompts", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--vllm-command", default="vllm")
    parser.add_argument("--vllm-gpu-memory", type=float, default=0.35)
    args = parser.parse_args()

    cfg = RunConfig(model=args.model, max_new_tokens=args.max_new_tokens, temperature=1.0)
    policy = Policy(cfg)
    tasks = [t for t in ALL if t.key in TASKS]
    prompts = [
        policy.encode(policy.render([{"role": "user", "content": single_turn(t, locks(t))}]))
        for t in tasks[: args.prompts]
    ]
    batch = [p for p in prompts for _ in range(args.group)]
    print(
        f"{args.model}: {len(prompts)} prompts x {args.group} samples, "
        f"<= {args.max_new_tokens} new tokens, median of {args.reps}"
    )
    policy.generate(batch)  # warm-up
    hf = timed(lambda: policy.generate(batch), args.reps)
    print(f"  {'HF generate()':>16s}: {hf[0]:6.2f} s per step   {hf[1]:8.0f} tokens/s")

    policy.cfg = replace(
        cfg,
        generator="vllm",
        vllm_command=args.vllm_command,
        vllm_gpu_memory=args.vllm_gpu_memory,
    )
    with tempfile.TemporaryDirectory() as live:
        policy.start_vllm(Path(live))
        try:
            policy.generate(batch)  # warm-up
            vl = timed(lambda: policy.generate(batch), args.reps)
        finally:
            assert policy.vllm is not None
            policy.vllm.close()
    print(f"  {'vLLM (LoRA)':>16s}: {vl[0]:6.2f} s per step   {vl[1]:8.0f} tokens/s")
    print(f"  speedup: {hf[0] / vl[0]:.1f}x per step")


if __name__ == "__main__":
    main()
