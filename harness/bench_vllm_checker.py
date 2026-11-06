"""The proven sampling check inside vLLM: its cost per token, and detection (needs a GPU).

Starts a vLLM server three times with the plugin (`pck.vllm_plugin`): check off, check on,
and check on with ``--fault-rate`` of the sampled tokens of each step replaced by a token
no correct sampler returns. Each run decodes the same batch: ``--concurrency`` requests
at a time, top-k 50, top-p 0.9, temperature 0.8 (FlashInfer's fused rejection sampler,
which never builds the selected set), ``--max-tokens`` each, ``--rounds`` times. Reports
generated tokens per second and the checker's totals: tokens checked, rejected, injected,
caught, false alarms, and the closest an accepted token came to the top-p boundary.

The plugin must be installed in vLLM's environment:
  uv tool install vllm --with-editable .
Run: uv run python -m harness.bench_vllm_checker [--model Qwen/Qwen3-0.6B]
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pck.rl.vllm_backend import VLLMGenerator, _post

PROMPTS = [
    "Explain how a GPU executes a matrix-vector product.",
    "Write a short story about a lighthouse keeper.",
    "List ten facts about the Moon.",
    "Describe the rules of chess for a beginner.",
    "What is floating-point rounding error?",
    "Summarize the history of the printing press.",
    "Give a recipe for vegetable soup.",
    "Explain reinforcement learning to a child.",
]


def decode(url: str, model: str, concurrency: int, max_tokens: int, seed: int) -> int:
    """One round: ``concurrency`` requests at once; returns the generated tokens."""

    def one(i: int) -> int:
        body = {
            "model": model,
            "prompt": PROMPTS[i % len(PROMPTS)],
            "max_tokens": max_tokens,
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.9,
            "seed": seed + i,
            "ignore_eos": True,
        }
        out: dict[str, Any] = _post(f"{url}/v1/completions", body, 600)
        return int(out["usage"]["completion_tokens"])

    with ThreadPoolExecutor(concurrency) as pool:
        return sum(pool.map(one, range(concurrency)))


def run(args: argparse.Namespace, mode: str, log: Path) -> dict[str, Any]:
    os.environ.pop("PCK_CHECK_SAMPLING", None)
    os.environ["PCK_FAULT_RATE"] = "0"
    if mode != "off":
        os.environ |= {"PCK_CHECK_SAMPLING": "1", "PCK_CHECK_LOG": str(log)}
        os.environ["PCK_CHECK_EVERY"] = "8"
    if mode == "faults":
        os.environ["PCK_FAULT_RATE"] = str(args.fault_rate)
    gen = VLLMGenerator(
        args.model,
        command=args.vllm_command,
        gpu_memory=args.gpu_memory,
        max_num_seqs=max(32, args.concurrency),
    )
    try:
        decode(gen.url, args.model, args.concurrency, 32, seed=0)  # warm-up
        start, tokens = time.perf_counter(), 0
        for r in range(args.rounds):
            tokens += decode(gen.url, args.model, args.concurrency, args.max_tokens, 1000 * r)
        seconds = time.perf_counter() - start
    finally:
        gen.close()
    totals = json.loads(log.read_text().splitlines()[-1]) if log.exists() else {}
    return {"mode": mode, "tokens_per_s": tokens / seconds, **totals}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--fault-rate", type=float, default=0.01)
    parser.add_argument("--gpu-memory", type=float, default=0.5)
    parser.add_argument("--vllm-command", default="vllm")
    args = parser.parse_args()
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for mode in ("off", "check", "faults"):
            results.append(run(args, mode, Path(tmp) / f"{mode}.jsonl"))
    print(
        f"{args.model}: {args.concurrency} concurrent requests x {args.max_tokens} tokens x "
        f"{args.rounds} rounds; top-k 50, top-p 0.9, temperature 0.8"
    )
    base = results[0]["tokens_per_s"]
    print(
        f"{'mode':>7s} {'tokens/s':>9s} {'cost':>6s} {'checked':>8s} {'rejected':>8s}"
        f" {'injected':>8s} {'caught':>7s} {'false alarms':>12s} {'max r-p':>9s}"
    )
    for r in results:
        cost = 1 - r["tokens_per_s"] / base
        print(
            f"{r['mode']:>7s} {r['tokens_per_s']:9.0f} {cost:6.1%} {r.get('rows', 0):8d}"
            f" {r.get('rejected', 0):8d} {r.get('injected', 0):8d} {r.get('caught', 0):7d}"
            f" {r.get('false_alarms', 0):12d} {r.get('max_excess', float('nan')):9.2e}"
        )


if __name__ == "__main__":
    main()
