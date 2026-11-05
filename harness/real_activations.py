"""Tightness and "why not tests" on real decode-time activations of a small open model.

Loads a causal LM with Hugging Face transformers, generates a few tokens per prompt, and
records the input vector of every linear layer at each decode step (batch 1, one token).
Each recorded (W, x) pair goes through the bit-exact CPU model of the coalesced CUDA GEMV
and the metrics of harness/tightness.py, summarized per layer type:

  * per-row tightness: gamma_h * sum_j |W_ij x_j| / actual error
  * per-block tightness: tau / |rho| at b = 64 and 128
  * "why not tests": the verdicts of two per-element tests and of the proven checker
    (b = 128, one sign draw) on the honest output and on four planted shortcuts, all from
    the bit-exact model of the naive kernel (which has the exploit switches). The tests
    compare against an FP64 reference with a fixed tolerance, as a test suite would:
    PyTorch's FP32 defaults (atol 1e-5, rtol 1.3e-6), and a per-layer atol calibrated on
    the first prompt (10x the largest honest error) and applied to the other prompts.

Needs the `models` extra and a download of the model weights (about 1 GB for the default):
  uv sync --extra gpu --extra models
  uv run python -m harness.real_activations [--model Qwen/Qwen2.5-0.5B-Instruct]
  (runs on CPU too, in a few minutes)
  uv run python -m harness.real_activations --tiny-random   (smoke test, no download)
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from harness.tightness import block_ratios, summary
from pck import (
    BlockedChecker,
    abs_row_sums,
    bf16,
    emulate_gemv,
    emulate_gemv_coalesced,
    gamma,
    reference_gemv,
    tree_height,
)

THREADS = 256
PROMPTS = [
    "Explain why the sky is blue in two sentences.",
    "Write a Python function that checks whether a number is prime.",
    "Translate to French: The weather is nice today.",
    "List three causes of the French Revolution.",
]

BLOCK = 128
TORCH_ATOL, TORCH_RTOL = 1e-5, 1.3e-6  # torch.testing.assert_close defaults for float32
OUTPUTS = ["honest", "drop tail (4)", "BF16 accumulator", "BF16 partials", "BF16 output"]
TESTS = ["torch defaults", "calibrated atol", "proven checker"]

Pair = tuple[NDArray[np.float32], list[tuple[int, NDArray[np.float32]]]]


def capture(
    model: torch.nn.Module, prompts: list[torch.Tensor], new_tokens: int, per_prompt: int
) -> dict[str, Pair]:
    """Greedy-decode each prompt (token ids, shape (1, L)) and record up to ``per_prompt``
    decode-step inputs per prompt of every linear layer except the output head, each
    tagged with its prompt index."""
    seen: dict[str, Pair] = {}
    hooks = []
    prompt = [0]
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and not name.endswith("lm_head"):
            seen[name] = (mod.weight.detach().float().cpu().numpy(), [])

            def hook(
                _mod: torch.nn.Module, args: tuple[torch.Tensor, ...], name: str = name
            ) -> None:
                x = args[0]
                kept = sum(p == prompt[0] for p, _ in seen[name][1])
                if x.shape[-2] == 1 and kept < per_prompt:  # one decode token
                    seen[name][1].append((prompt[0], x.reshape(-1).float().cpu().numpy()))

            hooks.append(mod.register_forward_pre_hook(hook))
    try:
        with torch.no_grad():
            for i, ids in enumerate(prompts):
                prompt[0] = i
                model.generate(  # type: ignore[operator, unused-ignore]
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    max_new_tokens=new_tokens,
                    do_sample=False,
                )
    finally:
        for h in hooks:
            h.remove()
    return seen


def load(model_name: str, device: str) -> tuple[torch.nn.Module, list[torch.Tensor]]:
    """Load a pretrained model and tokenize the prompts (downloads the weights once)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    # transformers' decorated constructors confuse mypy; treat the model as untyped.
    model: Any = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    model.to(device).eval()
    prompts = [tok(p, return_tensors="pt")["input_ids"].to(device) for p in PROMPTS]
    return model, prompts


def tiny_random(device: str) -> tuple[torch.nn.Module, list[torch.Tensor]]:
    """A small randomly initialized Qwen2-style model, for smoke tests without downloads."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=384,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    model_cls: Any = Qwen2ForCausalLM
    model = model_cls(cfg).to(device=device, dtype=torch.bfloat16).eval()
    prompts = [torch.randint(0, 512, (1, 6), device=device) for _ in range(2)]
    return model, prompts


def why_not_tests(
    W: NDArray[np.float32], xs: list[tuple[int, NDArray[np.float32]]], rng: np.random.Generator
) -> list[tuple[str, str, bool]]:
    """(test, output, failed) for each input of prompts after the first, for one layer."""
    m, n = W.shape
    h = tree_height(n, THREADS)
    calib = [bf16(x) for p, x in xs if p == 0]
    if not calib:
        return []
    atol = 10 * max(
        float(np.max(np.abs(emulate_gemv(W, x, THREADS).astype(np.float64) - reference_gemv(W, x))))
        for x in calib
    )
    found = []
    for p, x in xs:
        if p == 0:
            continue
        x = bf16(x)
        ref = reference_gemv(W, x)
        honest = emulate_gemv(W, x, THREADS)
        outs = {
            "honest": honest,
            "drop tail (4)": emulate_gemv(W, x, THREADS, drop_tail=4),
            "BF16 accumulator": emulate_gemv(W, x, THREADS, acc_bf16=True),
            "BF16 partials": emulate_gemv(W, x, THREADS, partial_bf16=True),
            "BF16 output": bf16(honest),
        }
        checker = BlockedChecker(W, x, BLOCK if m % BLOCK == 0 else m, h)
        for out, y in outs.items():
            err = np.abs(y.astype(np.float64) - ref)
            found += [
                ("torch defaults", out, bool(np.any(err > TORCH_ATOL + TORCH_RTOL * np.abs(ref)))),
                ("calibrated atol", out, bool(np.any(err > atol))),
                ("proven checker", out, not bool(checker.verdict(y, rng).all())),
            ]
    return found


def layer_type(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tightness on real activations.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument(
        "--per-prompt", type=int, default=2, help="decode inputs kept per layer and prompt"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tiny-random", action="store_true", help="smoke test, no download")
    args = parser.parse_args()

    model, prompts = tiny_random(args.device) if args.tiny_random else load(args.model, args.device)
    seen = capture(model, prompts, args.new_tokens, args.per_prompt)
    rng = np.random.default_rng(0)
    rows: dict[str, list[NDArray[np.float64]]] = defaultdict(list)
    blocks: dict[tuple[str, int], list[NDArray[np.float64]]] = defaultdict(list)
    flagged: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    proven_alarms = 0
    for name, (W, xs) in seen.items():
        if not xs:
            continue
        kind = layer_type(name)
        m, n = W.shape
        h = tree_height(n, THREADS, vec=8)
        g = gamma(h)
        for _, x in xs:
            x = bf16(x)
            y = emulate_gemv_coalesced(W, x, THREADS)
            err = np.abs(y.astype(np.float64) - reference_gemv(W, x))
            s_row = abs_row_sums(W, x)
            proven_alarms += int((err > g * s_row).sum())
            nz = err > 0
            rows[kind].append(g * s_row[nz] / err[nz])
            for b in (64, 128):
                if m % b == 0:
                    blocks[(kind, b)].append(block_ratios(W, x, y, b, h, rng))
        for test, out, flag in why_not_tests(W, xs, rng):
            flagged[(kind, test, out)].append(flag)

    kinds = sorted(rows)
    label = "tiny random Qwen2 (smoke test)" if args.tiny_random else args.model
    print(f"model: {label}; decode-step inputs, batch 1; coalesced kernel ({THREADS} threads)")
    print("\nPer-row tightness, proven bound / actual error")
    for kind in kinds:
        print(f"  {kind:>10s}: {summary(np.concatenate(rows[kind]))}")
    for b in (64, 128):
        print(f"\nPer-block tightness, tau / |rho|, b = {b}")
        for kind in kinds:
            if blocks[(kind, b)]:
                print(f"  {kind:>10s}: {summary(np.concatenate(blocks[(kind, b)]))}")
    print(
        "\nWhy not tests: share of decode inputs where a test fails (naive kernel, inputs of"
        "\nprompts 2-4; the calibrated atol is 10x the largest honest error on prompt 1)"
    )
    for test in TESTS:
        print(f"\n  {test:<18s}" + "".join(f"{k:>10s}" for k in kinds))
        for out in OUTPUTS:
            cells = "".join(f"{np.mean(flagged[(k, test, out)]):10.1%}" for k in kinds)
            print(f"  {out:<18s}{cells}")
    honest_proven = sum(sum(flagged[(k, "proven checker", "honest")]) for k in kinds)
    print(f"\nproven checker false alarms: {honest_proven} (must be 0)")
    print(f"\nproven bound violations: {proven_alarms} (must be 0)")


if __name__ == "__main__":
    main()
