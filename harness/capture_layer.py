"""Capture decoder layers of a real model for the agent's real tasks.

Greedy-decodes a set of prompts, records the input of each of a layer's seven linear
layers at every decode step (batch 1, one token), and saves the weights and inputs as BF16
bits to $PCK_CACHE/qwen2.5-0.5b-l<layer>.npz (see pck.agent.real), one file per layer.
The last two prompts are held out for the hidden adversarial suite. The defaults capture
layers 12 (training tasks) and 20 (held-out tasks).

Needs the `models` extra and the model weights (downloaded once, about 1 GB); runs on CPU
in a minute or two:
  uv run python -m harness.capture_layer [--model Qwen/Qwen2.5-0.5B-Instruct --layers 12 20]
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch

from pck.agent.real import LAYERS, cache_dir, to_bits

PROMPTS = [
    "Explain why the sky is blue in two sentences.",
    "Write a Python function that checks whether a number is prime.",
    "Translate to French: The weather is nice today.",
    "List three causes of the French Revolution.",
    "What is the derivative of x^3 + 2x?",
    "Summarize the plot of Romeo and Juliet.",
    "Give me a recipe for a simple tomato soup.",  # held out
    "How does a transistor work?",  # held out
]


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--layers", type=int, nargs="+", default=[12, 20])
    parser.add_argument("--prefix", default="qwen2.5-0.5b-l")
    parser.add_argument("--new-tokens", type=int, default=24)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model: Any = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).eval()
    mods: dict[tuple[int, str], torch.nn.Module] = {}
    for layer in args.layers:
        block = model.model.layers[layer]
        for name in LAYERS:
            parent = (
                block.self_attn if name in ("q_proj", "k_proj", "v_proj", "o_proj") else block.mlp
            )
            mods[(layer, name)] = getattr(parent, name)
    xs: dict[tuple[int, str], list[np.ndarray]] = {key: [] for key in mods}
    ps: dict[tuple[int, str], list[int]] = {key: [] for key in mods}
    prompt = [0]
    hooks = []
    for key, mod in mods.items():

        def hook(
            _m: torch.nn.Module, a: tuple[torch.Tensor, ...], key: tuple[int, str] = key
        ) -> None:
            if a[0].shape[-2] == 1:  # a decode step
                xs[key].append(a[0].reshape(-1).float().numpy())
                ps[key].append(prompt[0])

        hooks.append(mod.register_forward_pre_hook(hook))
    with torch.no_grad():
        for i, text in enumerate(PROMPTS):
            prompt[0] = i
            enc: Any = tok.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            ids = enc["input_ids"]
            model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=args.new_tokens,
                do_sample=False,
            )
    for h in hooks:
        h.remove()

    save: Any = np.savez_compressed  # numpy's stubs reject **kwargs of arrays
    for layer in args.layers:
        out: dict[str, np.ndarray] = {}
        for name in LAYERS:
            weight: Any = mods[(layer, name)].weight
            out[f"W_{name}"] = to_bits(weight.detach().float().numpy())
            out[f"X_{name}"] = to_bits(np.stack(xs[(layer, name)]))
            out[f"P_{name}"] = np.array(ps[(layer, name)], np.int16)
        path = cache_dir() / f"{args.prefix}{layer}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        save(path, **out)
        sizes = ", ".join(f"{name} {out[f'X_{name}'].shape[0]}" for name in LAYERS)
        print(f"wrote {path} (decode inputs: {sizes})")


if __name__ == "__main__":
    main()
