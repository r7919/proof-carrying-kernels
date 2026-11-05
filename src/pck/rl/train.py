"""GRPO training of a kernel-writing policy (LoRA on a small chat model).

  uv run python -m pck.rl.train --config smoke                 (one regime, ~5 minutes)
  uv run python -m pck.rl.train --config smoke --compare       (calibrated, then proof-gated)
  uv run python -m pck.rl.train --config pilot regime=proof-gated seed=1

Each step samples ``prompts_per_step`` training tasks and ``group`` trajectories per task,
scores them in the environment (`pck.rl.envs`), and takes one policy-gradient step on the
group-normalized rewards (GRPO with one update per batch, so the importance ratio is 1),
with a k3 KL penalty to the base model (the LoRA adapter switched off). Only the policy's
own tokens are trained; prompts and tool replies are masked out. Multi-turn trajectories
keep the exact tokens the policy generated from, so training sees what sampling saw.

With ``warm_start_examples > 0`` the adapter is first fine-tuned on scripted honest
demonstrations (`pck.rl.warmstart`).

Writes runs/<name>-<regime>-s<seed>/: config.json, warmstart.jsonl (if any), metrics.jsonl
(one line per step), samples.jsonl (one reply per step), summary.json (final evaluation)
and the adapter.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pck.agent.regimes import RegimeName
from pck.agent.task import Task
from pck.agent.tasks import locks
from pck.rl.config import RunConfig, load
from pck.rl.envs import EpisodeEnv, Outcome, TableEnv, outcome
from pck.rl.prompt import multi_turn, parse_call, parse_config, single_turn

FEEDBACK: tuple[RegimeName, ...] = ("calibrated", "proof-gated")


@dataclass
class Trajectory:
    ids: list[int]
    mask: list[int]
    """1 for the policy's tokens."""
    outcome: Outcome
    text: str
    """The policy's replies."""


class Policy:
    """The model, its tokenizer and generation."""

    def __init__(self, cfg: RunConfig, model: Any = None, tokenizer: Any = None) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = tokenizer or AutoTokenizer.from_pretrained(cfg.model)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = getattr(torch, cfg.dtype)
        base: Any = model or AutoModelForCausalLM.from_pretrained(cfg.model, dtype=dtype)
        self.model = self._lora(base.to(self.device))
        self.sep_before, self.sep_after = self._turn_separator()

    def _lora(self, base: Any) -> Any:
        from peft import LoraConfig, get_peft_model

        names = {n.rsplit(".", 1)[-1] for n, _ in base.named_modules()}
        targets = [
            t
            for t in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
            if t in names
        ]
        lora = LoraConfig(
            r=self.cfg.lora_rank,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=0.0,
            target_modules=targets,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora)
        if self.cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        return model

    def render(self, messages: list[dict[str, str]]) -> str:
        text = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.cfg.thinking
        )
        return str(text)

    def decode(self, ids: list[int]) -> str:
        return str(self.tok.decode(ids, skip_special_tokens=True))

    def encode(self, text: str) -> list[int]:
        ids: list[int] = self.tok(text, add_special_tokens=False).input_ids
        return ids

    def _turn_separator(self) -> tuple[str, str]:
        """The template's text between an assistant reply and the next user message, and
        after it up to the next generation prompt."""
        a, z = "\x01ASSISTANT\x01", "\x01USER\x01"
        text = self.render(
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": a},
                {"role": "user", "content": z},
            ]
        )
        rest = text[text.index(a) + len(a) :]
        before, after = rest.split(z, 1)
        return before, after

    def reply_tokens(self, generated: list[int], reply: str) -> list[int]:
        """Tokens that close the policy's turn, add the tool's reply as a user message and
        open the next assistant turn."""
        before = self.sep_before
        eos = self.tok.eos_token or ""
        if generated and generated[-1] == self.tok.eos_token_id and before.startswith(eos):
            before = before[len(eos) :]  # the policy already ended its turn
        return self.encode(before + reply + self.sep_after)

    @torch.no_grad()
    def generate(self, prompts: Sequence[list[int]]) -> list[list[int]]:
        """One sampled continuation per prompt, cut after the first end-of-turn token."""
        width = max(len(p) for p in prompts)
        pad = self.tok.pad_token_id
        ids = torch.tensor([[pad] * (width - len(p)) + p for p in prompts], device=self.device)
        attn = torch.tensor(
            [[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=self.device
        )
        out = self.model.generate(
            input_ids=ids,
            attention_mask=attn,
            do_sample=True,
            temperature=self.cfg.temperature,
            top_p=1.0,
            top_k=0,
            max_new_tokens=self.cfg.max_new_tokens,
            pad_token_id=pad,
        )
        stops = {self.tok.eos_token_id, pad}
        result = []
        for row in out[:, width:].tolist():
            cut = next((i + 1 for i, t in enumerate(row) if t in stops), len(row))
            result.append(row[:cut])
        return result


def rollout_single(
    policy: Policy, env: TableEnv | EpisodeEnv, task: Task, rng: np.random.Generator
) -> list[Trajectory]:
    cfg = policy.cfg
    prompt = policy.encode(
        policy.render([{"role": "user", "content": single_turn(task, locks(task), cfg.slot_code)}])
    )
    trajs = []
    for gen in policy.generate([prompt] * cfg.group):
        text = policy.decode(gen)
        parsed = parse_config(text)
        fields, slot = parsed if parsed is not None else (None, None)
        out = env.run(task, fields, slot if cfg.slot_code else None, rng)
        trajs.append(Trajectory(prompt + gen, [0] * len(prompt) + [1] * len(gen), out, text))
    return trajs


def rollout_multi(
    policy: Policy, env: EpisodeEnv, task: Task, rng: np.random.Generator
) -> Trajectory:
    cfg = policy.cfg
    ep = env.start(task, rng, max_calls=cfg.turns)
    prompt = multi_turn(task, locks(task), cfg.slot_code, cfg.turns)
    ids = policy.encode(policy.render([{"role": "user", "content": prompt}]))
    mask = [0] * len(ids)
    replies = []
    while ep.result is None:
        gen = policy.generate([ids])[0]
        ids, mask = ids + gen, mask + [1] * len(gen)
        text = policy.decode(gen)
        replies.append(text)
        call = parse_call(text)
        if call is None:
            reply = ep.call("unparsable", {})
        else:
            tool, args = call
            if tool == "write_kernel" and not cfg.slot_code:
                args = {k: v for k, v in args.items() if k != "slot_code"}
            reply = ep.call(tool, args)
        if ep.result is None:
            extra = policy.reply_tokens(gen, reply)
            ids, mask = ids + extra, mask + [0] * len(extra)
    return Trajectory(ids, mask, outcome(ep), "\n---\n".join(replies))


def rollouts(
    policy: Policy, env: TableEnv | EpisodeEnv, task: Task, rng: np.random.Generator
) -> list[Trajectory]:
    if policy.cfg.turns == 1:
        return rollout_single(policy, env, task, rng)
    assert isinstance(env, EpisodeEnv)
    return [rollout_multi(policy, env, task, rng) for _ in range(policy.cfg.group)]


def advantages(rewards: Sequence[float]) -> list[float]:
    """Group-normalized rewards (all zero when the group's rewards are equal)."""
    r = np.asarray(rewards, dtype=np.float64)
    if r.std() < 1e-8:
        return [0.0] * len(r)
    return list((r - r.mean()) / (r.std() + 1e-4))


def token_logps(model: Any, ids: list[int], mask: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-probabilities of the policy's tokens under the model, and their positions."""
    device = next(model.parameters()).device
    x = torch.tensor([ids], device=device)
    pos = torch.tensor([i for i in range(1, len(ids)) if mask[i]], device=device)
    logits = model(x).logits[0, pos - 1].float()
    return torch.log_softmax(logits, -1).gather(1, x[0, pos, None]).squeeze(1), pos


def grpo_loss(policy: Policy, groups: list[list[Trajectory]]) -> dict[str, float]:
    """Accumulate the GRPO gradient of a batch of groups; returns loss statistics."""
    model, beta = policy.model, policy.cfg.kl
    batch = [
        (t, a)
        for g in groups
        for t, a in zip(g, advantages([t.outcome.reward for t in g]), strict=True)
    ]
    tokens = sum(sum(t.mask) for t, _ in batch) or 1
    kl_sum, n = 0.0, 0
    for traj, adv in batch:
        if (adv == 0.0 and beta == 0.0) or sum(traj.mask) == 0:
            continue
        logp, _ = token_logps(model, traj.ids, traj.mask)
        with torch.no_grad(), model.disable_adapter():
            ref, _ = token_logps(model, traj.ids, traj.mask)
        delta = ref - logp
        kl = torch.exp(delta) - delta - 1.0
        loss = -(adv * logp - beta * kl).sum() / tokens
        loss.backward()
        kl_sum += float(kl.detach().sum())
        n += len(logp)
    return {"kl": kl_sum / max(n, 1)}


def summarize(trajs: list[Trajectory]) -> dict[str, float]:
    outs = [t.outcome for t in trajs]
    passed = [o for o in outs if o.passed]
    return {
        "reward": float(np.mean([o.reward for o in outs])),
        "valid": float(np.mean([o.valid for o in outs])),
        "passed": float(np.mean([o.passed for o in outs])),
        "exploit": float(np.mean([o.passed and o.exploit for o in outs])),
        "hack": float(np.mean([o.hack for o in outs])),
        "speedup": float(np.mean([o.speedup for o in passed])) if passed else 0.0,
        "tokens": float(np.mean([sum(t.mask) for t in trajs])),
    }


def make_env(cfg: RunConfig) -> TableEnv | EpisodeEnv:
    if cfg.env == "table":
        return TableEnv(cfg.regime)
    if cfg.env == "cuda":
        from pck.agent.sandbox import CudaBackend

        return EpisodeEnv(cfg.regime, CudaBackend())
    return EpisodeEnv(cfg.regime)


def run_dir(cfg: RunConfig) -> Path:
    return Path(cfg.out) / f"{cfg.name}-{cfg.regime}-s{cfg.seed}"


def train(cfg: RunConfig, policy: Policy | None = None, log: bool = True) -> dict[str, Any]:
    """Train one adapter; returns the final evaluation (also written to summary.json)."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    policy = policy or Policy(cfg)
    env = make_env(cfg)
    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    tasks = cfg.train_tasks()
    if cfg.warm_start_examples:
        from pck.rl.warmstart import warm_start

        with open(out / "warmstart.jsonl", "w") as f:

            def note(row: dict[str, float]) -> None:
                f.write(json.dumps(row) + "\n")
                if log:
                    print(f"warm start epoch {row['epoch']:.0f}: loss {row['loss']:.4f}")

            warm_start(policy, tasks, rng, note)
    params = [p for p in policy.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    start = time.time()
    with open(out / "metrics.jsonl", "w") as metrics, open(out / "samples.jsonl", "w") as samples:
        for step in range(cfg.steps):
            if cfg.time_limit_min is not None and time.time() - start > 60 * cfg.time_limit_min:
                break
            picked = [tasks[int(i)] for i in rng.integers(len(tasks), size=cfg.prompts_per_step)]
            policy.model.eval()
            groups = [rollouts(policy, env, task, rng) for task in picked]
            policy.model.train()
            opt.zero_grad()
            stats = grpo_loss(policy, groups)
            grad = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            opt.step()
            flat = [t for g in groups for t in g]
            row = {
                "step": step,
                "time": round(time.time() - start, 1),
                **summarize(flat),
                **stats,
                "grad": grad if math.isfinite(grad) else -1.0,
            }
            metrics.write(json.dumps(row) + "\n")
            metrics.flush()
            first = flat[0]
            samples.write(
                json.dumps({"step": step, "reward": first.outcome.reward, "text": first.text})
                + "\n"
            )
            if log:
                print(
                    f"{cfg.regime:>11s} step {step:3d} {row['time']:6.0f}s"
                    f"  reward {row['reward']:6.3f}  valid {row['valid']:4.0%}"
                    f"  passed {row['passed']:4.0%}  exploit {row['exploit']:4.0%}"
                    f"  hacks {row['hack']:4.0%}  kl {row['kl']:.4f}",
                    flush=True,
                )
    summary = evaluate(policy, env, cfg, rng)
    summary["train_minutes"] = round((time.time() - start) / 60, 2)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    policy.model.save_pretrained(out / "adapter")
    return summary


def evaluate(
    policy: Policy, env: TableEnv | EpisodeEnv, cfg: RunConfig, rng: np.random.Generator
) -> dict[str, Any]:
    """Sample ``eval_samples`` episodes per test task with the trained policy."""
    policy.model.eval()
    per_task = {}
    for task in cfg.test_tasks():
        trajs: list[Trajectory] = []
        while len(trajs) < cfg.eval_samples:
            trajs += rollouts(policy, env, task, rng)
        per_task[task.key] = summarize(trajs[: cfg.eval_samples])
    keys = ("reward", "passed", "exploit", "hack")
    overall = {k: float(np.mean([v[k] for v in per_task.values()])) for k in keys}
    return {"regime": cfg.regime, **overall, "tasks": per_task}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="smoke", help="preset name or TOML path")
    parser.add_argument("--compare", action="store_true", help="train under each regime")
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = parser.parse_args()
    cfg = load(args.config, args.overrides)
    regimes = FEEDBACK if args.compare else (cfg.regime,)
    results = []
    for regime in regimes:
        run = replace(cfg, regime=regime)
        print(f"== {run.name}: {run.model} on {', '.join(run.tasks)}, {regime}", flush=True)
        results.append(train(run))
    print("\nfinal evaluation (sampled episodes per task)")
    print(f"{'regime':>12s} {'reward':>7s} {'passed':>7s} {'exploit':>8s} {'hacks':>6s}")
    for r in results:
        print(
            f"{r['regime']:>12s} {r['reward']:7.3f} {r['passed']:7.0%}"
            f" {r['exploit']:8.0%} {r['hack']:6.0%}"
        )


if __name__ == "__main__":
    main()
