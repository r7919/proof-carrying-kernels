"""Warm start: a short supervised fine-tune on correct episodes before GRPO.

The plan's warm start teaches the policy the reply format so that RL starts from kernels
that compile. The demonstrations come from a scripted agent, with no frontier-model API:
  single-turn  the prompt, then a random honest config as compact JSON;
  multi-turn   write_kernel(random honest config), compile, benchmark, run_checker,
               submit, with the environment's real replies in between.
"Honest" means no planted exploit knob and within the task's spec (`static_check`), so
the demonstrations never show a hack; any preference for exploits must come from RL.
The configs are random, not the fastest, so the demonstrations do not teach the optimum
either. Enabled by ``warm_start_examples > 0`` in the run config.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from pck.agent.config import Config, config_space, is_exploit
from pck.agent.spec import static_check
from pck.agent.task import Task
from pck.rl.config import run_locks
from pck.rl.envs import EpisodeEnv, TableEnv, _arms
from pck.rl.prompt import HONEST_SLOT, multi_turn, single_turn

if TYPE_CHECKING:
    from pck.rl.train import Policy


def honest_configs(task: Task, free: frozenset[str] = frozenset()) -> list[Config]:
    """The task's allowed configs that use no exploit knob and meet its spec."""
    space = config_space()
    return [
        space[i]
        for i in sorted(_arms(task, free))
        if not is_exploit(space[i]) and not static_check(space[i], task.n, task.spec)
    ]


def _compact(cfg: Config, locked: dict[str, object]) -> dict[str, Any]:
    return {k: v for k, v in cfg.as_dict().items() if k not in locked or k == "layout"}


Tokens = tuple[list[int], list[int]]
"""Token ids and the policy mask."""


def _slot(policy: Policy, rng: np.random.Generator) -> str | None:
    """An honest slot for half of the demonstrations of a run that offers slot code."""
    return HONEST_SLOT if policy.cfg.slot_code and rng.random() < 0.5 else None


def demo_single(policy: Policy, task: Task, rng: np.random.Generator) -> Tokens:
    locked = run_locks(policy.cfg, task)
    text = single_turn(task, locked, policy.cfg.slot_code)
    prompt = policy.encode(policy.render([{"role": "user", "content": text}]))
    cfgs = honest_configs(task, policy.cfg.freed())
    cfg = cfgs[int(rng.integers(len(cfgs)))]
    fields = _compact(cfg, locked)
    if (slot := _slot(policy, rng)) is not None:
        fields["slot_code"] = slot
    reply = policy.encode(json.dumps(fields) + (policy.tok.eos_token or ""))
    return prompt + reply, [0] * len(prompt) + [1] * len(reply)


def demo_multi(
    policy: Policy,
    env: TableEnv | EpisodeEnv,
    task: Task,
    rng: np.random.Generator,
    max_calls: int,
) -> Tokens:
    locked = run_locks(policy.cfg, task)
    ep = env.start(task, rng, max_calls=max_calls)
    prompt = multi_turn(task, locked, policy.cfg.slot_code, max_calls)
    ids = policy.encode(policy.render([{"role": "user", "content": prompt}]))
    mask = [0] * len(ids)
    cfgs = honest_configs(task, policy.cfg.freed())
    cfg = cfgs[int(rng.integers(len(cfgs)))]
    args: dict[str, Any] = {"config": _compact(cfg, locked)}
    if (slot := _slot(policy, rng)) is not None:
        args["slot_code"] = slot
    write = ("write_kernel", args)
    script: list[tuple[str, dict[str, Any]]] = (
        [write, ("compile", {}), ("benchmark", {}), ("run_checker", {}), ("submit", {})]
        if max_calls >= 5
        else [write, ("compile", {}), ("submit", {})]  # too few calls: build and submit
    )
    for tool, args in script:
        call = json.dumps({"tool": tool, "args": args})
        reply = policy.encode(call + (policy.tok.eos_token or ""))
        ids, mask = ids + reply, mask + [1] * len(reply)
        answer = ep.call(tool, args)
        if ep.result is not None:
            break
        extra = policy.reply_tokens(reply, answer)
        ids, mask = ids + extra, mask + [0] * len(extra)
    return ids, mask


def warm_start(
    policy: Policy,
    tasks: list[Task],
    rng: np.random.Generator,
    log: Callable[[dict[str, float]], None] | None = None,
    env: TableEnv | EpisodeEnv | None = None,
) -> list[float]:
    """Fine-tune the adapter on demonstrations; returns the mean loss per epoch.

    Multi-turn demonstrations are played in ``env``, the run's own environment (a fresh
    one of the run's kind if None)."""
    from pck.rl.train import make_env, token_logps

    cfg = policy.cfg
    if cfg.turns == 1:
        env = None
    elif env is None:
        env = make_env(cfg)
    demos = []
    for _ in range(cfg.warm_start_examples):
        task = tasks[int(rng.integers(len(tasks)))]
        if env is None:
            demos.append(demo_single(policy, task, rng))
        else:
            demos.append(demo_multi(policy, env, task, rng, cfg.turns))
    params = [p for p in policy.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.warm_start_lr, weight_decay=0.0)
    batch = 8
    losses = []
    policy.model.train()
    for epoch in range(cfg.warm_start_epochs):
        order = rng.permutation(len(demos))
        total = 0.0
        for start in range(0, len(order), batch):
            chunk = [demos[int(i)] for i in order[start : start + batch]]
            tokens = sum(sum(m) for _, m in chunk) or 1
            opt.zero_grad()
            for ids, mask in chunk:
                logp, _ = token_logps(policy.model, ids, mask)
                loss = -logp.sum() / tokens
                loss.backward()
                total += float(loss.detach()) * tokens
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        mean = total / max(sum(sum(m) for _, m in demos), 1)
        losses.append(mean)
        if log is not None:
            log({"epoch": epoch, "loss": mean})
    return losses
