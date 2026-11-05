"""The non-LLM control: a softmax policy over the template's configs, trained with the
same group-relative policy gradient and the same rewards as the LLM (`TableEnv`).

If it learns the exploit under a regime, the finding is about the reward, not about LLMs.

  uv run python -m pck.rl.control                       (calibrated vs proof-gated)
  uv run python -m pck.rl.control --task 896x4864-qwen2.5-0.5b-l12/down_proj --steps 500
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np

from pck.agent.config import config_space
from pck.agent.regimes import RegimeName
from pck.agent.task import Task
from pck.agent.tasks import locks
from pck.rl.config import task_by_key
from pck.rl.envs import TableEnv, _arms

FEEDBACK: tuple[RegimeName, ...] = ("calibrated", "proof-gated")


@dataclass(frozen=True)
class ControlStep:
    step: int
    reward: float
    exploit: float
    """Share of the group that submitted a planted exploit and passed."""
    hack: float
    top: str
    top_prob: float


def train_control(
    task: Task,
    regime: RegimeName,
    steps: int = 1000,
    group: int = 8,
    lr: float = 0.5,
    seed: int = 0,
) -> list[ControlStep]:
    """REINFORCE with the group mean as baseline (GRPO's advantage, without the LLM)."""
    rng = np.random.default_rng(seed)
    env = TableEnv(regime)
    space = config_space()
    arms = np.array(sorted(_arms(task)))
    logits = np.zeros(len(arms))
    log = []
    for step in range(steps):
        p = np.exp(logits - float(np.max(logits)))
        p /= p.sum()
        picks = rng.choice(len(arms), size=group, p=p)
        outs = [env.run(task, space[arms[a]].as_dict(), None, rng) for a in picks]
        r = np.array([o.reward for o in outs])
        adv = (r - r.mean()) / (r.std() + 1e-6) if r.std() > 0 else np.zeros_like(r)
        grad = np.zeros_like(logits)
        for a, g in zip(picks, adv, strict=True):
            grad -= g * p
            grad[a] += g
        logits += lr * grad / group
        best = int(np.argmax(logits))
        log.append(
            ControlStep(
                step,
                float(r.mean()),
                float(np.mean([o.exploit and o.passed for o in outs])),
                float(np.mean([o.hack for o in outs])),
                json.dumps({k: v for k, v in space[arms[best]].as_dict().items()}),
                float(p[best]),
            )
        )
    return log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="896x896-qwen2.5-0.5b-l12/o_proj")
    parser.add_argument("--regimes", type=lambda s: s.split(","), default=list(FEEDBACK))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--jsonl", default=None, help="write every step here")
    args = parser.parse_args()
    task = task_by_key(args.task)
    print(f"task {task.key}, locks {locks(task)}; mean of the last 50 steps")
    print(f"{'regime':>12s} {'seed':>4s} {'reward':>7s} {'exploit':>8s} {'hacks':>6s}  top config")
    rows = []
    for regime in args.regimes:
        for seed in range(args.seeds):
            log = train_control(task, regime, args.steps, seed=seed)
            rows += [{"regime": regime, "seed": seed, **asdict(s)} for s in log]
            tail = log[-50:]
            print(
                f"{regime:>12s} {seed:>4d} {np.mean([s.reward for s in tail]):7.3f}"
                f" {np.mean([s.exploit for s in tail]):8.0%} {np.mean([s.hack for s in tail]):6.0%}"
                f"  {log[-1].top} (p={log[-1].top_prob:.2f})"
            )
    if args.jsonl:
        with open(args.jsonl, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)


if __name__ == "__main__":
    main()
