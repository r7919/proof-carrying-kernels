"""The GRPO trainer on a tiny random Qwen3 (real tokenizer, CPU): rollouts, masks, the
loss's direction and an end-to-end run. Skipped without the rl extra or the tokenizer."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from pck.rl.config import RunConfig  # noqa: E402
from pck.rl.envs import EpisodeEnv, Outcome  # noqa: E402
from pck.rl.train import (  # noqa: E402
    Policy,
    Trajectory,
    advantages,
    grpo_loss,
    rollout_multi,
    token_logps,
    train,
)

MODEL = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def tokenizer() -> Any:
    try:
        return transformers.AutoTokenizer.from_pretrained(MODEL)
    except OSError as e:  # offline and not cached
        pytest.skip(f"tokenizer unavailable: {e}")


def tiny(tokenizer: Any) -> Any:
    torch.manual_seed(0)
    config = transformers.Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        tie_word_embeddings=True,
    )
    return transformers.Qwen3ForCausalLM(config)


def policy(tokenizer: Any, **kwargs: Any) -> Policy:
    cfg = RunConfig(
        model=MODEL, dtype="float32", group=2, max_new_tokens=8, steps=2, eval_samples=2
    )
    return Policy(replace(cfg, **kwargs), tiny(tokenizer), tokenizer)


def test_turn_separator_closes_and_reopens_turns(tokenizer: Any) -> None:
    p = policy(tokenizer)
    assert p.sep_before.startswith("<|im_end|>") and "<|im_start|>user" in p.sep_before
    assert p.sep_after.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    ended = p.encode("hi<|im_end|>")
    text = str(p.tok.decode(p.reply_tokens(ended, "REPLY")))
    assert text.startswith("\n<|im_start|>user\nREPLY<|im_end|>")  # no second <|im_end|>


def test_multi_turn_rollouts_mask_only_the_policy(tokenizer: Any) -> None:
    p = policy(tokenizer, env="emulator", turns=3)
    task, rng = p.cfg.train_tasks()[0], np.random.default_rng(0)
    traj = rollout_multi(p, EpisodeEnv("calibrated"), task, rng)
    assert len(traj.ids) == len(traj.mask) and traj.outcome.reward < 0  # random model: no config
    policy_text = p.tok.decode([t for t, m in zip(traj.ids, traj.mask, strict=True) if m])
    assert "Write a fast CUDA kernel" not in policy_text
    assert traj.text.count("---") == 2  # three turns, the last one submitted by the budget


def test_grpo_raises_the_better_reply(tokenizer: Any) -> None:
    p = policy(tokenizer, kl=0.0, lr=1e-2)
    prompt = p.encode(p.render([{"role": "user", "content": "pick"}]))
    good, bad = p.encode('{"vec": 8}<|im_end|>'), p.encode('{"vec": 1}<|im_end|>')

    def traj(reply: list[int], reward: float) -> Trajectory:
        out = Outcome(reward, True, True, reward, False, False, None)
        return Trajectory(prompt + reply, [0] * len(prompt) + [1] * len(reply), out, "")

    group = [traj(good, 1.5), traj(bad, 0.0)]
    params = [q for q in p.model.parameters() if q.requires_grad]
    opt = torch.optim.SGD(params, lr=1.0)

    def gap() -> float:
        with torch.no_grad():
            lg = float(token_logps(p.model, group[0].ids, group[0].mask)[0].sum())
            lb = float(token_logps(p.model, group[1].ids, group[1].mask)[0].sum())
        return lg - lb

    before = gap()
    for _ in range(5):
        opt.zero_grad()
        grpo_loss(p, [group])
        opt.step()
    assert gap() > before + 0.1
    assert advantages([1.0, 1.0]) == [0.0, 0.0]


def test_a_short_run_writes_its_logs(tokenizer: Any, tmp_path: Path) -> None:
    p = policy(tokenizer, out=str(tmp_path), name="t", tasks=("896x896-gauss",))
    summary = train(p.cfg, p, log=False)
    run = tmp_path / "t-calibrated-s0"
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and {"reward", "exploit", "hack", "kl"} <= set(rows[0])
    assert summary["regime"] == "calibrated" and "896x896-gauss" in summary["tasks"]
    assert (run / "adapter").is_dir() and (run / "summary.json").exists()
