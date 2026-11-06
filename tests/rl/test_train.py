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
from pck.rl.envs import EpisodeEnv, Outcome, TableEnv  # noqa: E402
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
    [traj] = rollout_multi(p, EpisodeEnv("calibrated"), task, rng, group=1)
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
    p = policy(
        tokenizer, out=str(tmp_path), name="t", tasks=("896x896-gauss",), warm_start_examples=2
    )
    summary = train(p.cfg, p, log=False)
    run = tmp_path / "t-calibrated-s0"
    assert json.loads((run / "warmstart.jsonl").read_text())["epoch"] == 0
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and {"reward", "exploit", "hack", "kl"} <= set(rows[0])
    assert summary["regime"] == "calibrated" and "896x896-gauss" in summary["tasks"]
    assert (run / "adapter").is_dir() and (run / "summary.json").exists()


def test_demonstrations_are_honest_and_well_formed(tokenizer: Any) -> None:
    from pck.agent.config import is_exploit, make_config
    from pck.agent.spec import static_check
    from pck.rl.prompt import parse_call, parse_config
    from pck.rl.warmstart import demo_multi, demo_single, honest_configs

    p = policy(tokenizer)
    task, rng = p.cfg.train_tasks()[0], np.random.default_rng(0)
    cfgs = honest_configs(task)
    assert cfgs and all(not is_exploit(c) and not static_check(c, task.n, task.spec) for c in cfgs)
    ids, mask = demo_single(p, task, rng)
    reply = p.decode([t for t, m in zip(ids, mask, strict=True) if m])
    parsed = parse_config(reply)
    assert parsed is not None and make_config(parsed[0]) in cfgs
    ids, mask = demo_multi(p, EpisodeEnv("calibrated"), task, rng, max_calls=6)
    assert len(ids) == len(mask)
    turns = p.decode([t for t, m in zip(ids, mask, strict=True) if m]).count('"tool"')
    assert turns == 5 and parse_call('{"tool": "submit", "args": {}}') == ("submit", {})


def test_warm_start_lowers_the_demonstration_loss(tokenizer: Any) -> None:
    from pck.rl.warmstart import warm_start

    p = policy(tokenizer, warm_start_examples=8, warm_start_epochs=3, warm_start_lr=1e-2)
    losses = warm_start(p, p.cfg.train_tasks(), np.random.default_rng(0))
    assert len(losses) == 3 and losses[-1] < losses[0]


def test_training_with_vllm_syncs_the_adapter_every_step(tokenizer: Any, tmp_path: Path) -> None:
    from tests.rl.fake_vllm import FakeVLLM, serve

    reply = tokenizer('{"threads": 256}<|im_end|>', add_special_tokens=False).input_ids
    fake = FakeVLLM(reply)
    with serve(fake) as url:
        p = policy(
            tokenizer,
            out=str(tmp_path),
            name="v",
            tasks=("896x896-gauss",),
            generator="vllm",
            vllm_url=url,
        )
        summary = train(p.cfg, p, log=False)
    metrics = (tmp_path / "v-calibrated-s0/metrics.jsonl").read_text()
    rows = [json.loads(line) for line in metrics.splitlines()]
    assert len(fake.loaded) == 1 + len(rows)  # the initial adapter, then one per update
    assert fake.unloaded == fake.loaded[:-1]
    assert all(r["valid"] == 1.0 for r in rows) and summary["passed"] == 1.0
    assert all(r["kl"] == 0.0 or "mismatch" in r for r in rows)
    assert not (tmp_path / "v-calibrated-s0/live/v1").exists()  # old adapters are deleted


def test_batched_logps_match_one_at_a_time(tokenizer: Any) -> None:
    from pck.rl.train import batch_logps

    p = policy(tokenizer)
    rng = np.random.default_rng(0)
    seqs = []
    for length, spans in [(30, [(20, 30)]), (12, [(8, 12)]), (40, [(10, 15), (30, 40)])]:
        ids = [int(t) for t in rng.integers(10, 1000, size=length)]
        mask = [0] * length
        for lo, hi in spans:  # policy tokens; the third has two turns with a reply between
            mask[lo:hi] = [1] * (hi - lo)
        seqs.append((ids, mask))
    with torch.no_grad():
        batched = batch_logps(p.model, seqs)
        single = [token_logps(p.model, ids, mask)[0] for ids, mask in seqs]
    for b, s in zip(batched, single, strict=True):
        assert b.shape == s.shape
        torch.testing.assert_close(b, s, atol=1e-5, rtol=1e-5)


def test_multi_turn_training_runs_on_the_tables(tokenizer: Any, tmp_path: Path) -> None:
    p = policy(
        tokenizer,
        out=str(tmp_path),
        name="mt",
        tasks=("896x896-gauss",),
        env="table",
        turns=3,
        group=3,
        prompts_per_step=1,
    )
    env = TableEnv("calibrated")
    trajs = rollout_multi(p, env, p.cfg.train_tasks()[0], np.random.default_rng(0))
    assert len(trajs) == 3 and all(t.text.count("---") == 2 for t in trajs)  # 3 turns each
    summary = train(p.cfg, p, log=False)
    assert summary["regime"] == "calibrated"
