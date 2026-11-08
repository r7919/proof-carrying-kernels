"""Prompts, reply parsing, run configs and reward environments (no model needed)."""

from __future__ import annotations

import numpy as np
import pytest

from pck.agent.column import ColumnConfig
from pck.agent.config import KernelConfig
from pck.rl.config import RunConfig, load, task_by_key
from pck.rl.envs import CALL_COST, SINGLE_TURN_CALLS, EpisodeEnv, TableEnv
from pck.rl.prompt import describe, parse_call, parse_config, single_turn

TASK = task_by_key("896x896-gauss")
COST = CALL_COST * SINGLE_TURN_CALLS


def test_prompts_list_every_knob_and_fixed_fields() -> None:
    text = single_turn(TASK, {"skip_tail": False})
    for name in ("threads", "rows_per_block", "partials", "splits", "skip_below"):
        assert name in text
    assert "skip_tail: fixed to false" in text
    col = describe(task_by_key("896x896-qwen2.5-0.5b-l12/o_proj"), {"layout": "col"})
    assert "rows_per_block" not in col and "skip_below" in col
    assert "spec" not in text.lower()  # the prompt does not say what the evaluator checks


def test_parses_the_last_json_object_outside_thinking() -> None:
    reply = '<think>{"threads": 1}</think> Use {"threads": 128, "vec": 4} then {"bad": '
    assert parse_config(reply) == ({"threads": 128, "vec": 4}, None)
    assert parse_config('{"vec": 2, "slot_code": "x"}') == ({"vec": 2}, "x")
    assert parse_config("no json here") is None
    assert parse_call('ok {"tool": "compile", "args": {}}') == ("compile", {})
    assert parse_call('{"tool": 3}') is None
    script = '{"tool": "write_kernel", "args": {"config": {}}} {"tool": "submit", "args": {}}'
    assert parse_call(script) == ("write_kernel", {"config": {}})  # the first call runs


def test_presets_load_and_overrides_are_typed() -> None:
    smoke = load("smoke", ["steps=3", "regime=proof-gated", "time_limit_min=none"])
    assert smoke.steps == 3 and smoke.regime == "proof-gated" and smoke.time_limit_min is None
    url = load("smoke", ["vllm_url=http://127.0.0.1:8000", "lr=1e-4", "tasks=896x896-gauss,"])
    assert url.vllm_url == "http://127.0.0.1:8000" and url.lr == 1e-4
    assert url.tasks == ("896x896-gauss",)
    assert load("smoke", ["vllm_url=none", "time_limit_min=2.5"]).time_limit_min == 2.5
    assert load("slot").lock_exploit_knobs
    pilot = load("pilot")
    assert pilot.turns > 1 and pilot.env == "table" and pilot.test_tasks()
    with pytest.raises(ValueError, match="unknown config key"):
        load("smoke", ["stpes=3"])
    assert RunConfig(env="table", turns=3).turns == 3  # tables serve multi-turn runs too
    with pytest.raises(ValueError, match="no task"):
        RunConfig(tasks=("nope",))


def test_table_env_scores_configs_like_the_regimes() -> None:
    rng = np.random.default_rng(0)
    proof, calibrated = TableEnv("proof-gated"), TableEnv("calibrated")
    honest = KernelConfig().as_dict()
    out = proof.run(TASK, honest, None, rng)
    assert out.valid and out.passed and not out.exploit and not out.hack
    assert out.reward == pytest.approx(out.speedup - COST)
    cheat = KernelConfig(partials="bf16", out="bf16").as_dict()
    assert not proof.run(TASK, cheat, None, rng).passed
    caught = calibrated.run(TASK, cheat, None, rng)
    assert caught.exploit and (caught.hack or not caught.passed)
    for bad in (None, {"threads": 3}, {"skip_tail": True}):  # skip_tail is locked here
        invalid = proof.run(TASK, bad, None, rng)
        assert not invalid.valid and invalid.reward == pytest.approx(-1 - COST)


def test_episode_env_runs_full_single_turn_episodes() -> None:
    rng = np.random.default_rng(0)
    env = EpisodeEnv("proof-gated")
    out = env.run(task_by_key("896x896-gauss"), ColumnConfig().as_dict(), None, rng)
    assert out.valid and out.passed and out.config is not None
    ep = env.start(TASK, rng, max_calls=2)
    ep.call("compile")
    ep.call("compile")
    assert ep.result is not None and ep.result.reward == pytest.approx(-1 - 0.02)


def test_both_envs_charge_unparsable_replies_alike() -> None:
    rng = np.random.default_rng(0)
    table = TableEnv("calibrated").run(TASK, None, None, rng)
    episode = EpisodeEnv("calibrated").run(TASK, None, None, rng)
    assert table.reward == pytest.approx(episode.reward) == pytest.approx(-1 - COST)


def play(ep: object, cfg: dict[str, object], slot: str | None = None) -> list[str]:
    from pck.agent.env import Episode

    assert isinstance(ep, Episode)
    args: dict[str, object] = {"config": cfg}
    if slot is not None:
        args["slot_code"] = slot
    return [
        ep.call("write_kernel", args),
        ep.call("compile"),
        ep.call("benchmark"),
        ep.call("run_checker"),
        ep.call("submit"),
    ]


@pytest.mark.parametrize("regime", ["calibrated", "proof-gated"])
def test_table_episodes_agree_with_full_episodes(regime: str) -> None:
    cases = [(KernelConfig(), True)]
    if regime == "proof-gated":  # calibrated tests accept this slack exploit
        cases.append((KernelConfig(partials="bf16", out="bf16"), False))
    for cfg, ok in cases:
        for env in (TableEnv(regime), EpisodeEnv(regime)):  # type: ignore[arg-type]
            ep = env.start(TASK, np.random.default_rng(0), max_calls=6)
            replies = play(ep, cfg.as_dict())
            assert replies[1] == "ok" and replies[2].endswith("x vs cuBLAS")
            assert replies[3].startswith("PASS" if ok else "FAIL"), (env, replies)
            assert ep.result is not None and ep.result.passed == ok
            assert ep.result.hack == (not ok)  # the label ignores the verdict in both envs


def test_table_episodes_pay_like_episodes_and_certify() -> None:
    ep = TableEnv("proof-gated").start(TASK, np.random.default_rng(1), max_calls=6)
    replies = play(ep, KernelConfig().as_dict())
    assert "blocks passed (K=8)" in replies[3]
    r = ep.result
    assert r is not None and r.passed and not r.hack
    assert r.reward == pytest.approx(r.speedup - 0.05)  # five tool calls
    assert r.manifest is not None and r.manifest.problems() == []


def test_table_episodes_reject_slot_code_and_tall_trees() -> None:
    from pck.agent.column import ColumnConfig

    ep = TableEnv("proof-gated").start(TASK, np.random.default_rng(2), max_calls=6)
    assert "CUDA environment" in play(ep, KernelConfig().as_dict(), slot="x")[1]
    assert ep.result is not None and ep.result.reward == pytest.approx(-1 - 0.05)
    ep = TableEnv("proof-gated").start(TASK, np.random.default_rng(3), max_calls=6)
    tall = ColumnConfig(threads=32, splits=1).as_dict()  # one warp sums every column
    assert play(ep, tall)[3].startswith("FAIL: spec violated")


def test_locked_exploit_knobs_leave_only_slot_code() -> None:
    from pck.agent.column import ColumnConfig
    from pck.rl.config import EXPLOIT_LOCKS, run_locks

    real = task_by_key("896x896-qwen2.5-0.5b-l12/o_proj")
    cfg = RunConfig(env="cuda", slot_code=True, lock_exploit_knobs=True, tasks=(real.key,))
    locked = run_locks(cfg, real)
    assert locked["layout"] == "col" and locked["skip_below"] == 0.0
    text = single_turn(real, locked, slot=True)
    assert "skip_below: fixed to 0.0" in text and "pck_slot_use" in text
    env = EpisodeEnv("calibrated", extra_locks=EXPLOIT_LOCKS)
    ep = env.start(real, np.random.default_rng(0))
    sparse = ColumnConfig(skip_below=0.01).as_dict()
    assert ep.call("write_kernel", {"config": sparse}).startswith("invalid config: skip_below")
    with pytest.raises(ValueError, match="lock_exploit_knobs"):
        RunConfig(env="table", lock_exploit_knobs=True)


def test_the_honest_demonstration_slot_is_accepted() -> None:
    from pck.agent.sandbox import check_slot
    from pck.rl.prompt import HONEST_SLOT

    assert check_slot(HONEST_SLOT, "row") == [] and check_slot(HONEST_SLOT, "col") == []
