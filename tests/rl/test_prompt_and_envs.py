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


def test_presets_load_and_overrides_are_typed() -> None:
    smoke = load("smoke", ["steps=3", "regime=proof-gated", "time_limit_min=none"])
    assert smoke.steps == 3 and smoke.regime == "proof-gated" and smoke.time_limit_min is None
    pilot = load("pilot")
    assert pilot.turns > 1 and pilot.env == "emulator" and pilot.test_tasks()
    with pytest.raises(ValueError, match="unknown config key"):
        load("smoke", ["stpes=3"])
    with pytest.raises(ValueError, match="single-turn"):
        RunConfig(env="table", turns=3)
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
