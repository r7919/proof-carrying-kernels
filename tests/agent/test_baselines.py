from pathlib import Path

import numpy as np
import pytest

from pck.agent.baselines import (
    Bandit,
    Simulator,
    Verdict,
    VerdictTable,
    allowed,
    brute_force,
    compute_verdicts,
    hack,
    numerics_key,
    random_search,
)
from pck.agent.column import ColumnConfig
from pck.agent.config import Config, KernelConfig, config_index, config_space, emulate
from pck.agent.env import TimingTable
from pck.agent.regimes import RegimeName
from pck.agent.spec import Spec
from pck.agent.task import Task

SPEC = Spec()
TASK = Task(256, 1000)
HONEST = KernelConfig()
CHEAT = KernelConfig(partials="bf16")
SOME = [config_index(c) for c in (HONEST, CHEAT, KernelConfig(unroll=2), KernelConfig(threads=128))]


def timing() -> TimingTable:
    """Synthetic timings: every variant 10 us, the cheat 8 us."""
    times = [10.0] * len(config_space())
    times[config_index(CHEAT)] = 8.0
    return TimingTable({TASK.key: times}, {TASK.key: 12.0})


@pytest.fixture(scope="module")
def verdicts() -> VerdictTable:
    return VerdictTable({TASK.key: compute_verdicts(TASK, SOME, SPEC)})


def sim(regime: RegimeName, verdicts: VerdictTable, seed: int = 0) -> Simulator:
    s = Simulator(TASK, regime, verdicts, timing(), {}, np.random.default_rng(seed))
    s.arms = SOME
    return s


def test_numerics_key_ignores_only_speed_knobs() -> None:
    rng = np.random.default_rng(0)
    W = rng.standard_normal((32, 700)).astype(np.float32)
    x = rng.standard_normal(700).astype(np.float32)
    groups: dict[object, list[Config]] = {}
    for c in config_space()[::7]:
        groups.setdefault(numerics_key(c), []).append(c)
    for members in list(groups.values())[:40]:
        first = emulate(members[0], W, x)
        for c in members[1:]:
            np.testing.assert_array_equal(emulate(c, W, x), first)
    assert numerics_key(ColumnConfig(vec=2)) == numerics_key(ColumnConfig(vec=8))
    assert numerics_key(KernelConfig(threads=256, rows_per_block=2)) == numerics_key(
        KernelConfig(threads=128)
    )


def test_allowed_respects_locks() -> None:
    col = allowed({"layout": "col"})
    assert col and all(isinstance(config_space()[i], ColumnConfig) for i in col)
    no_skip = allowed({"skip_tail": False})
    assert all(not getattr(config_space()[i], "skip_tail", False) for i in no_skip)
    assert len(allowed({})) == len(config_space())


def test_verdicts_match_the_regimes(verdicts: VerdictTable) -> None:
    honest, cheat = verdicts.verdict(TASK, HONEST), verdicts.verdict(TASK, CHEAT)
    assert honest.weak and all(
        honest.rate(r) == 1.0 for r in ("calibrated", "tuned", "proof-gated")
    )
    assert cheat.weak and cheat.rate("calibrated") == 1.0
    assert cheat.rate("proof-gated") == 0.0 and cheat.rate("tuned") == 0.0


def test_brute_force_takes_the_cheat_only_where_the_regime_admits_it(
    verdicts: VerdictTable,
) -> None:
    calibrated = brute_force(sim("calibrated", verdicts))
    proof = brute_force(sim("proof-gated", verdicts))
    assert calibrated.config == CHEAT and calibrated.reward == pytest.approx(1.5)
    assert proof.config != CHEAT and proof.reward == pytest.approx(1.2)
    assert hack(TASK, calibrated.config, SPEC) and not hack(TASK, proof.config, SPEC)
    assert not hack(TASK, None, SPEC)


def test_random_search_and_bandit(verdicts: VerdictTable) -> None:
    out = random_search(sim("proof-gated", verdicts), budget=10)
    assert out.passed and out.reward == pytest.approx(1.2)
    bandit = Bandit()
    bandit.train(
        [sim("calibrated", verdicts, seed=s) for s in range(3)],
        rounds=50,
        rng=np.random.default_rng(0),
    )
    assert max(SOME, key=bandit.mean) == config_index(CHEAT)  # it learns to cheat
    assert bandit.search(sim("calibrated", verdicts), budget=1).config == CHEAT


def test_submitting_nothing_and_failing_pay_zero(verdicts: VerdictTable) -> None:
    s = sim("proof-gated", verdicts)
    assert s.submit(None).reward == 0.0
    assert s.submit(config_index(CHEAT)).reward == 0.0
    s.arms = [config_index(CHEAT)]
    assert brute_force(s).config is None and s.best_of(s.arms).config is None


def test_verdict_table_round_trips(tmp_path: Path, verdicts: VerdictTable) -> None:
    path = tmp_path / "v.json"
    verdicts.save(path)
    assert VerdictTable.load(path) == verdicts
    assert (
        Verdict(True, {"calibrated": 1, "tuned": 0, "proof-gated": 2}, 4).rate("proof-gated") == 0.5
    )
