from pathlib import Path

import numpy as np
import pytest

from pck.agent import real
from pck.agent.spec import Spec
from pck.agent.task import Task, activations
from pck.agent.tasks import (
    ALL,
    HELD_OUT,
    REAL_HELD_OUT,
    REAL_TRAIN,
    SYNTHETIC_HELD_OUT,
    SYNTHETIC_TRAIN,
    TRAIN,
    TRAIN_LOCKED,
    shapes,
)


def test_split_is_disjoint_and_every_task_fits_the_checker() -> None:
    assert not {t.key for t in TRAIN} & {t.key for t in HELD_OUT}
    assert not set(shapes(SYNTHETIC_TRAIN)) & set(shapes(SYNTHETIC_HELD_OUT))
    assert all(t.m % Spec().block == 0 for t in ALL)
    assert len({t.key for t in ALL}) == len(ALL)
    assert {t.source.split("/")[1] for t in REAL_TRAIN} == set(real.LAYERS)  # type: ignore[union-attr]
    assert all("l20" in (t.source or "") for t in REAL_HELD_OUT)


def test_held_out_has_shapes_where_skip_tail_drops_work() -> None:
    assert TRAIN_LOCKED == {"skip_tail": False}
    assert any(t.n % 4096 for t in HELD_OUT)
    assert {t.dist for t in SYNTHETIC_HELD_OUT} - {t.dist for t in SYNTHETIC_TRAIN}


def fake_capture(tmp: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    """A capture file with one 128 x 64 layer and 10 inputs from 5 prompts."""
    rng = np.random.default_rng(0)
    W = real.from_bits(real.to_bits(rng.standard_normal((128, 64)).astype(np.float32)))
    X = real.from_bits(real.to_bits(rng.standard_normal((10, 64)).astype(np.float32)))
    np.savez(
        tmp / f"{name}.npz",
        W_q_proj=real.to_bits(W),
        X_q_proj=real.to_bits(X),
        P_q_proj=np.repeat(np.arange(5), 2).astype(np.int16),
    )
    return W, X


def test_real_task_loads_weights_and_keeps_held_out_prompts_apart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PCK_CACHE", str(tmp_path))
    W, X = fake_capture(tmp_path, "fake-a")
    task = Task(128, 64, "real", source="fake-a/q_proj")
    np.testing.assert_array_equal(task.W, W)
    rng = np.random.default_rng(1)
    seen = {tuple(task.fresh_x(rng)) for _ in range(50)}
    assert seen == {tuple(x) for x in X[:6]}  # prompts 0-2; prompts 3-4 are held out
    np.testing.assert_array_equal(np.stack(task.held_out_x()), X[6:])
    assert tuple(task.fixed_x) in seen and task.key == "128x64-fake-a/q_proj"


def test_real_task_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PCK_CACHE", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="capture_layer"):
        _ = Task(128, 64, "real", source="missing/q_proj").W
    fake_capture(tmp_path, "fake-b")
    with pytest.raises(ValueError, match="shape"):
        _ = Task(256, 64, "real", source="fake-b/q_proj").W
    with pytest.raises(ValueError, match="source"):
        Task(128, 64, "real")
    with pytest.raises(ValueError, match="captured"):
        activations("real", 4, np.random.default_rng(0))
    assert Task(128, 64).held_out_x() == []


def test_bits_round_trip() -> None:
    a = real.from_bits(np.arange(0, 65536, 97, dtype=np.uint16))
    np.testing.assert_array_equal(real.to_bits(a), np.arange(0, 65536, 97, dtype=np.uint16))
