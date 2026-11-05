"""The non-LLM control policy (no model needed)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from pck.rl import control
from pck.rl.config import task_by_key


def test_control_trains_and_logs_each_step() -> None:
    log = control.train_control(task_by_key("896x896-gauss"), "proof-gated", steps=4, group=4)
    assert [s.step for s in log] == [0, 1, 2, 3]
    assert all(0 <= s.hack <= 1 and 0 <= s.exploit <= 1 and 0 <= s.top_prob <= 1 for s in log)
    assert json.loads(log[-1].top)["layout"] in ("row", "col")


def test_control_cli_prints_and_writes_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "steps.jsonl"
    argv = ["control", "--task", "896x896-gauss", "--steps", "3", "--seeds", "1"]
    monkeypatch.setattr(sys, "argv", [*argv, "--regimes", "calibrated", "--jsonl", str(out)])
    control.main()
    assert "calibrated" in capsys.readouterr().out
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 3 and rows[0]["regime"] == "calibrated"
