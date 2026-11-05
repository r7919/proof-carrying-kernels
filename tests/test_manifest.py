"""The pinned checker hash and per-variant manifests."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from pck import manifest
from pck.manifest import CheckerMismatch, Manifest, checker_hash, pinned, verify_pinned

LEAN = Path(manifest.ROOT) / "lean" / "PCK"


def test_the_lock_pins_the_current_checker() -> None:
    """Fails after any edit to a trusted file: review it, then python -m pck.manifest --update."""
    assert checker_hash() == pinned()["checker"]
    assert set(pinned()["files"]) == set(manifest.TRUSTED)
    verify_pinned()


def test_a_changed_checker_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock = pinned()
    lock["files"]["checker.py"] = "0" * 64
    lock["checker"] = manifest.combined(lock["files"])
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock))
    monkeypatch.setattr(manifest, "LOCK", path)
    with pytest.raises(CheckerMismatch, match=r"checker\.py"):
        verify_pinned()


@pytest.mark.skipif(not LEAN.exists(), reason="needs the Lean sources")
def test_the_named_theorems_exist() -> None:
    text = "\n".join(f.read_text() for f in LEAN.glob("*.lean"))
    for name in manifest.THEOREMS.values():
        short = name.removeprefix("PCK.")
        assert re.search(rf"^theorem {short}\b", text, re.M), name


def test_manifests_round_trip_and_verify() -> None:
    m = Manifest("896x896-gauss", {"layout": "row"}, None, {"h_spec": 64}, 1.2)
    assert Manifest.from_json(m.to_json()) == m
    assert m.problems() == []
    stale = Manifest("t", {}, None, {}, 1.0, checker="0" * 64, lean="1" * 64)
    assert len(stale.problems()) == 2 if manifest.lean_hash() else 1
