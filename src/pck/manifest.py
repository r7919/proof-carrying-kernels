"""Pinned checker hash and per-variant manifests.

The trusted checker's sources are hash-pinned in ``checker_lock.json``: the evaluator
refuses to run proof-gated episodes if the code on disk does not match (`verify_pinned`),
and a test keeps the lock current. Every variant the proof-gated evaluator certifies gets
a `Manifest`: its config and slot code, the spec's parameters, the checker hash, a hash of
the Lean sources, the theorems the certificate rests on, and the measured speedup.

Update the lock after a deliberate change to the checker:
  uv run python -m pck.manifest --update
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SRC = Path(__file__).parent
ROOT = SRC.parents[1]
LOCK = SRC / "checker_lock.json"
TRUSTED = (
    "fp.py",
    "checker.py",
    "sampling.py",
    "member.py",
    "cuda/checker.py",
    "cuda/sampling.py",
    "cuda/csrc/checker.h",
    "cuda/csrc/checker_kernel.cu",
    "cuda/csrc/sampling_kernel.cu",
    "cuda/csrc/bindings.cpp",  # shape checks, seg_len (the chain length the bound assumes)
    "cuda/_ext.py",  # compile flags (--fmad=false)
    "vllm_plugin.py",  # builds every input to the in-vLLM membership verdicts
)
THEOREMS = {
    "T1 no false alarms": "PCK.t1_block",
    "T2 guaranteed detection": "PCK.t2_block",
    "T3 safe runtime threshold": "PCK.t3_no_false_alarm",
    "T4 randomized detection": "PCK.t4_computed",
    "T4 spread errors": "PCK.t4_spread_computed",
}
"""What a GEMV certificate rests on (Lean names; lean/PCK/*.lean)."""


class CheckerMismatch(RuntimeError):
    """The checker on disk is not the pinned one."""


def file_hashes() -> dict[str, str]:
    return {f: hashlib.sha256((SRC / f).read_bytes()).hexdigest() for f in TRUSTED}


def combined(hashes: dict[str, str]) -> str:
    text = "".join(f"{name}\0{h}\n" for name, h in sorted(hashes.items()))
    return hashlib.sha256(text.encode()).hexdigest()


def checker_hash() -> str:
    return combined(file_hashes())


def pinned() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(LOCK.read_text())
    return data


def verify_pinned() -> None:
    """Raise `CheckerMismatch` naming the changed files if the checker is not the pinned one."""
    lock = pinned()
    if checker_hash() == lock["checker"]:
        return
    now = file_hashes()
    changed = sorted(f for f in TRUSTED if lock["files"].get(f) != now[f])
    raise CheckerMismatch(
        f"trusted checker differs from the pinned one ({', '.join(changed)}); "
        "review the change, then run: python -m pck.manifest --update"
    )


def lean_hash() -> str | None:
    """Hash of the Lean sources and build configuration (None outside a checkout)."""
    lean = ROOT / "lean"
    if not (lean / "lakefile.toml").exists():
        return None
    files = sorted(lean.glob("PCK/**/*.lean")) + [
        lean / n for n in ("PCK.lean", "lakefile.toml", "lake-manifest.json", "lean-toolchain")
    ]
    h = hashlib.sha256()
    for f in files:
        h.update(str(f.relative_to(lean)).encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


@dataclass(frozen=True)
class Manifest:
    task: str
    config: dict[str, Any]
    slot_code: str | None
    spec: dict[str, int]
    speedup: float
    checker: str = field(default_factory=checker_hash)
    lean: str | None = field(default_factory=lean_hash)
    theorems: dict[str, str] = field(default_factory=lambda: dict(THEOREMS))
    commit: str | None = field(default_factory=git_commit)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text: str) -> Manifest:
        return cls(**json.loads(text))

    def problems(self) -> list[str]:
        """Why this manifest does not match the pinned checker and current proofs."""
        out = []
        if self.checker != pinned()["checker"]:
            out.append("checker hash differs from the pinned checker")
        current = lean_hash()
        if current is not None and self.lean != current:
            out.append("Lean sources differ from the ones the certificate was made with")
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--update", action="store_true", help="re-pin the current checker")
    args = parser.parse_args()
    hashes = file_hashes()
    if args.update:
        LOCK.write_text(json.dumps({"checker": combined(hashes), "files": hashes}, indent=2) + "\n")
        print(f"pinned {combined(hashes)}")
        return
    try:
        verify_pinned()
        print(f"checker matches the pin {checker_hash()}")
    except CheckerMismatch as e:
        print(e)


if __name__ == "__main__":
    main()
