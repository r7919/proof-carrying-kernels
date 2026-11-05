"""The PTX audit of the trusted checkers (pck/cuda/ptx_audit.py)."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import pck

# Loaded by path: importing the pck.cuda package needs torch, the audit does not.
_PATH = Path(pck.__file__).parent / "cuda" / "ptx_audit.py"
_SPEC = importlib.util.spec_from_file_location("ptx_audit", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
ptx_audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ptx_audit)
check_flags, instructions, violations = (
    ptx_audit.check_flags,
    ptx_audit.instructions,
    ptx_audit.violations,
)

PTX = """
.visible .entry k(
)
{
\tadd.rn.f32 \t%f1, %f2, %f3;
\t@%p1 ex2.approx.ftz.f32 \t%f4, %f5;
\t@!%p2 div.full.f32 \t%f6, %f7, %f8;
\tfma.rn.f64 \t%fd1, %fd2, %fd3, %fd4;
\tret;
}
"""


def test_reads_opcodes_with_their_modifiers() -> None:
    ops = instructions(PTX)
    assert ops["add.rn.f32"] == 1 and ops["ex2.approx.ftz.f32"] == 1 and ops["ret"] == 1
    assert ".visible" not in ops


def test_flags_approximations_outside_the_allowlist() -> None:
    assert violations(PTX, set()) == {"ex2.approx.ftz.f32": 1, "div.full.f32": 1}
    assert violations(PTX, {"ex2.approx.ftz.f32"}) == {"div.full.f32": 1}


def test_the_extension_builds_without_fast_math() -> None:
    assert check_flags((_PATH.parent / "_ext.py").read_text()) == []
    assert check_flags("['-O3', '--use_fast_math']") == [
        "the extension must build with --fmad=false",
        "the extension must not use use_fast_math",
    ]


def nvcc() -> str | None:
    home = os.environ.get("CUDA_HOME")
    if home and Path(home, "bin", "nvcc").exists():
        return str(Path(home, "bin", "nvcc"))
    return shutil.which("nvcc")


@pytest.mark.skipif(nvcc() is None, reason="needs nvcc")
def test_the_trusted_checkers_pass_the_audit() -> None:
    script = _PATH
    proc = subprocess.run(
        [sys.executable, str(script), "--nvcc", str(nvcc())], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ok   checker_kernel.cu" in proc.stdout
