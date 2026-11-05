"""Audit the trusted checkers' PTX for approximate floating-point instructions.

The checkers' error bounds assume correctly rounded arithmetic (and, for `expf`, its
documented error), so their compiled code must not contain approximate or flush-to-zero
instructions that the source never asked for. This script compiles each trusted source
with the flags the extension uses (`pck.cuda._ext`) and fails if the PTX contains a
forbidden instruction outside the per-file allowlist, or if the extension's flags allow
contraction or fast math.

Allowlist:
  sampling_kernel.cu  ex2.approx.ftz.f32   the core of CUDA's accurate `expf` (with range
                                           reduction; at most 2 ulp), whose error the
                                           top-p bound takes as a hypothesis (`eps`, see
                                           pck.sampling.check_topp_gpu and TopP.lean)

Standard library only, so CI can run it right after installing nvcc:
  python3 src/pck/cuda/ptx_audit.py [--nvcc PATH] [--arch sm_120]
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
CSRC = HERE / "csrc"
TRUSTED = ("checker_kernel.cu", "sampling_kernel.cu")
ALLOWED: dict[str, set[str]] = {"sampling_kernel.cu": {"ex2.approx.ftz.f32"}}
FLAGS = ("-std=c++17", "-O3", "--fmad=false")
FORBIDDEN = re.compile(r"\.approx\b|\.ftz\b|\bdiv\.full\b")
INSTRUCTION = re.compile(r"^\s*(?:@!?%\w+\s+)?([a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*)", re.M)


def instructions(ptx: str) -> Counter[str]:
    """Opcode counts (with modifiers) of a PTX listing, directives excluded."""
    return Counter(m.group(1) for m in INSTRUCTION.finditer(ptx))


def violations(ptx: str, allowed: set[str]) -> dict[str, int]:
    return {
        op: n for op, n in instructions(ptx).items() if FORBIDDEN.search(op) and op not in allowed
    }


def system_include(nvcc: str) -> list[str]:
    """CUDA's headers as system headers, as torch extensions build them: CUDA 12.8's math
    declarations otherwise clash with glibc >= 2.41's."""
    found = shutil.which(nvcc) or nvcc
    include = Path(found).resolve().parents[1] / "include"
    return ["-isystem", str(include)] if include.is_dir() else []


def check_flags(ext: str) -> list[str]:
    """Problems with the extension's CUDA flags (the source of pck/cuda/_ext.py)."""
    problems = []
    if "--fmad=false" not in ext:
        problems.append("the extension must build with --fmad=false")
    for bad in ("use_fast_math", "--ftz=true", "--prec-div=false", "--prec-sqrt=false"):
        if bad in ext:
            problems.append(f"the extension must not use {bad}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nvcc", default=shutil.which("nvcc") or "nvcc")
    parser.add_argument("--arch", default="sm_120")
    args = parser.parse_args()
    failed = check_flags((HERE / "_ext.py").read_text())
    for problem in failed:
        print(f"FAIL {problem}")
    with tempfile.TemporaryDirectory() as tmp:
        for name in TRUSTED:
            out = Path(tmp) / f"{name}.ptx"
            cmd = [args.nvcc, *FLAGS, *system_include(args.nvcc), f"-arch={args.arch}"]
            cmd += ["-ptx", str(CSRC / name), "-o"]
            subprocess.run([*cmd, str(out)], check=True)
            ptx = out.read_text()
            bad = violations(ptx, ALLOWED.get(name, set()))
            used = {op: n for op, n in instructions(ptx).items() if op in ALLOWED.get(name, ())}
            if bad:
                failed.append(name)
                print(f"FAIL {name}: forbidden instructions {bad}")
            else:
                note = f" (allowed: {used})" if used else ""
                print(f"ok   {name}{note}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
