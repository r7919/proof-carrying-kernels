# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync --extra rl                   # setup with every extra (GPU parts need CUDA 12.8 nvcc)
uv run pytest -q                     # all tests; GPU tests skip without a GPU
uv run pytest tests/gpu              # GPU tests only
uv run pytest tests/test_checker.py::test_name   # a single test
uv run mypy                          # strict, whole tree
uv run ruff check . && uv run ruff format .
cd lean && lake build && lake env lean scripts/Axioms.lean   # proofs + axiom audit
```

CI installs only the base package (no torch). Check coverage the way CI sees it:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/ci-venv uv sync --locked
UV_PROJECT_ENVIRONMENT=/tmp/ci-venv uv run pytest --cov   # must stay >= 95%
```

## Architecture

- `src/pck/`: CPU library. FP helpers (`fp.py`), bit-exact CPU models of each GEMV
  kernel (`gemv.py`), the blocked checksum checker (`checker.py`), sampling checks
  (`sampling.py`, `member.py`).
- `src/pck/cuda/`: CUDA kernels and GPU checkers (`csrc/`), their bindings. Every kernel
  has a CPU model it must match bit for bit.
- `src/pck/agent/`: the kernel-writing environment (configs, tasks, regimes, slot-code
  sandbox). `src/pck/rl/`: GRPO training on it.
- `lean/PCK/`: Lean 4 proofs of the checker's thresholds (T1–T4, top-k, top-p).
- `harness/`: experiments; each script's output is saved next to it as `*_out.txt`.

## Trusted files

The files in `TRUSTED` in `src/pck/manifest.py` are hash-pinned in
`src/pck/checker_lock.json`; proof-gated episodes refuse to run if the hashes differ.

- After editing a trusted file, run `uv run python -m pck.manifest --update` and commit
  the new lock in the same commit as the edit.
- Never edit a trusted file while a training or benchmark run is in progress.
- Changes must keep matching the Lean theorems' hypotheses (rounding direction, error
  constants, overflow).

## Conventions

- Small commits, one change each, straight on `main`. Never force-push. Pre-commit runs
  ruff, mypy and `uv lock --check`; pre-push runs pytest and Lean.
- Numbers in `README.md` and `docs/results.md` come from `harness/` outputs: regenerate
  them, don't edit by hand. Word claims no stronger than the evidence.
- Ask before starting GPU training runs; they take hours on the one GPU.
- Python 3.11+, fully typed, line length 100. Docstrings are short prose saying what a
  function guarantees, not `Args:`/`Returns:` blocks. Warnings are errors in tests; GPU
  tests carry `pytest.mark.gpu`.
