# AGENTS.md

Notes for coding agents (and people) working in this repo. The README covers what the
project is; `docs/runbook.md` covers how to run everything.

## Commands

```bash
uv sync --extra rl                  # setup with every extra (GPU parts need CUDA 12.8 nvcc)
uv run pytest -q                    # all tests; GPU tests skip without a GPU
uv run pytest tests/gpu             # GPU tests only
uv run mypy                         # strict, whole tree
uv run ruff check . && uv run ruff format .
cd lean && lake build && lake env lean scripts/Axioms.lean   # proofs + axiom audit
```

CI installs only the base package (no torch), so check coverage there too:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/ci-venv uv sync --locked
UV_PROJECT_ENVIRONMENT=/tmp/ci-venv uv run pytest --cov   # must stay >= 95%
```

## Trusted files: the rule that matters most

The checker files listed in `TRUSTED` in `src/pck/manifest.py` are hash-pinned in
`src/pck/checker_lock.json`. Proof-gated episodes refuse to run if the hashes differ.

- After editing a trusted file, run `uv run python -m pck.manifest --update` and commit
  the new lock **in the same commit** as the edit.
- Do not edit trusted files while a training or benchmark run is in progress: the run
  will fail its pin check partway through.
- A change to a trusted file must keep matching the hypotheses of the Lean theorems in
  `lean/PCK/` (rounding direction, error constants). Say in the commit which bound it
  relies on.

## Git

- Small commits, one change each, straight on `main`. Never force-push.
- Commits made with an AI assistant end with a `Co-Authored-By:` trailer naming it.
- Pre-commit runs ruff, mypy and `uv lock --check`; pre-push runs pytest and Lean.
  Commit in dependency order so each commit passes on its own.

## Results and docs

- Every number in `README.md` and `docs/results.md` comes from a script in `harness/`
  whose output is saved next to it as `*_out.txt`. Regenerate the output rather than
  editing numbers by hand.
- Word claims no stronger than the evidence (name the shapes, seeds and caveats).
- Ask before starting GPU training runs: they take hours on the one GPU.

## Style

- Python 3.11+, fully typed, line length 100. Docstrings are short prose that say what
  a function guarantees, not `Args:`/`Returns:` blocks.
- Tests use pytest and Hypothesis; warnings are errors. GPU tests carry
  `pytest.mark.gpu`.
- Prefer clear names and plain explanations over comments that restate the code.
