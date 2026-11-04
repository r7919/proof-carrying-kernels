# proof-carrying-kernels

[![CI](https://github.com/r7919/proof-carrying-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/r7919/proof-carrying-kernels/actions/workflows/ci.yml)

Proven accuracy checks for GPU kernels, used to catch hardware faults and stop RL reward hacking.

A small trusted checker inspects every output of a fast (possibly buggy or adversarial)
GEMV kernel using randomized checksums. Its threshold is derived from the kernel's
reduction-tree height, so it never raises a false alarm for any FP32 kernel inside the
spec, yet still catches low-precision shortcuts, dropped work and bit flips.

## Status

Phase 1, CPU side. The bit-faithful kernel model, checker and fault injector are in
`src/pck/`, tested against the properties the Lean theorems will state. CUDA kernels and
Lean proofs come next.

## Layout

```
src/pck/      Python package: FP helpers, GEMV emulator, blocked checker, fault injection
tests/        pytest + Hypothesis property tests (mirrors of theorems T1 and T2)
harness/      experiments, e.g. the numerics study that fixes block size b
```

## Development

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                          # create .venv with all dev tools
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
uv run pytest                    # tests (HYPOTHESIS_PROFILE=ci for the CI run)
uv run ruff check . && uv run ruff format --check . && uv run mypy
uv run python harness/numerics_sim.py   # numerics study, about 7 min on CPU
```

CI runs lint, type checks and the test suite on Python 3.11 to 3.14. GPU tests will run
locally, since hosted CI has no GPUs.

## Numerics study

`harness/numerics_sim.py` emulates a BF16 GEMV (m = 4096, n = 4100, 256 threads, tree
height 24) and checks every planted exploit. Output: `harness/sim_out.txt`.

| Case | b = 64, K = 8 | atol = rtol = 1e-2 |
|---|---|---|
| Legit FP32 kernel | pass | pass |
| Dropped tail | caught | caught |
| BF16 accumulators | caught | caught |
| BF16 partials before tree | caught | missed (Gaussian x) |
| BF16 output | caught | missed |

Zero false alarms in 12,800 block checks, for about 3.1% extra memory traffic.
