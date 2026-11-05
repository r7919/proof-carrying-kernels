# proof-carrying-kernels

[![Python](https://github.com/r7919/proof-carrying-kernels/actions/workflows/python.yml/badge.svg)](https://github.com/r7919/proof-carrying-kernels/actions/workflows/python.yml)
[![Lean](https://github.com/r7919/proof-carrying-kernels/actions/workflows/lean.yml/badge.svg)](https://github.com/r7919/proof-carrying-kernels/actions/workflows/lean.yml)
[![CUDA](https://github.com/r7919/proof-carrying-kernels/actions/workflows/cuda.yml/badge.svg)](https://github.com/r7919/proof-carrying-kernels/actions/workflows/cuda.yml)

Proven accuracy checks for GPU inference kernels, to catch silent hardware faults and, later,
to stop RL agents from reward hacking.

A fast GPU kernel can be anything: hand-written, agent-written, buggy or adversarial. A small
trusted **checker** inspects every output after it runs, using secret randomized checksums.
Its threshold comes from a **Lean 4 proof**: it never raises a false alarm for any FP32
kernel within the declared summation-tree height, yet catches low-precision shortcuts,
dropped work, bit flips and wrong sampling sets. *Verify the checker, not the kernel.*

## At a glance

Measured on an RTX 5060 Laptop GPU; details in [docs/results.md](docs/results.md).

- **GEMV kernel** at 96-101% of copy bandwidth, 1.05-1.5x faster than `torch.mv` on large
  decode shapes, and bit-identical to its CPU model.
- **Checker overhead 3.9%** of a full Llama-3-8B decode layer (b = 128, one launch per layer).
- **Zero false alarms in 4 x 10^7 block checks**; honest outputs stay at least 180x below
  the threshold.
- **Every planted shortcut caught**: dropped tail, BF16 accumulators, BF16 partial sums, BF16
  output; 99-100% of exponent and sign bit flips.
- **Sampling checker** at 7-11% of the top-k sampler's cost and 19-26% of top-p
  selection's, catching every wrong set from an approximate (sharded) top-k.
- **Proven in Lean** (T1-T4, top-k, top-p), no `sorry`, standard axioms only.

## How it works

For each block of `b` output rows the checker holds secret signs `r`, checksums
`c = r^T W` and bounds `a = sum |W_i|` (rounded up), and after the GEMV computes

```
rho = c . x - r . y        tau = (gamma_h + u) * (a . |x|) + (its own rounding)
pass  iff  every y is finite  and  |rho| <= tau
```

`gamma_h` depends on the kernel's summation-tree height `h` (23 for n = 4096), not the
number of terms, which makes the proven threshold about 170x tighter than the textbook one.
See [docs/design.md](docs/design.md) for the threat model, the constants and the map from
each theorem to the code.

## What is proven

`lean/PCK/`, built and axiom-audited in CI.

| Theorem | Statement |
|---|---|
| `tree_sum_error` | Any summation tree of height h: \|fl(sum) - sum\| <= gamma_h * sum \|x_i\| |
| `row_within_bound` | A GEMV row summed by any tree visiting each column once is within that bound (covers both CUDA kernels) |
| `t1_block` | **T1, no false alarms**: for every sign vector and input, \|c . x - r . y\| <= (gamma_h + u) * (a . \|x\|) |
| `t2_block` | **T2, detection**: one output off by more than twice the threshold, others in spec: always caught |
| `t3_no_false_alarm`, `t3_single_fault_detected` | **T3**: T1 and T2 hold for the check *as computed* (rounded residual, round-up threshold) |
| `t4_computed` | **T4, randomized detection**: one output off by more than the threshold, others arbitrary (even crafted): passes at most half of all sign vectors, 2^-K for K of them |
| `t4_spread_computed` | **T4, spread-out errors**: every output only slightly off but \|\|D\|\|_2 >= 2x the threshold: passes at most 13/16 of sign vectors, (13/16)^K for K (Paley-Zygmund) |
| `topK_check_iff_max_sum` | The top-k check accepts exactly the k-sets of largest total logit |
| `topP_computed_minimal`, `topP_computed_sound` | **Top-p**: an accepted nucleus reaches mass p - slack and is no larger than any set of mass p + slack; a correct sampler always passes |

Assumptions (the rounding model, exact FP32 x BF16 products in FP64, the Dot2 bound for
compensated summation, and the accuracy of the top-p checker's masses) are listed in [docs/design.md](docs/design.md#assumed-not-proven).

## Getting started

Requires [uv](https://docs.astral.sh/uv/). For the GPU parts: Linux or WSL2, an NVIDIA
driver and CUDA 12.8 (`nvcc`); CUDA 12.8 supports host GCC up to 14, so on newer distros
install `g++-14` and set `NVCC_CCBIN=g++-14`.

```bash
uv sync                                    # CPU: library, tests, dev tools
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
uv run pytest
```

```bash
uv sync --extra gpu                        # PyTorch cu128; kernels compile on first use
uv run pytest tests/gpu
uv run python harness/bench_decode_layer.py
```

```bash
cd lean && lake exe cache get && lake build && lake env lean scripts/Axioms.lean
```

Every experiment is a script in `harness/`; [docs/results.md](docs/results.md) lists them
with their outputs.

## Layout

```
src/pck/        CPU library: FP helpers, bit-exact GEMV models, checker, sampling checker, faults
src/pck/cuda/   CUDA kernels and trusted checkers (csrc/), PyTorch bindings, timing
tests/          pytest + Hypothesis property tests (mirrors of T1, T2, T4)
tests/gpu/      GPU tests: kernels bit-exact against their models, checkers against the CPU ones
lean/           Lean 4 + Mathlib proofs (T1-T4, top-k, top-p)
harness/        experiments and benchmarks, with their recorded outputs
docs/           design and results
```

## Checks and CI

The pre-commit hooks run the checks locally, so CI is the final gate (the repo is private,
with limited Actions minutes):

- every commit: ruff check and format, mypy, `uv lock --check`, file hygiene;
- every push: the CPU test suite, plus the Lean build and axiom audit when `lean/` changed;
- by hand on a GPU machine: `uv run pytest tests/gpu`.

CI runs on pull requests and pushes to `main`, as three path-filtered workflows: Python
(lint, types, tests on 3.11 and 3.14), Lean (build and axiom audit) and CUDA (compile for
`sm_120` with warnings as errors). Hosted CI has no GPU, so kernel tests run locally.

## Status

| Phase | Status |
|---|---|
| 0. Setup | done |
| 1. Numerics study, GEMV kernels, GPU checker (overhead < 5%, zero false alarms) | done |
| 2. Lean proofs T1-T3 and CI | done |
| 3. Sampling checker, faults, tightness (synthetic and real activations), T4 | done |
| 4. Agent environment, regimes, exploit window, search baselines | done |
| 5-6. GRPO training, write-up | not started |
