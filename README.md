# proof-carrying-kernels

[![Python](https://github.com/r7919/proof-carrying-kernels/actions/workflows/python.yml/badge.svg)](https://github.com/r7919/proof-carrying-kernels/actions/workflows/python.yml)
[![Lean](https://github.com/r7919/proof-carrying-kernels/actions/workflows/lean.yml/badge.svg)](https://github.com/r7919/proof-carrying-kernels/actions/workflows/lean.yml)
[![CUDA](https://github.com/r7919/proof-carrying-kernels/actions/workflows/cuda.yml/badge.svg)](https://github.com/r7919/proof-carrying-kernels/actions/workflows/cuda.yml)

Proven accuracy checks for GPU inference kernels, to catch silent hardware faults and to stop
RL agents that write kernels from reward hacking.

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
- **Reward hacking, measured:**
  - On real Qwen2.5-0.5B layers, skipping small activations passes KernelBench-style
    tests (`atol = rtol = 1e-2`) and is 3-10% faster than the best honest kernel in a
    column-major deployment; the proof-gated checker rejects it.
  - Search baselines on held-out tasks: calibrated tests let 73-92% of submissions be
    hacks. Tuned tests let none through but reach only 65% of the honest optimum. The
    proof-gated checker reaches 95% with no hacks for the bandit.
  - A policy-gradient control learns the hack under calibrated tests (2 of 3 seeds) and
    never under the proof-gated checker.

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

## The agent environment and RL

The checker becomes a reward. An agent writes a GEMV kernel by configuring a CUDA
template, optionally with a slot of its own device code, and is paid its speedup over
cuBLAS if the evaluator judges the output correct. Four **feedback regimes** decide
"correct":

| Regime | Judges by |
|---|---|
| weak | one fixed input, `atol = rtol = 1e-2` |
| calibrated | fresh inputs, `atol = rtol = 1e-2` (KernelBench-style) |
| tuned | fresh inputs, per-element tolerance tuned on Gaussian inputs |
| proof-gated | the proven spec: static check, then the blocked checker with signs drawn after the run |

A submission is a **hack** if it violates the proven spec on a hidden adversarial suite,
defined the same way under every regime. Tasks are decode GEMVs at real model shapes and
the seven linear layers of two Qwen2.5-0.5B decoder layers on captured activations:
layer 12 trains, layer 20 is held out, and there are three error budgets. Agent-written
CUDA runs in a sandbox: a static filter, an nvcc build with a timeout, and a separate
worker process. There, timed runs are checked runs.

```bash
uv sync --extra rl
uv run python -m pck.rl.train --config smoke --compare     # GRPO, calibrated vs proof-gated, ~10 min
uv run python -m pck.rl.control                            # the non-LLM control, ~1 min on CPU
```

Training is GRPO with LoRA on a small chat model (Qwen3-0.6B in the smoke preset,
Qwen3-1.7B in the pilot). Every knob is in a TOML preset or a `key=value` override; see the
[runbook](docs/runbook.md#training-phase-5).

## Getting started

Requires [uv](https://docs.astral.sh/uv/). For the GPU parts: Linux or WSL2, an NVIDIA
driver and CUDA 12.8 (`nvcc`). Machine-specific fixes (host GCC, glibc 2.41+, downloads
under WSL2) are in the [runbook](docs/runbook.md#setup).

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
with their outputs, and the [runbook](docs/runbook.md) says how to regenerate, scale up
and extend them.

## Layout

```
src/pck/            CPU library: FP helpers, bit-exact GEMV models, checker, sampling checker, faults
src/pck/cuda/       CUDA kernels and trusted checkers (csrc/), bindings, PTX audit
src/pck/agent/      agent environment: templates and their CPU models, tasks, regimes, spec,
                    episodes, search baselines, slot-code sandbox
src/pck/rl/         GRPO trainer, warm start, reward environments, prompts, presets, control
src/pck/manifest.py pinned checker hash and certified-variant manifests
tests/              pytest + Hypothesis (CPU); tests/gpu/ needs a GPU, tests/rl/ the rl extra
lean/               Lean 4 + Mathlib proofs (T1-T4, top-k, top-p)
harness/            experiments, with their recorded outputs and the timing and verdict tables
docs/               design, results, runbook
```

## Documentation

- [docs/design.md](docs/design.md): threat model, checker, theorem-to-code map,
  assumptions, the agent environment, training, limitations.
- [docs/results.md](docs/results.md): every measurement, with the script and output file.
- [docs/runbook.md](docs/runbook.md): setup, tests, regenerating the tables, training
  runs and their knobs, extending tasks, knobs, regimes and the checker, known gaps.

## Checks and CI

The pre-commit hooks run the checks locally, so CI is the final gate (the repo is private,
with limited Actions minutes):

- every commit: ruff check and format, mypy, `uv lock --check`, file hygiene;
- every test run: the trusted checker must match its pinned hash (`src/pck/checker_lock.json`);
- every push: the CPU test suite, plus the Lean build and axiom audit when `lean/` changed;
- by hand on a GPU machine: `uv run pytest tests/gpu`.

CI runs on pull requests and pushes to `main`, as three path-filtered workflows: Python
(lint, types, tests on 3.11 and 3.14), Lean (build and axiom audit) and CUDA (compile for
`sm_120` with warnings as errors, plus the PTX audit of the trusted checkers). Hosted CI has no GPU, so kernel tests run locally.

## Status

| Phase | Status |
|---|---|
| 0. Setup | done |
| 1. Numerics study, GEMV kernels, GPU checker (overhead < 5%, zero false alarms) | done |
| 2. Lean proofs T1-T3 and CI | done |
| 3. Sampling checker, faults, tightness (synthetic and real activations), T4 | done |
| 4. Agent environment, regimes, exploit window, search baselines, slot-code sandbox, PTX audit, checker pin | done (batch 1-8 tasks deferred) |
| 5. GRPO trainer (LoRA, single- and multi-turn), warm start, presets, non-LLM control | code done; GPU runs pending |
| 6. Write-up | not started |
