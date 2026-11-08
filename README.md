# proof-carrying-kernels

[![Python](https://github.com/r7919/proof-carrying-kernels/actions/workflows/python.yml/badge.svg)](https://github.com/r7919/proof-carrying-kernels/actions/workflows/python.yml)
[![Lean](https://github.com/r7919/proof-carrying-kernels/actions/workflows/lean.yml/badge.svg)](https://github.com/r7919/proof-carrying-kernels/actions/workflows/lean.yml)
[![CUDA](https://github.com/r7919/proof-carrying-kernels/actions/workflows/cuda.yml/badge.svg)](https://github.com/r7919/proof-carrying-kernels/actions/workflows/cuda.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

**GPU kernel checkers whose accuracy thresholds are proven correct in Lean 4, and an
experiment showing they stop AI agents from cheating when the agents write kernels.**

[Design](docs/design.md) · [Results](docs/results.md) · [Runbook](docs/runbook.md)

<p align="center"><img src="docs/img/checkers.svg" width="760" alt="A guessed 1% tolerance lets a cheating kernel's answer of 99.9 pass; the proven rounding bound catches it, while the honest answer of 100.001 passes both."></p>

## The idea in one minute

AI models are starting to write GPU code, and they learn from a reward: *faster is better,
as long as the output is correct*. "Correct" is usually checked with a guessed tolerance,
such as "within 1% of the right answer". But honest floating-point rounding is far
smaller than 1%, so a kernel can **cut corners** (skip work, use lower precision), stay
inside the tolerance, and be paid more for being wrong. An AI being trained will find
that.

This project derives the tolerance instead of guessing it. Every floating-point
operation rounds by a known tiny amount, so the largest error an *honest* kernel can have
follows from a formula, and that formula is **proven correct in Lean 4**, a proof
assistant that checks every step of the math. A checker using that bound:

- **never rejects honest work.** That's proven, and there were 0 false alarms in 40 million GPU checks;
- **leaves no room to cut corners.** Every planted shortcut was caught;
- **is cheap.** It adds 3.9% to a full Llama-3-8B decode layer.

The same checkers also catch silent hardware faults such as bit flips. The motto: *verify the
checker, not the kernel*. The kernel can be hand-written, AI-written, buggy or
adversarial, and only the small checker has to be trusted.

<p align="center"><img src="docs/img/loop.svg" width="760" alt="The experiment loop: an AI agent writes a CUDA kernel, the GPU runs it, the checker judges it, and the reward trains the agent. Under a guessed tolerance the agent learned to cheat in 3 of 3 runs, under the proven bound in 0 of 3."></p>

## Key results

Measured on one RTX 5060 Laptop GPU; every number links to a script and its output in
[docs/results.md](docs/results.md).

| | Guessed tolerance (KernelBench-style) | Proven bound (this project) |
|---|---|---|
| AI trained with GRPO (Qwen3-0.6B, 3 seeds) | learned to cheat in **3 of 3** | **0 of 3** |
| Simple RL control policy (3 seeds) | learned to cheat in 2 of 3 | **0 of 3** |
| Search over kernels on held-out tasks | 73-92% of submissions are cheats | **no cheats**, 95% of the best honest speed |

- **Fast kernels:** the GEMV kernel reaches 96-101% of memory bandwidth, 1.05-1.5x faster
  than `torch.mv` on large decode shapes.
- **Checking an LLM's sampling live:** a vLLM plugin checks every token the model samples
  against its top-k/top-p rule. It caught 115 of 115 injected bad tokens with 0 false
  alarms.
- **Proofs:** 10+ theorems in Lean 4 and Mathlib, with no `sorry` and only the standard
  axioms, checked in CI.

<details>
<summary><b>All headline measurements</b></summary>

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
- **Every token vLLM samples checked** by a proven membership check (plugin, opt-in):
  115 of 115 injected bad tokens caught, 0 false alarms in 12,884 honest tokens; costs
  27-31% of decode throughput on Qwen3-0.6B, mostly kernel-launch overhead.
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
  - GRPO on Qwen3-0.6B (5-minute runs, one task, 3 seeds): under calibrated tests the
    policy learns activation sparsity in 3 of 3 seeds, paid 7% more on average; under
    the proof-gated checker it never does (0 of 3). Runs at a higher learning rate
    collapse early, keeping or dropping the hack by chance.
  - When the policy writes CUDA (exploit knobs locked, so cheating must be in code), it
    never invents the cheat in 15 minutes under either regime: 0% hacks, so this run
    doesn't separate the checkers yet.
- **Rollouts 7.8x faster with vLLM** (1,840 vs 233 tokens/s), serving the policy's LoRA
  adapter hot-loaded after every GRPO update.

</details>

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
| `topK_member_iff`, `topP_member_sound`, `topP_member_detects` | **Sampled-token membership** (for samplers that return only a token, like vLLM's): fewer than k larger logits iff the token is in a correct top-k set; a correct top-p sample always passes, and a rejected one is outside the nucleus by more than the sampler's tolerance |

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

The proven sampling check also runs inside vLLM as a plugin (`pck.vllm_plugin`). It
checks every token vLLM samples against its logits and the request's top-k/top-p
setting, on the GPU ([design](docs/design.md#the-check-inside-vllm),
[runbook](docs/runbook.md#the-sampling-check-inside-vllm)).

Training is GRPO with LoRA on a small chat model (Qwen3-0.6B in the smoke preset,
Qwen3-1.7B in the pilot), with rollouts from Hugging Face or, with `generator=vllm`, a
vLLM server that hot-loads the adapter every step. Every knob is in a TOML preset or a `key=value` override; see the
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
src/pck/rl/         GRPO trainer, vLLM rollouts, warm start, reward environments, prompts,
                    presets, control
src/pck/manifest.py pinned checker hash and certified-variant manifests
src/pck/member.py   sampled-token membership checks; vllm_plugin.py runs them inside vLLM
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

The pre-commit hooks run the checks locally, and CI is the final gate:

- every commit: ruff check and format, mypy, `uv lock --check`, file hygiene;
- every test run: the trusted checker must match its pinned hash (`src/pck/checker_lock.json`);
- every push: the CPU test suite, plus the Lean build and axiom audit when `lean/` changed;
- by hand on a GPU machine: `uv run pytest tests/gpu`.

CI runs on pull requests and pushes to `main`, as three path-filtered workflows: Python
(lint, types, tests on 3.11 and 3.14), Lean (build and axiom audit) and CUDA (compile for
`sm_120` with warnings as errors, plus the PTX audit of the trusted checkers). Hosted CI has no GPU, so kernel tests run locally.

## Status and limitations

The code, proofs and experiments described here are complete, and CI is green. The
project is **paused**, not abandoned. Open questions, with what would answer them:

- **Scale.** The RL results use small models (Qwen3-0.6B), one task and 3 seeds. A
  multi-turn Qwen3-1.7B pilot is configured (`--config pilot`) but not yet run.
- **Is the cheat discovered or offered?** In the RL runs, the corner-cutting option is
  part of what the agent can choose. When the agent has to write the cheat as CUDA
  itself, it never found it in 15 minutes under either checker
  ([results](docs/results.md#slot-code-the-policy-writes-cuda)), so that test doesn't
  separate them yet.
- **What is proven, exactly.** Lean proves the error formulas. The checker code that
  evaluates them is small, tested and hash-pinned, but not itself formally verified, and
  the hardware is assumed to round as IEEE 754 specifies
  ([assumptions](docs/design.md#assumed-not-proven)).
- **Cost of the live sampling check.** Inside vLLM it costs 27-31% of decode throughput
  on a small model, mostly kernel-launch overhead. A fused kernel or a CUDA graph would
  cut it.

## Citation

If you use this work, please cite it with the metadata in [CITATION.cff](CITATION.cff)
(GitHub shows it under *Cite this repository*).

## License

[Apache License 2.0](LICENSE).
