# proof-carrying-kernels

[![Python](https://github.com/r7919/proof-carrying-kernels/actions/workflows/python.yml/badge.svg)](https://github.com/r7919/proof-carrying-kernels/actions/workflows/python.yml)
[![Lean](https://github.com/r7919/proof-carrying-kernels/actions/workflows/lean.yml/badge.svg)](https://github.com/r7919/proof-carrying-kernels/actions/workflows/lean.yml)
[![CUDA](https://github.com/r7919/proof-carrying-kernels/actions/workflows/cuda.yml/badge.svg)](https://github.com/r7919/proof-carrying-kernels/actions/workflows/cuda.yml)

Proven accuracy checks for GPU kernels, used to catch hardware faults and stop RL reward hacking.

A small trusted checker inspects every output of a fast (possibly buggy or adversarial)
GEMV kernel using randomized checksums. Its threshold is derived from the kernel's
reduction-tree height, so it never raises a false alarm for any FP32 kernel inside the
spec, yet still catches low-precision shortcuts, dropped work and bit flips.

## Status

Phase 1 complete (GEMV kernels, GPU checker, overhead and false-alarm gates met); Phase 2
(Lean proofs) next. The bit-faithful kernel model, checker and fault injector are in `src/pck/`,
tested against the properties the Lean theorems will state. Two CUDA GEMV kernels are in
`src/pck/cuda/`: a naive one and a coalesced one (16-byte loads, warp-shuffle reduction).
Each matches its CPU model in `pck.gemv` bit for bit, so its summation tree, and therefore
its proven error bound, is known exactly. The trusted checker also runs on the GPU
(`pck.cuda.GpuBlockedChecker`): FP64 residuals, with the threshold evaluated using
upward-rounded arithmetic. Lean proofs come next.

## Layout

```
src/pck/        Python package: FP helpers, GEMV emulator, blocked checker, fault injection
src/pck/cuda/   CUDA GEMV kernels and trusted checker (csrc/), PyTorch bindings, timing
tests/          pytest + Hypothesis property tests (mirrors of theorems T1 and T2)
tests/gpu/      kernel and checker tests on the GPU (need the gpu extra and a GPU)
lean/           Lean 4 + Mathlib proofs of the checker's guarantees (T1-T4)
harness/        experiments: numerics study, GEMV benchmark, checker overhead
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

### GPU (Linux or WSL2, CUDA 12.8)

Needs an NVIDIA driver (on Windows for WSL2) and the CUDA 12.8 toolkit (`nvcc`).
CUDA 12.8 supports host GCC up to 14; on newer distros install `g++-14` and set
`NVCC_CCBIN=g++-14`.

```bash
uv sync --extra gpu              # adds PyTorch cu128 and ninja
uv run pytest tests/gpu          # first run compiles the kernels (about a minute)
uv run python harness/bench_gemv.py
uv run python harness/bench_checker.py   # checker overhead by block size
```

### Lean proofs

```bash
cd lean && lake exe cache get && lake build   # Mathlib cache, then the proofs
lake env lean scripts/Axioms.lean              # fails on sorry or any non-standard axiom
```

Proven so far, in the standard rounding model for addition (`PCK.AddModel`), with no
`sorry` and only Lean's standard axioms:

| Theorem | Statement |
|---|---|
| `PCK.tree_sum_error` | Any binary summation tree t with height(t) * u < 1 has \|fl(t) - sum\| <= gamma_height(t) * sum \|x_i\| |
| `PCK.row_within_bound` | A GEMV row summed by any tree visiting each (zero-padded) column once, of height <= h, is within gamma_h * sum_j \|W_ij x_j\| of exact. Covers both CUDA kernels |
| `PCK.t1_block` | **T1 (no false alarms)**: for every sign vector and input, \|c . x - r . y\| <= (gamma_h + u) * (a . \|x\|), the checker's threshold before its own rounding |
| `PCK.t2_block` | **T2 (detection)**: if one output is off by more than twice that threshold and the rest are within spec, the residual exceeds the threshold, for every sign vector |
| `PCK.evalExact_le_evalUp` | Sums rounded up at every step (any tree) never fall below the exact sum: the threshold's `a . \|x\|` and `sum \|y\|` |
| `PCK.perturbed_tree_error`, `PCK.compound_error` | The checker's FP64 residual over compensated partials is within (gH + g2 + gH g2)(A + Y) + (1 + gH) eta of exact |
| `PCK.t3_no_false_alarm` | **T3**: the check as computed (rounded residual, round-up threshold) passes whenever T1's bound holds |
| `PCK.t3_single_fault_detected` | **T2 as computed**: one output off by more than 2 tau_hat always fails, for every sign vector |
| `PCK.topK_check_iff_max_sum`, `PCK.prefix_iff_min_ge_max` | **Sampling**: the top-k check accepts exactly the k-sets of largest total logit (any tie-breaking), and the kernel's min >= max test decides the same thing |

Assumed, not proven (stated as hypotheses where used): every operation satisfies the
standard rounding model (no overflow); FP32 x BF16 products are exact in FP64; and the
per-thread compensated FP32 dot product meets the Dot2 error bound of Ogita, Rump and Oishi
(2005), whose proof needs bit-level floating-point semantics. Formalizing the proofs found
one gap in the CUDA checker's constant: the Dot2 and FP64 errors compound, so `geps` now
includes their product term.

### Checks: local first, CI on pull requests and `main`

The pre-commit hooks run the checks locally, so CI is a final gate rather than the main
feedback loop (the repo is private, with limited Actions minutes):

- on every commit: ruff check and format, mypy, `uv lock --check`, file hygiene;
- on every push: the CPU test suite, and the Lean build plus axiom audit when `lean/` changed;
- by hand on a GPU machine: `uv run pytest tests/gpu`.

CI runs on pull requests and on pushes to `main`, split into three workflows that each run
only when their files change: Python (lint, types, tests on 3.11 and 3.14), Lean (build and
axiom audit) and CUDA (compile for `sm_120` with warnings as errors). Kernel tests run on a GPU machine, since hosted CI has no GPUs.

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

## GEMV benchmark

RTX 5060 Laptop GPU, cold L2, median of 200 runs, as a share of measured device copy
bandwidth (338 GB/s). Full output: `harness/bench_gemv_out.txt`.

| m x n | torch.mv (cuBLAS) | naive | coalesced/256 | coalesced vs torch.mv |
|---|---|---|---|---|
| 2048 x 2048 | 56% | 71% | 84% | 1.50x |
| 4096 x 4096 | 78% | 70% | 96% | 1.22x |
| 14336 x 4096 | 96% | 76% | 101% | 1.05x |

`torch.mv` writes a BF16 output while these kernels write FP32, so traffic is comparable
but the work is not identical. GEMV is read-only, which is why it can slightly exceed the
read-plus-write copy bandwidth. Below about 2048 x 2048, launch latency dominates.

## Checker overhead on a decode layer

One Llama-3-8B decode layer at batch 1 (fused qkv, o, gate_up, down: 416 MiB of BF16
weights), RTX 5060 Laptop GPU, cold L2, K = 1. Full output: `harness/bench_decode_layer_out.txt`.

| Mode | b = 64 | b = 128 |
|---|---|---|
| Serial (check after each GEMV) | 8.3% | 5.6% |
| Overlapped (checks on a side stream) | 9.6% | 9.3% |
| **Batched (one checker launch per layer)** | **6.1%** | **3.9%** |
| Checksum bank size, share of weights | 4.7% | 2.3% |

Zero false alarms in every mode. At b = 128 the batched checker meets the Phase 1 target of
under 5%. At b = 64 its memory traffic alone is 4.7%, and the single launch adds about
1.4%. Overlapping on a side stream is slower than serial: the GEMV already saturates the
GPU, so the checker has nothing to overlap with and only adds cross-stream latency.

## False alarms and bit-flip detection on the GPU

`harness/campaign.py` on the RTX 5060; full output: `harness/campaign_out.txt`.

**Zero false alarms in 40 million block checks**: 10^7 each for the coalesced and naive
kernels at b = 64 and b = 128 (K = 8), over a fused Llama-3-8B decode layer with fresh
Gaussian, Student-t, mixed-scale and outlier activations and periodically redrawn signs.
The largest |rho| / tau any legit output reached was 5.5e-3, a margin of about 180x.

Share of single bit flips detected (K = 1; 6,400-12,800 flips per cell):

| Bit | b = 64 Gaussian | b = 64 outliers | b = 128 Gaussian | b = 128 outliers |
|---|---|---|---|---|
| mantissa 14 | 0.1% | 52.5% | 0.0% | 21.5% |
| mantissa 16 | 44.1% | 87.3% | 11.7% | 75.2% |
| mantissa 18 | 84.6% | 96.7% | 69.5% | 94.0% |
| mantissa 20 | 96.3% | 99.2% | 92.3% | 98.2% |
| mantissa 22 | 99.0% | 99.8% | 98.1% | 99.7% |
| exponent 23-30, sign | 99.5-100% | 99.9-100% | 99.1-100% | 99.9-100% |

These match the CPU simulation (`harness/sim_out.txt`). More sign vectors (K = 4) do not
help single flips, as T2 predicts: a large single-output error is caught for every sign
vector. The few exponent and sign flips that are missed hit outputs near zero, where the
flip barely changes the value.

## Sampling checker

`pck.sampling` (CPU) and `pck.cuda.check_topk_gpu` / `check_topp_gpu` (GPU) check a
sampler's output in O(vocabulary): top-k by exact comparison (every selected logit >= every
rejected one, k distinct in-range indices, no NaN), top-p by the same prefix property plus
minimality of the nucleus with FP64 masses and an explicit tolerance, and that the sampled
token is in the set. The fault model `sharded_topk` reproduces the approximate-top-k bug
class (top logits clustered in one shard are dropped); the checker catches every wrong set.

Cost next to the sampler being checked (RTX 5060, V = 151,936, k = 50, p = 0.9; full output:
`harness/bench_sampling_out.txt`):

| Batch | top-k sampler | top-k checker | top-p selection | top-p checker |
|---|---|---|---|---|
| 1 | 386 us | 25 us | 256 us | 91 us |
| 8 | 363 us | 36 us | 814 us | 311 us |

## Tightness

`harness/tightness.py` measures how far the proven thresholds sit above the actual error of
the coalesced kernel (4096 x 4096, tree height 23; bit-exact CPU model; synthetic
activations). Full output: `harness/tightness_out.txt`.

| | Gaussian | Student-t | Outliers | Mixed-scale |
|---|---|---|---|---|
| Row bound / error, median (min) | 1118x (124x) | 853x (71x) | 194x (25x) | 252x (36x) |
| Block tau / \|rho\|, b = 64, median (min) | 8118x (1479x) | 6131x (1370x) | 1558x (268x) | 1909x (476x) |
| Block tau / \|rho\|, b = 128, median (min) | 11535x (2606x) | 9796x (1817x) | 2144x (560x) | 2977x (664x) |

- The proven block threshold is much looser than V-ABFT's statistical 48-158x (BF16): rho
  sums b row errors with random signs, which mostly cancel, while tau must cover the case
  where they all add up. That is the price of never false-alarming.
- The Higham-Mary probabilistic bound does not help here: at tree height 23 it is larger
  than the proven gamma_h (1.70e-6 vs 1.37e-6). Its sqrt(h) advantage only matters for long
  summation chains, which the tree-height bound already avoids.
- A per-element tolerance calibrated on Gaussian activations (10x the largest error seen)
  raises false alarms on 0.04% of outputs with outlier activations and 99.99% with
  mixed-scale ones; the proven threshold raises none, since it scales with sum |W x|.
