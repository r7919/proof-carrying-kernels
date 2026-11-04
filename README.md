# proof-carrying-kernels

[![CI](https://github.com/r7919/proof-carrying-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/r7919/proof-carrying-kernels/actions/workflows/ci.yml)

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

Next: T3 (the checker's own FP64 and round-up arithmetic never lowers the threshold below
this bound, including the compensated dot product).

CI runs lint, type checks and the CPU tests on Python 3.11 to 3.14, compiles the CUDA
kernels for `sm_120`, and builds and axiom-audits the Lean proofs. Kernel tests run on a GPU machine, since hosted CI has no GPUs.

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
