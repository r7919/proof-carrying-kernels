# proof-carrying-kernels

[![CI](https://github.com/r7919/proof-carrying-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/r7919/proof-carrying-kernels/actions/workflows/ci.yml)

Proven accuracy checks for GPU kernels, used to catch hardware faults and stop RL reward hacking.

A small trusted checker inspects every output of a fast (possibly buggy or adversarial)
GEMV kernel using randomized checksums. Its threshold is derived from the kernel's
reduction-tree height, so it never raises a false alarm for any FP32 kernel inside the
spec, yet still catches low-precision shortcuts, dropped work and bit flips.

## Status

Phase 1. The bit-faithful kernel model, checker and fault injector are in `src/pck/`,
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

CI runs lint, type checks and the CPU tests on Python 3.11 to 3.14, and compiles the CUDA
kernels for `sm_120`. Kernel tests run on a GPU machine, since hosted CI has no GPUs.

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
