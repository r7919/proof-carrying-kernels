# Design

This document describes what the checkers guarantee, how, and under which assumptions. For
measured numbers see [results.md](results.md).

## Threat model and trust boundary

The kernels that do the work (GEMV, sampling) are **untrusted**: they may be buggy, run on
faulty hardware, or, during RL evaluation, be written by an agent that tries to pass the
check without computing the right answer. A small, fixed, hand-written **checker** inspects
every output after the kernel finishes. Only the checker is trusted, and the Lean proofs are
about the checker, not about the kernels.

```
 untrusted                                   trusted (small, fixed, proven)
 ┌──────────────────────────┐   y (FP32)    ┌────────────────────────────────────────┐
 │ GEMV kernel  y = W x     │ ────────────▶ │ blocked checksum checker               │
 │ (naive, coalesced, or    │               │   rho = c.x - r.y, tau = threshold     │
 │  agent-written)          │               │   pass iff y finite and |rho| <= tau   │
 └──────────────────────────┘               └────────────────────────────────────────┘
 ┌──────────────────────────┐ selection, token ┌─────────────────────────────────────┐
 │ top-k / top-p sampler    │ ───────────────▶ │ sampling checker (exact comparisons,│
 └──────────────────────────┘                  │ FP64 masses with explicit slack)    │
                                               └─────────────────────────────────────┘
```

Rules that keep the boundary meaningful:

- The checker is a separate kernel that reads `y` from memory after the producing kernel has
  finished; it never trusts anything the kernel reports about itself.
- Secret sign vectors `r` are drawn by trusted code. In production (non-adversarial faults)
  they are drawn once at weight load. In RL evaluation they must be drawn **after** the
  untrusted kernel finishes (commit-then-challenge), because untrusted code shares GPU
  memory and could otherwise read them. T4 assumes `r` is independent of `y`.
- The kernel writes `y` in FP32; downcasting to BF16 is a separate trusted step.
- Checker code is compiled without fast math (`--fmad=false`, explicit `__*_rn` / `__*_ru`
  intrinsics) and with warnings as errors (CI).

## The GEMV checker

For a block `beta` of `b` output rows of `y = W x` (BF16 `W`, `x`; FP32 accumulation):

| Quantity | Definition | Stored / computed |
|---|---|---|
| `r` | secret signs in {+-1}^b | int8, per check |
| `c` | `r^T W_beta` | computed in FP64, stored FP32 (nearest) |
| `a` | `sum_{i in beta} \|W_i\|` | stored BF16, **rounded up** |
| `rho` | `c . x - r . y_beta` | compensated FP32 per thread (Dot2), FP64 across threads |
| `s` | `a . \|x\|` | FP32, every operation rounded up |
| `tau` | `coef * s + geps * ((1 + u32) s + sum \|y\|) + tau_floor` | FP64, rounded up |

with `coef >= gamma_h + u32`, `geps >= gH + g2 + gH g2` (`gH = gamma_H(u64)` for the
height `H` of the checker's FP64 reduction, `g2 = gamma_{2L}(u32)^2` for chains of `L`
terms) and `tau_floor >= n 2^-148`, all rounded up on the host (`pck.cuda.checker`).

`h` is the summation-tree height of the kernel being checked: `tree_height(n, threads, vec)`
(23 for the coalesced kernel at n = 4096). Using the height instead of the number of terms
makes `gamma_h` about 170x smaller than `gamma_{n-1}`, which is what lets the proven
threshold catch BF16 shortcuts at all.

## From theorems to code

All theorems are in `lean/PCK/` and checked in CI (`lake build` plus an axiom audit that
rejects `sorry` and any non-standard axiom).

| Theorem | What it needs (hypotheses) | Where the code meets them |
|---|---|---|
| `tree_sum_error` | each addition satisfies the rounding model | FP32 `__fadd_rn` |
| `row_within_bound` | the kernel's tree visits every column once, height <= h | `emulate_gemv*` match the CUDA kernels bit for bit (`tests/gpu/test_gemv_cuda.py`); `tree_height` |
| `t1_block` (T1) | rows within bound; `c` within FP32 rounding of `r^T W`; `a >= sum \|W\|` | `GpuBlockedChecker.__init__` (FP64 `einsum`, `round_up_bf16`) |
| `t2_block` (T2) | as T1, one row off by more than 2 tau | |
| `evalExact_le_evalUp` | every operation rounds up | `__fadd_ru`, `__fmul_ru`, `__dadd_ru` in `checker_kernel.cu` |
| `perturbed_tree_error`, `compound_error` | FP64 reduction in the rounding model; Dot2 bound per chain (assumed) | `checker_kernel.cu` phase 1/2; `geps`, `tau_floor` in `checker.py` |
| `t3_no_false_alarm` (T3) | T1 plus the above | the kernel's verdict `fabs(rho) <= tau` |
| `t3_single_fault_detected` | T2 plus the above | |
| `t4_computed` (T4) | `r` uniform and independent of `y` | trusted sign draw; commit-then-challenge in evaluation |
| `t4_spread_computed` (T4, Paley-Zygmund) | as T4; deviation with \|\|D\|\|_2 >= 2 (tau + E + u a.\|x\|) | same |
| `topK_check_iff_max_sum`, `prefix_iff_min_ge_max` | logits are real numbers (no NaN) | NaN check and `min_sel >= max_rej` in `sampling_kernel.cu` |

### Assumed, not proven

- Every floating-point operation satisfies the standard rounding model (no overflow;
  gradual underflow makes subnormal sums exact).
- A product of an FP32 and a BF16 value is exact in FP64 (24 + 8 significand bits).
- The per-thread compensated dot product (TwoProduct + TwoSum, i.e. Dot2 of Ogita, Rump and
  Oishi 2005) is within `gamma_{2L}(u32)^2 * sum |c_j x_j|` of its chain's exact sum, plus
  an underflow floor. Its proof needs bit-level floating-point semantics (exactness of
  TwoSum), which the real-number rounding model cannot express.
- CUDA's FP64 `exp` is within 1 ulp (NVIDIA's documented bound), for the top-p masses.
- The proofs cover the checker's algorithm; the CUDA source is kept small and tested
  against FP64 references to close that gap.

Not yet proven: the top-p check with its tolerance.

## Design decisions and findings

- **b = 128 by default.** The checker's traffic is `(2K + 1) / b` of the GEMV's; at b = 64
  that alone is 4.7%, too close to the 5% budget. b = 64 stays available at about 2x finer
  detection.
- **BF16 `a`, rounded up.** Halves the bound's traffic for at most 0.8% looser `tau`.
- **Compensated FP32 instead of FP64 in the hot loop.** Consumer GPUs run FP64 at 1/64 of
  FP32; the FP64 FMA cost 8-10 us per check on large matrices. Dot2 costs nothing measurable
  and adds about 4e-12 relative to `tau`.
- **One checker launch per layer (`check_many`).** A checker launch has a fixed cost of
  about 10 us on this GPU under WSL2, which dominated small GEMVs. Batching up to 8 GEMVs
  per launch brought a Llama-3-8B decode layer to 3.9% overhead at b = 128.
- **Side streams do not help.** Overlapping checks with the next GEMV was slower than serial
  (9.3-9.6% vs 5.6-8.3%): the GEMV saturates the GPU, so there is nothing to overlap with.
- **Segmented reductions.** Both checkers split long rows across segments with a last-block
  merge (atomic ticket, fixed merge order), so a single row uses every SM.
- **The probabilistic tier is not needed.** At tree height 23 the Higham-Mary bound is
  larger than the proven `gamma_h`; its sqrt(h) advantage only matters for long chains.
- **Formalizing found a real gap.** The Dot2 and FP64 errors compound, so `geps` needed a
  `gH * g2` term the first version of the checker lacked (about 1e-25 relative, but a proof
  needs it).

## Limitations

- The proven block threshold is 1,500-12,000x above the actual residual (median), much
  looser than statistical checkers like V-ABFT (48-158x). It never false-alarms; the cost is
  coarser detection of small single-output errors (mantissa bits below about 16-18).
- Tensor cores are outside the spec: their accumulation order and rounding are not
  documented, so their tree height is unknown.
- All GPU measurements are from one laptop GPU (RTX 5060, WSL2); absolute numbers will
  differ on data-centre GPUs.
- Tightness and false-alarm studies so far use synthetic activations;
  `harness/real_activations.py` reruns them on a real model.
