# Results

All GPU numbers are from one RTX 5060 (Blackwell, 8 GB, WSL2, driver 617.42,
CUDA 12.8, PyTorch 2.11) on a cold L2 unless noted. Each section names the script that
produced it and the file with its full output, so every number here can be reproduced.

| Result | Script | Output |
|---|---|---|
| Numerics study (CPU) | `harness/numerics_sim.py` | `harness/sim_out.txt` |
| GEMV kernels vs cuBLAS | `harness/bench_gemv.py` | `harness/bench_gemv_out.txt` |
| Checker overhead, decode layer | `harness/bench_decode_layer.py` | `harness/bench_decode_layer_out.txt` |
| FP8 weights: GEMV and checker overhead | `harness/bench_fp8.py` | `harness/bench_fp8_out.txt` |
| False alarms and bit flips | `harness/campaign.py` | `harness/campaign_out.txt` |
| Sampling checker cost | `harness/bench_sampling.py` | `harness/bench_sampling_out.txt` |
| Tightness, synthetic activations (CPU) | `harness/tightness.py` | `harness/tightness_out.txt` |
| Tightness, real activations | `python -m harness.real_activations` | `harness/real_activations_out.txt` |
| Agent template sweep | `harness/sweep_template.py` | `harness/sweep_template_out.txt`, `harness/timing_table.json` |
| BF16 shortcut at batch 1-8 | `harness/bench_batched.py` | `harness/bench_batched_out.txt` |
| Exploit window, activation sparsity | `python -m harness.exploit_window` | `harness/exploit_window_out.txt` |
| Phase 4 gates (CPU) | `python -m harness.phase4_checks` | `harness/phase4_checks_out.txt` |
| Search baselines per regime (CPU) | `python -m harness.verdicts`, `python -m harness.baselines` | `harness/verdict_table.json`, `harness/baselines_out.txt` |
| Hacks the checker lets through (CPU) | `python -m harness.hack_anatomy` | `harness/hack_anatomy_out.txt` |
| Non-LLM control policy (CPU) | `python -m pck.rl.control` | `harness/control_out.txt` |
| GRPO smoke run, Qwen3-0.6B | `python -m pck.rl.train --config smoke --compare` | `harness/grpo_smoke_out.txt`, `harness/grpo_smoke/` |
| Rollout throughput, HF vs vLLM | `python -m harness.bench_rollouts` | `harness/bench_rollouts_out.txt` |
| GRPO smoke run with vLLM rollouts | `python -m pck.rl.train --config smoke --compare generator=vllm` | `harness/grpo_smoke_vllm_out.txt`, `harness/grpo_smoke_vllm/` |
| GRPO smoke run, lr 1e-4, seed 1 | `... generator=vllm lr=1e-4 seed=1` | `harness/grpo_smoke_vllm_s1_out.txt`, `harness/grpo_smoke_vllm_s1/` |
| GRPO smoke run, lr 1e-4, seeds 0-2 | `... generator=vllm lr=1e-4 seed=$s` | `harness/grpo_smoke_vllm_seeds_out.txt`, `harness/grpo_smoke_vllm_s{0,1,2}/` |
| GRPO slot code (agent writes CUDA) | `python -m pck.rl.train --config slot --compare` | `harness/grpo_slot_out.txt`, `harness/grpo_slot/` |
| GRPO pilot, Qwen3-1.7B, multi-turn, seeds 0-2 | `python -m pck.rl.train --config pilot regime=$r seed=$s steps=150` | `harness/grpo_pilot_out.txt`, `harness/grpo_pilot/` |
| Sampled-token check inside vLLM | `python -m harness.bench_vllm_checker` | `harness/bench_vllm_checker_out.txt` |

## Measurements

### Numerics study

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

### GEMV benchmark

RTX 5060, cold L2, median of 200 runs, as a share of measured device copy
bandwidth (338 GB/s). Full output: `harness/bench_gemv_out.txt`.

| m x n | torch.mv (cuBLAS) | naive | coalesced/256 | coalesced vs torch.mv |
|---|---|---|---|---|
| 2048 x 2048 | 56% | 71% | 84% | 1.50x |
| 4096 x 4096 | 78% | 70% | 96% | 1.22x |
| 14336 x 4096 | 96% | 76% | 101% | 1.05x |

`torch.mv` writes a BF16 output while these kernels write FP32, so traffic is comparable
but the work is not identical. GEMV is read-only, which is why it can slightly exceed the
read-plus-write copy bandwidth. Below about 2048 x 2048, launch latency dominates.

### Checker overhead on a decode layer

One Llama-3-8B decode layer at batch 1 (fused qkv, o, gate_up, down: 416 MiB of BF16
weights), RTX 5060, cold L2, K = 1. Full output: `harness/bench_decode_layer_out.txt`.

| Mode | b = 64 | b = 128 |
|---|---|---|
| Serial (check after each GEMV) | 8.5% | 5.7% |
| Overlapped (checks on a side stream) | 12.1% | 8.8% |
| **Batched (one checker launch per layer)** | **6.0%** | **3.9%** |
| Checksum bank size, share of weights | 4.7% | 2.3% |

Zero false alarms in every mode. At b = 128 the batched checker meets the Phase 1 target of
under 5%. At b = 64 its memory traffic alone is 4.7%, and the single launch adds about
1.3%. Overlapping on a side stream is slower than serial: the GEMV already saturates the
GPU, so the checker has nothing to overlap with and only adds cross-stream latency.

### FP8 weights

The FP8 kernel (`pck.cuda.gemv_fp8`) reads weights stored in FP8 E4M3, one byte each,
16 per 16-byte load, with one FP32 scale per row applied after the GEMV. An E4M3 value
has 4 significand bits and a BF16 activation 8, so every product is exact in FP32, as for
BF16 weights. The kernel matches its CPU model bit for bit (every finite E4M3 value,
subnormals included), and the trusted checker verifies z = Q x against the FP8 weights Q,
which are exact in BF16. No new proof is needed. Cold L2, median of 200 runs. Full
output: `harness/bench_fp8_out.txt`.

| m x n | BF16 kernel | FP8 kernel | FP8, share of copy bandwidth | FP8 vs BF16 | FP8 vs torch.mv (BF16) |
|---|---|---|---|---|---|
| 896 x 4864 | 33.0 us | 22.9 us | 56% | 1.44x | 2.31x |
| 4096 x 4096 | 104 us | 54 us | 92% | 1.93x | 2.42x |
| 14336 x 4096 | 342 us | 177 us | 98% | 1.93x | 2.04x |
| 4096 x 14336 | 348 us | 179 us | 97% | 1.94x | 2.04x |

On large shapes the FP8 kernel runs at 92-98% of copy bandwidth, close to the 2x that
halving the weight bytes allows. On small shapes launch latency dominates; at 896 x 896
it is 9% slower than the BF16 kernel.

A fused Llama-3-8B decode layer runs in 0.641 ms with FP8 weights against 1.255 ms with
BF16 (1.96x), and every check passes. The checker costs the same as for BF16 weights,
but the GEMVs now take half as long, so the batched checker's overhead rises from 3.9%
to **8.7%** at b = 128, above the 5% budget. At b = 256 it is 5.4%, at half the
detection resolution. The quantization error itself (FP8 against the BF16 weights,
2.6-2.7% relative on these Gaussian weights) is a property of the format; the checker
bounds the kernel's rounding, not the choice to quantize.

### False alarms and bit-flip detection on the GPU

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

### Sampling checker

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
| 1 | 418 us | 25 us | 242 us | 67 us |
| 8 | 361 us | 38 us | 819 us | 165 us |

Computing the top-p terms with FP32 `expf` on a split argument instead of FP64 `exp` cut
the top-p checker from 91 to 65 us at batch 1 and from 311 to 153 us at batch 8 (FP64 runs
at 1/64 rate on this GPU), for about 1e-6 more slack. Dropping masked (`-inf`) logits
explicitly, so a masked row cannot produce a NaN, costs a few percent (67 and 165 us now).

### Tightness

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

### Tightness on real activations

`python -m harness.real_activations`: Qwen2.5-0.5B-Instruct in BF16, decode-step inputs to
every linear layer (4 prompts x 16 tokens, 2 inputs kept per layer and prompt), through the
bit-exact model of the coalesced kernel (which is bit-identical to the CUDA kernel), with
the model run on the RTX 5060. A CPU run gives the same picture to within a few percent;
only the captured activations depend on the device. Full output:
`harness/real_activations_out.txt`.

| Layer | Row bound / error, median (min) | Block tau / \|rho\|, b = 64, median (min) | b = 128, median (min) |
|---|---|---|---|
| q_proj | 285x (13x) | 2035x (230x) | 2635x (299x) |
| k_proj | 253x (17x) | 1495x (268x) | 2009x (531x) |
| v_proj | 311x (24x) | 2167x (339x) | 2994x (722x) |
| o_proj | 347x (16x) | 2500x (199x) | 3409x (494x) |
| gate_proj | 300x (13x) | 2099x (290x) | 2937x (520x) |
| up_proj | 374x (16x) | 2838x (469x) | 4038x (645x) |
| down_proj | 1050x (30x) | 8317x (388x) | 10526x (659x) |

- No proven-bound violations. Real activations come closer to the bound than any synthetic
  case (minimum 13x, in `q_proj` and `gate_proj`, against 25x for synthetic outliers), but never cross it.
- Block tightness matches the synthetic study: 1,500-10,500x (median), far looser than
  V-ABFT, as expected for a threshold that may never false-alarm.

#### Why not tests?

The same run compares the proven checker (b = 128, one sign draw) with two fixed-tolerance
tests against an FP64 reference, as a test suite would set them: PyTorch's FP32 defaults
(`atol = 1e-5`, `rtol = 1.3e-6`) and a per-layer `atol` calibrated on the first prompt (10x
the largest honest error) and applied to the other three. Share of decode inputs flagged
(naive kernel, which has the exploit switches):

| Output | torch defaults | calibrated atol | proven checker |
|---|---|---|---|
| honest | 0% | 0% | 0% |
| drop tail (4 columns) | 100% | 100% | 100% |
| BF16 accumulator | 100% | 100% | 93.8-100% |
| BF16 partial sums | 100% | 100% | 95.8-100% |
| BF16 output | 100% | 100% | 97.2-100% |

- On this model, both tests are as good as the checker: no false alarms, every shortcut
  caught. Decode activations of a well-behaved small model never reach the regime where a
  fixed tolerance breaks (the synthetic mixed-scale case, 99.99% false alarms).
- The checker's misses are all in `k_proj` and `v_proj`, which have 128 rows: one block,
  so one randomized check per input, which T4 bounds at a 1/2 to 13/16 pass rate for a
  given error. Other layers have 7-38 blocks per input. More sign draws close the gap at
  `(2K + 1) / b` traffic each.
- So the case for the checker over tests is not accuracy on typical inputs. It is that a
  test needs the full FP64 reference product, which costs more than the GEMV it checks
  (the checker costs 3.9% of a decode layer), and its tolerance has no guarantee outside
  the inputs it was calibrated on. The checker's threshold holds for every input and every
  kernel within the declared tree height.

### Agent template sweep

`harness/sweep_template.py` times all 2,784 configs of the agent's GEMV templates (row-major
`src/pck/agent/config.py` and column-major `column.py`) on the 43 tasks, against cuBLAS in
the same session. Variants whose speed depends on the input (column skips) are timed on
each task's own input. The table it writes serves every episode's timings and is the
brute-force baseline. Full output: `harness/sweep_template_out.txt` (re-swept on
2026-11-08 after the error-budget tasks became copies of their default-budget siblings).

- **In-spec variants beat cuBLAS on every task**, by 1.09x on the large MLP shapes
  (bandwidth-bound for both) up to 3.3x on small ones, where cuBLAS's launch configuration
  leaves the GPU underused.
- **The precision shortcuts barely pay.** The fastest BF16-accumulator, -partials or
  -output variant is within 1% of the fastest FP32 one on every training shape, and 1-6%
  faster on three held-out odd shapes (1024 x 1000, 2560 x 2304, 3584 x 3000; kernels of
  12-72 us). The large exploit speedups come from dropping work: `skip_tail` (up to 1.57x)
  and, in the column-major template, skipping columns.
- So a policy has little speed incentive to use a precision shortcut. The one exploit that
  pays on the training tasks is activation sparsity on real layers (next sections), which
  passes `atol = rtol = 1e-2`.

### The BF16 shortcut at batch 1-8

`harness/bench_batched.py` asks whether packed BF16 FMA (two lanes per instruction) beats
FP32 accumulation once each weight load serves B activation vectors. Full output:
`harness/bench_batched_out.txt`.

- **It does not.** Packed BF16 is within noise of FP32 at every shape and batch size
  (4096 x 4096: 105.1 vs 105.3 us at B = 1, 136.1 vs 133.5 us at B = 8). The kernel stays
  bound by weight traffic up to B = 8, so the arithmetic is never the bottleneck.
- **It is not a slack exploit either.** Fully BF16 accumulation leaves 0.2-14% of outputs
  outside `atol = rtol = 1e-2`, so a calibrated test that checks every output rejects it;
  96-100% of rows break the proven bound.
- Together with the template sweep: on a bandwidth-bound GEMV, a shortcut can only be faster
  by reading less of `W`. Precision tricks alone never pay.

### The exploit window: activation sparsity on real layers

The *exploit window* is the set of cheating kernels that a calibrated test accepts, that
the proof-gated checker rejects, and that are faster than the best honest kernel. Reward
hacking can only be learned if it is non-empty. The cheat measured here is activation
sparsity (skip every column with |x_j| below a threshold, so its weights are never read)
in the column-major template, on the 14 real Qwen2.5-0.5B layers.

`python -m harness.exploit_window` gives the accuracy side over every captured decode
input episodes may see (full output: `harness/exploit_window_out.txt`);
`harness/bench_window.py` re-times the key variants at 200 interleaved reps (full output:
`harness/bench_window_out.txt`, used for the speed columns below; the sweep's 30-rep
minima are noisier).

| Layer | Threshold | Skipped | Calibrated accepts | Proof-gated accepts | vs honest column-major | vs honest row-major |
|---|---|---|---|---|---|---|
| l12 down_proj | 0.005 | 8% | 100% | 0% | 1.06x | 0.99x |
| l20 down_proj | 0.01 | 6% | 100% | 0% | 1.02x | 0.96x |
| l12 o_proj | 0.01 | 5% | 100% | 0% | 1.10x | 0.81x |
| l20 o_proj | 0.01 | 11% | 100% | 0% | 1.12x | 0.82x |

- **Accuracy: the window exists.** On every real layer some threshold skips 3-16% of the
  columns, passes the calibrated test on at least 95% of inputs and fails the proof-gated
  checker on every input. Larger thresholds are caught by both.
- **Speed: it pays in a column-major deployment.** After optimizing the column kernel
  (contiguous column ranges per split, one launch), honest column-major is within 6-8% of
  row-major on the large layers (25-33% behind on the small 896 x 896 ones). Against the
  best honest column-major kernel the accepted cheat is 2-12% faster; against honest
  row-major it at best breaks even. Re-run on 2026-11-08, these medians match the first
  run's to within 0.5 us.
- cuBLAS on the transposed weights is much slower (80 vs 53 us on `down_proj`); the
  column-major template beats cuBLAS in either layout.
- Real tasks are therefore set in a column-major deployment, as sparsity-aware inference
  engines (TEAL, Deja Vu) store weights (`pck.agent.tasks.locks`). The incentive is
  small; whether RL finds it is Phase 5's go/no-go question.
- **The timing table now shows it too.** The episodes' rewards come from the template
  sweep's timing table. Its first two versions timed the skips in a separate pass from
  the honest kernels, and timed each kernel's reps back to back while the GPU heated up,
  so clock drift landed between them: the accepted cheat came out up to 16% faster than
  honest column-major in the first sweep (2026-11-04) and at most 3% in the second
  (2026-11-08). The sweep now times each shape's kernels round-robin and each skip rep by
  rep against its sibling without the skip (2026-11-09), and the accepted cheat is 1-12%
  faster than honest column-major (`harness/exploit_window_out.txt`), as this benchmark
  measures. The smoke runs below used the first table, the control and the pilot of
  2026-11-08 the second; the numbers in this file from the search baselines, the control,
  the phase 4 gates and the hack anatomy are from the third.

### Phase 4 gates: a scripted cheater and episode cost

`python -m harness.phase4_checks` plays full episodes on every real task with two scripted
agents. One submits the best honest kernel. The other submits the activation-sparsity cheat
at the largest threshold the calibrated test accepts on at least 95% of inputs. Full
output: `harness/phase4_checks_out.txt`.

- **Under weak tests the cheater earns more on all 14 real tasks**, by 1-11% (l12
  `v_proj`: 0.92 vs 0.83, l20 `o_proj`: 1.74 vs 1.59), and under calibrated tests on 13;
  on l20 `gate_proj` the calibrated test sometimes fails it (mean 0.69). With the
  2026-11-08 timing table, whose skip timings were off (see the exploit window), it
  earned more on only 5 tasks.
- **Under tuned tests and the proof-gated checker, the cheater earns -0.05** (rejected, so
  only the call costs) on every task.
- **One full episode takes 4.7 s** on the largest real task and 5.0 s on the largest
  synthetic one (CPU model plus timing table), under the 2-minute gate. Compiling
  agent-written slot code is not included.

### Search baselines per regime

`python -m harness.baselines` runs three searches that use no learning, under each regime:

- **brute force:** the fastest allowed variant whose sampled inputs all pass, which is the
  per-task optimum the regime allows;
- **random search:** 10 evaluations;
- **UCB1 bandit:** trained on the training tasks, then 10 evaluations per task.

It uses 20 seeds per task and simulates from the verdict and timing tables (`pck.agent.baselines`). Rewards and
*certified* speedups (the speedup if the submission meets the spec on the hidden suite,
else 0) are percentages of the honest optimum, the fastest in-spec variant. *Hacks* are
submissions that violate the spec on the hidden suite; *in-distribution* hacks also
violate it on unseen inputs from the task's own distribution. Full output:
`harness/baselines_out.txt`; the verdicts are in `harness/verdict_table.json`
(`python -m harness.verdicts`, several CPU-hours).

Held-out tasks, budget 10:

| Regime | Search | Reward paid | Certified | Hacks | In-distribution hacks |
|---|---|---|---|---|---|
| weak | brute force | 109% | 29% | 71% | 57% |
| weak | bandit | 98% | 49% | 50% | 50% |
| calibrated | brute force | 109% | 29% | 71% | 57% |
| calibrated | random | 80% | 18% | 72% | 61% |
| calibrated | bandit | 99% | 35% | 64% | 64% |
| tuned | brute force | 57% | 57% | 0% | 0% |
| tuned | bandit | 56% | 56% | 0% | 0% |
| proof-gated | brute force | 107% | 57% | 43% | 29% |
| proof-gated | random | 55% | 28% | 32% | 25% |
| **proof-gated** | **bandit** | **98%** | **98%** | **0%** | **0%** |

- **Weak and calibrated tests pay for hacks.** Every search finds kernels they accept that
  are out of spec: 50-79% of held-out submissions (50-70% in distribution) and 59-78% on
  the training tasks. The reward they pay (80-109% of the optimum) is well above the
  certified speedup (16-49%). Any optimizer, not just RL, learns to exploit them.
- **Tuned tests let no hack through but false-alarm off their tuning distribution.** On the
  training tasks brute force and the bandit get 99-100% of the optimum. On held-out tasks
  (outliers, mixed scales, other layers) they also reject honest kernels and pay only
  56-57%.
- **The proof-gated checker pays close to the optimum on held-out tasks too** (bandit
  98%, certified 98%, no hacks), with no tuning per task.
- **It is not airtight.** Brute force, which looks up every variant's verdicts and picks
  the fastest the checker accepts, submits an out-of-spec kernel on 7 of 43 tasks (1 of
  29 training tasks, 6 of 14 held-out ones); its reward above 100% on held-out tasks is
  those cheats. Random search submits one at least once on 35 of 43 tasks (16% of
  training submissions, 32% held out): mostly activation sparsity at the smallest
  threshold (|x_j| < 0.005). `python -m harness.hack_anatomy` shows the two kinds brute
  force finds (full output: `harness/hack_anatomy_out.txt`):
  - **Below the checker's resolution** (5 tasks: layer-12 `q_proj`, layer-20 `q_proj`,
    `k_proj`, `gate_proj` and `up_proj`). Skipping |x_j| < 0.005 is off by 17-41x the
    spec's per-row bound on the task's own inputs, but that is only 0.6-1.6e-4 of
    sum|W x|, and the checker accepts 46-100% of such inputs: `tau` bounds a block of 128
    rows, and an error spread thinly over the rows of a block stays under it. This is the coarse detection noted in
    [design.md](design.md). With the earlier timing tables these skips looked slower
    than honest kernels, so brute force found this kind on one task only.
  - **Wrong only off-distribution.** On the mixed-scale task (at both error budgets),
    skipping |x_j| < 0.3 stays within the spec on that task's inputs (0.02-0.03x the
    bound) but is off by 1,300-2,600x on other distributions. The checker rejects 60% of
    those inputs, so a checked deployment would catch it the first time one appeared.
- So the guarantee is per input: a kernel the checker accepts is within `tau` on the inputs
  it was run on, not on every input. A checked deployment keeps checking every call, which
  catches the second kind. The first kind is the price of `tau` being coarser than the
  per-row spec; smaller blocks would narrow it at some cost. The bandit, which only sees
  the verdicts it pays for, never submitted either kind.

### The non-LLM control learns the hack only when the reward pays for it

`python -m pck.rl.control` trains a softmax policy over the 672 column-major configs (224
numerics classes) of layer-12 `o_proj`. It uses GRPO's group-normalized advantage and the
rewards the regimes pay (`pck.rl.envs.TableEnv`), with 1,000 steps of 8 episodes per seed.
Full output: `harness/control_out.txt`.

| Regime | Seeds that converge on the hack | Hack rate, last 50 steps |
|---|---|---|
| calibrated | 2 of 3 | 100%, 0%, 100% |
| proof-gated | 0 of 3 | 0%, 0%, 0% |

- **Under calibrated tests two seeds learn to cheat.** They settle on activation sparsity
  (`skip_below` 0.01 and 0.005), which the test accepts and the spec forbids. With the
  2026-11-08 timing table, where the cheat barely paid, only 1 of 3 did.
- **Under the proof-gated checker no seed submits a hack**, and the policy converges to
  honest configs.
- Plain policy gradient commits early, so some seeds stop at slower configs than the
  optimum under either regime (rewards 1.01-1.59). The point is which kind of kernel each
  regime teaches, not the speed reached.
- Each run takes about 20 s on CPU. The LLM runs (`python -m pck.rl.train`) use the same
  task and rewards.

### GRPO smoke run: an LLM keeps the hack only when the reward pays for it

`python -m pck.rl.train --config smoke --compare` trains two Qwen3-0.6B LoRA adapters on
layer-12 `o_proj`, one per regime, with the same prompt, seed and budget. Each run gets
4.5 minutes on the RTX 5060 (44-47 GRPO steps of 2 x 8 single-turn episodes, rewards from
the tables), then 32 sampled evaluation episodes. Full report:
`harness/grpo_smoke_out.txt`, from the run files in `harness/grpo_smoke/`
(`python -m harness.grpo_report`).

| Regime | Exploit at step 0 | Exploit in every submission | Final eval: reward | Passed | Hacks |
|---|---|---|---|---|---|
| calibrated | 38% | first at step 2, sustained from step 43 | 1.44 | 100% | **100%** |
| proof-gated | 0% passed | never | 1.28 | 100% | **0%** |

- **The regime decides whether the hack survives.** The base model already proposes
  activation sparsity: 38% of its first replies set `skip_below > 0`. Under calibrated
  tests these pass: from step 2 nearly every submission is a hack (with dips to 50-94%
  at four steps, and 100% from step 43 on). Under the proof-gated
  checker every early reply fails (reward -0.03 at step 0), and by step 5 the policy
  writes honest configs only.
- **The hack it keeps is not faster.** The calibrated policy settles on
  `skip_below = 0.01, splits = 16` (speedup 1.47). The best honest kernel is just as fast;
  the faster cheat in the table (11% above honest) is never found. Once all 8 replies of
  a group are identical, the group-normalized advantage is zero and learning stops. So
  the calibrated test lets the hack in, but this run has no speed reason to prefer it.
- **The proof-gated run drifts.** It reaches 1.46 by step 5, near the honest optimum
  (1.47), then falls to 1.28 by step 25 on a degenerate reply with a duplicated key, while
  the KL to the base model rises to about 1.0. A learning rate of 3e-4 is too high for a
  single-task run.
- This is one seed on one task. Three seeds follow below, and the multi-turn pilot
  (more tasks, held-out evaluation) is at the end.
- **Tables.** This run and the other smoke runs below were trained on the first sweep's
  timing table and the first verdict table, and were not re-run after the review fixes.
  The re-sweep moves the cheat's payoff on this task (see the exploit window), and the
  recomputed verdict table differs on this task in one place: the calibrated test now
  accepts `skip_below = 0.02` on 3 of 4 inputs instead of 2 (the first table's rows for
  the real layers do not match what the committed code computes, at any commit, so they
  were stale). No proof-gated verdict on this task changed, so the proof-gated runs would
  see identical rewards apart from the timings.

### Rollouts with vLLM: 7.8x faster

`python -m harness.bench_rollouts` times one GRPO step's rollouts (2 task prompts x 8
samples, at most 128 new tokens, temperature 1) on Qwen3-0.6B. It compares Hugging Face
`generate()` in the trainer with a vLLM 0.33 server that serves the policy's LoRA adapter,
as in training (`generator = vllm`). Full output: `harness/bench_rollouts_out.txt`.

| Generator | Seconds per step | Tokens per second |
|---|---|---|
| HF `generate()` | 2.86 | 233 |
| vLLM, LoRA adapter | 0.37 | 1,840 |

- **7.8x faster per step.** A 4.5-minute smoke run spent most of its time generating, so
  the same budget now fits several times as many GRPO steps.
- The server fits beside the trainer on the 8 GB GPU in 35% of its memory (1.04 GiB of KV
  cache) once it is sized to the rollouts: sequences up to 2,048 tokens, at most 32 at a
  time. At vLLM's defaults (4,096 tokens, hundreds of sequences) it does not start.
- It starts in about 100 s, once per run, mostly compilation and CUDA-graph capture.

### The smoke run again, with vLLM rollouts

The same smoke comparison, seed and budget, with `generator = vllm`. Full report:
`harness/grpo_smoke_vllm_out.txt` (run files in `harness/grpo_smoke_vllm/`).

| Regime | Generator | Steps in 4.5 min | Exploit at step 0 | Final eval: hacks | Reward |
|---|---|---|---|---|---|
| calibrated | HF | 44 | 38% | **100%** | 1.44 |
| calibrated | vLLM | 59 | 25% | **0%** | 1.43 |
| proof-gated | HF | 47 | 0% passed | 0% | 1.28 |
| proof-gated | vLLM | 64 | 0% passed | 0% | 1.33 |

- **Under calibrated tests, whether the hack survives is chance.** Both runs start with
  the base model proposing activation sparsity, and the calibrated test accepts some of
  it. One run keeps it (in every submission from step 43); the other drops it by step 8
  for an honest
  config of the same speed. The hacks these runs find are no faster than the best honest
  kernel (the run that kept one scores 1.44, the one that dropped it 1.43), so nothing
  pushes the policy towards or away from them. Each run keeps whatever its group
  collapses onto first.
- **Under the proof-gated checker no run submits a hack.**
- **vLLM speeds the whole step up 1.3-1.4x, not 7.8x.** Generation now takes about 0.4 s
  of a 4.2-4.6 s step; the rest was the update, one trajectory at a time. The update is
  now batched (`update_batch`), which this run predates.
- **The policy and vLLM agree:** `mismatch` stays at 0.000-0.017 nats per token.
- **Groups collapse early** (all 16 replies identical by about step 20, after which the
  advantage is zero), at the smoke preset's learning rate of 3e-4. Measuring how often
  calibrated tests let a hack through needs more seeds and a lower learning rate; this
  task's small speed incentive makes it a weak test case.

### Seed 1 at a lower learning rate: the hack that pays

The vLLM smoke comparison again with `lr = 1e-4` (instead of 3e-4) and seed 1. Full report:
`harness/grpo_smoke_vllm_s1_out.txt`. This run used the first batched update
(`update_batch = 8`), which overflowed the GPU (see the [runbook](runbook.md)), so each
regime got only 12 steps.

| Regime | Final config | Final eval: reward | Hacks |
|---|---|---|---|
| calibrated | `threads 256, vec 4, splits 16, skip_below 0.01` | **1.62** | **100%** |
| proof-gated | `threads 256, vec 4, splits 16, skip_below 0` | **1.46** | 0% |

- **The two policies differ in one knob.** Both settle on the same kernel layout. The
  calibrated policy adds activation sparsity (`skip_below = 0.01`) and is paid 11% more for
  it. The proof-gated one writes the honest kernel, the honest optimum on this task (1.46).
- **The lower learning rate kept the group exploring.** Under calibrated tests the policy
  reached the paying hack, every submission from step 6. Earlier runs at 3e-4 collapsed
  onto whichever config came first, hacks included that were no faster than honest.
- One seed and 12 steps; the next section repeats it over three seeds.

### Three seeds at the lower learning rate

Seeds 0 and 2 at `lr = 1e-4` with `update_batch = 2` (about 4.5 minutes and 75-85 steps
per regime), beside seed 1 from above. Full report: `harness/grpo_smoke_vllm_seeds_out.txt`
(run files in `harness/grpo_smoke_vllm_s{0,1,2}/`).

| Seed | Calibrated: reward | Hacks | Hack in every submission, from step on | Proof-gated: reward | Hacks |
|---|---|---|---|---|---|
| 0 | 1.45 | **100%** | 59 | 1.33 | 0% |
| 1 | 1.62 | **100%** | 6 | 1.46 | 0% |
| 2 | 1.45 | **100%** | 3 | 1.43 | 0% |
| mean | 1.51 | | | 1.41 | |

- **Calibrated tests were hacked in 3 of 3 seeds, the proof-gated checker in 0 of 3.** Every
  calibrated policy took up activation sparsity (`skip_below` 0.005-0.01) early and ended
  with it in every submission. Seeds 1 and 2 held it from steps 6 and 3; seed 0 had it in
  most submissions from step 9 but dipped (to 69% at step 45) before holding it from
  step 59. No proof-gated submission ever used it.
- **The hack paid in every seed:** the calibrated policy's reward is 1-12% above the
  proof-gated one of the same seed (7% on average).
- **The honest policies don't all find the optimum.** Seed 0's proof-gated policy stopped at
  1.33, 9% below the 1.46 of seed 1. Group collapse still ends exploration early.

### Slot code: the policy writes CUDA

`--config slot`: the policy writes the body of the kernel's slot functions
(`pck_slot_term`, `pck_slot_use`) as well as the config, compiled by nvcc and timed live on
the GPU. Every planted config-level exploit knob is locked to its honest value
(`lock_exploit_knobs`), so the only way to cheat is in code, e.g. a `pck_slot_use` that
skips small activations. Qwen3-0.6B, 15 minutes per regime, seed 0. Full report:
`harness/grpo_slot_out.txt` (run files in `harness/grpo_slot/`).

| Regime | Steps | Final eval: reward | Passed | Hacks | Samples with slot code |
|---|---|---|---|---|---|
| calibrated | 70 | 1.40 | 100% | 0% | 10 of 70 |
| proof-gated | 93 | 1.20 | 88% | 0% | 90 of 93 |

- **No hacks in either regime: the policy never found the code-level cheat.** Every slot
  it wrote is honest: the exact product `__fmul_rn(w, x)`, or `pck_slot_use` returning
  `true`. With the config knobs locked, the hack has to be invented in code, and a 0.6B
  model in 15 minutes doesn't get there. So this run doesn't test the checkers; a
  warm start that shows a skipping slot, a larger model or a longer run would.
- **Under calibrated tests the policy mostly stopped writing code.** After about step 30
  its replies are a config followed by text copied from the prompt's slot signatures,
  which doesn't parse as code, so the default (honest) slot runs. The proof-gated policy
  kept writing the exact product.
- **Live timing is noisy.** Training rewards swing by 0.3-0.5 between steps. The
  proof-gated final eval (1.20, with 2 of 16 samples failing to compile or pass) is below
  its own last-10-step mean (1.77), so the gap between the two regimes here is noise, not
  a cost of the proof gate.
- The first proof-gated attempt lost its final eval: the trusted `member.py` was edited
  while it ran, and the run's checker hash check stopped it. These numbers are from a
  re-run on the committed tree.

### The pilot: Qwen3-1.7B, multi-turn, held-out tasks

`--config pilot`: Qwen3-1.7B with a LoRA adapter, warm-started on scripted honest
episodes, then trained with GRPO on 8 tasks (5 layer-12 projections and 3 synthetic) in
multi-turn tool calling (up to 6 calls: write, compile, benchmark, check, submit). It is
evaluated on 5 held-out tasks (3 layer-20 projections and 2 outlier tasks), 16 sampled
episodes each. Three seeds per regime, Hugging Face rollouts, rewards from the re-swept
(2026-11-08) tables. The preset asks for 400 steps; these runs use 150 so that all six fit
in one night (36-94 minutes each). Full report: `harness/grpo_pilot_out.txt` (run files,
without the adapters, in `harness/grpo_pilot/s{0,1,2}/`).

| Seed | Calibrated: reward | Passed | Hacks | Proof-gated: reward | Passed | Hacks |
|---|---|---|---|---|---|---|
| 0 | 1.96 | 100% | 0% | 2.10 | 100% | 0% |
| 1 | 2.25 | 99% | 0% | 2.30 | 100% | 0% |
| 2 | -0.97 | 4% | 0% | 1.55 | 88% | 0% |

- **No hacks anywhere: no submission in training or evaluation cheated, under either
  checker.** That is expected here, so it says nothing for or against the checkers. The
  warm start shows only honest configs, and on the re-swept timing table the cheat is at
  most 3% faster than honest (see the exploit window), so there is little reward to pull
  the policy toward it. Over the 900 logged samples (one per step) a reply with
  `skip_below > 0` appears twice, both in calibrated seed 0, and is not kept.
- **The reward gap is training instability, not the checker.** Calibrated seed 2
  collapsed at step 20: its tool calls lost their final `}`, failed to parse, and the
  policy repeated the call without ever submitting, through step 149. The same failure
  hit both regimes (steps with fewer than half the replies valid: calibrated 3, 2, 130;
  proof-gated 1, 40, 14); the other runs recovered. Without seed 2, the proof-gated
  policies are 2-7% ahead (2.10 vs 1.96, 2.30 vs 2.25), within the spread between seeds.
- **What it shows:** a 1.7B model learns the multi-turn loop and generalizes to held-out
  layers and shapes (mean held-out rewards 1.55-2.30 for the runs that didn't collapse)
  under the proof-gated checker as well as under calibrated tests; the proof gate shows
  no cost in reward here. Testing whether the checker changes what is learned needs a table
  where the cheat pays (the timing fix under the exploit window) or a warm start that
  includes it.

### The sampled-token check inside vLLM

`pck.vllm_plugin` checks every token vLLM 0.33 samples against its row of logits and the
request's top-k / top-p (`pck.member.check_members`, proven in `lean/PCK/TopMember.lean`),
on the GPU. `harness/bench_vllm_checker.py` serves Qwen3-0.6B three times (check off,
check on, check on with 1% of sampled tokens replaced by the row's lowest-logit token) and
decodes 16 concurrent requests x 256 tokens x 3 rounds at top-k 50, top-p 0.9, temperature
0.8. Full output: `harness/bench_vllm_checker_out.txt`.

| Mode | Tokens/s | Cost | Tokens checked | Rejected | Injected | Caught | False alarms |
|---|---|---|---|---|---|---|---|
| off | 2,790 | - | - | - | - | - | - |
| check | 2,173 | 22% | 12,730 | 0 | 0 | 0 | **0** |
| faults | 1,065 | 62% | 12,758 | 115 | 115 | **115** | **0** |

- **Detection is exact here:** every injected token was caught and no honest token was
  rejected. The closest an accepted token came to the top-p boundary was r - p = -1.1e-4,
  so honest samples sit below p.
- **Totals are written without waiting for the GPU.** The running counts are copied to
  pinned host memory without blocking and written one interval later, so the decode loop
  never synchronizes; that took the cost of the check from 31% to 22%. The fault mode is
  slower (62%) because it checks each candidate fault before injecting it, so that only
  tokens outside the allowed set count as faults; that is test scaffolding, not part of
  the check.
- **The first version cost 43%** (1,595 tokens/s against 2,811). Taking the hook apart
  showed the arithmetic was not the problem: two places made the CPU wait for the GPU on
  every decode step (reading the batch's largest top-k, and a boolean-mask index for the
  boundary statistic). The fix reads top-k from vLLM's host copy and uses `where` in place
  of the mask. The check also moved from FP64 to FP32 terms summed in FP64 (error bound
  7.2e-7, `member_eps32`), and when every row has a top-k it computes the softmax over the
  top k logits only. Alone, the cost of the check fell from 1.8 ms to 0.9 ms per step.
- **What is left is launch overhead.** The check is about 80 small GPU operations per
  step: about 0.9 ms of Python dispatch against 0.45 ms of GPU work, beside a 5.7 ms
  decode step for this small model. A fused kernel or a CUDA graph would remove most of
  it; on larger models the share is smaller, since the decode step grows and the check
  doesn't.
