# Results

All GPU numbers are from one RTX 5060 Laptop GPU (Blackwell, 8 GB, WSL2, driver 617.42,
CUDA 12.8, PyTorch 2.11) on a cold L2 unless noted. Each section names the script that
produced it and the file with its full output, so every number here can be reproduced.

| Result | Script | Output |
|---|---|---|
| Numerics study (CPU) | `harness/numerics_sim.py` | `harness/sim_out.txt` |
| GEMV kernels vs cuBLAS | `harness/bench_gemv.py` | `harness/bench_gemv_out.txt` |
| Checker overhead, decode layer | `harness/bench_decode_layer.py` | `harness/bench_decode_layer_out.txt` |
| False alarms and bit flips | `harness/campaign.py` | `harness/campaign_out.txt` |
| Sampling checker cost | `harness/bench_sampling.py` | `harness/bench_sampling_out.txt` |
| Tightness, synthetic activations (CPU) | `harness/tightness.py` | `harness/tightness_out.txt` |
| Tightness, real activations | `python -m harness.real_activations` | `harness/real_activations_out.txt` |
| Agent template sweep | `harness/sweep_template.py` | `harness/sweep_template_out.txt`, `harness/timing_table.json` |
| BF16 shortcut at batch 1-8 | `harness/bench_batched.py` | `harness/bench_batched_out.txt` |
| Exploit window, activation sparsity | `python -m harness.exploit_window` | `harness/exploit_window_out.txt` |

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

### Checker overhead on a decode layer

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
| 1 | 354 us | 25 us | 253 us | 65 us |
| 8 | 366 us | 39 us | 812 us | 153 us |

Computing the top-p terms with FP32 `expf` on a split argument instead of FP64 `exp` cut
the top-p checker from 91 to 65 us at batch 1 and from 311 to 153 us at batch 8 (FP64 runs
at 1/64 rate on this GPU), for about 1e-6 more slack.

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

`harness/sweep_template.py` times all 2,112 configs of the agent's GEMV template
(`src/pck/agent/config.py`) on the 15 task shapes, against cuBLAS in the same session. The
table it writes serves every episode's timings and is the brute-force baseline. Full
output: `harness/sweep_template_out.txt`.

- **In-spec variants beat cuBLAS on every shape**, by 1.05-1.07x on the large MLP shapes
  (bandwidth-bound for both) up to 2-3x on small ones, where cuBLAS's launch configuration
  leaves the GPU underused.
- **The precision exploits are not faster.** BF16 accumulators, partials or output give no
  speedup over the best in-spec variant on any shape (14336 x 4096: 1.05x either way);
  rounding to BF16 adds instructions and saves no memory traffic in a bandwidth-bound
  kernel. Every exploit speedup in the table comes from `skip_tail`, i.e. from not reading
  part of `W`.
- So under the current template, a policy has no speed incentive to use a precision
  shortcut, and the one exploit that pays (dropping work) is locked in training. The
  calibrated-vs-proof-gated comparison needs an exploit that is both faster and passes
  `atol = rtol = 1e-2`.

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

`python -m harness.exploit_window` runs the column-major template with activation sparsity
(skip every column with |x_j| below a threshold, so its weights are never read) on the 14
real Qwen2.5-0.5B layers, over every captured decode input episodes may see, and combines
the verdicts with the timing table. Full output: `harness/exploit_window_out.txt`.

| Layer | Threshold | Skipped | Calibrated accepts | Proof-gated accepts | vs honest column-major | vs honest row-major |
|---|---|---|---|---|---|---|
| l12 down_proj | 0.005 | 9% | 100% | 0% | 1.19x | 0.95x |
| l20 down_proj | 0.01 | 6% | 100% | 0% | 1.15x | 0.93x |
| l20 o_proj | 0.01 | 7% | 100% | 0% | 1.12x | 0.82x |

- **Accuracy: the window exists.** On every real layer some threshold skips 3-16% of the
  columns, passes the calibrated test on at least 95% of inputs and fails the proof-gated
  checker on every input. Larger thresholds are caught by both.
- **Speed: it pays only within the column-major layout.** Against the fastest honest
  column-major kernel the accepted cheat is 1.06-1.19x faster on the MLP and `o_proj`
  layers. Against the fastest honest kernel of either layout it never pays: the row-major
  kernel is 15-25% faster than the honest column-major one (its threads read long
  contiguous runs of one row), more than skipping saves.
- So a speed incentive to cheat exists when weights are stored transposed, as
  sparsity-aware inference engines (TEAL, Deja Vu) do to make skipping pay, and not when
  they are stored row-major.
