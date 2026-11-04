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
| 1 | 386 us | 25 us | 256 us | 91 us |
| 8 | 363 us | 36 us | 814 us | 311 us |

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
every linear layer (4 prompts x 16 tokens, 8 inputs kept per layer), through the bit-exact
model of the coalesced kernel. Full output: `harness/real_activations_out.txt`.

| Layer | Row bound / error, median (min) | Block tau / \|rho\|, b = 64, median (min) | b = 128, median (min) |
|---|---|---|---|
| q_proj | 293x (13x) | 2057x (223x) | 2653x (262x) |
| k_proj | 251x (18x) | 1554x (306x) | 2409x (327x) |
| v_proj | 311x (26x) | 2266x (363x) | 3171x (772x) |
| o_proj | 348x (15x) | 2406x (242x) | 3439x (539x) |
| gate_proj | 293x (10x) | 1982x (271x) | 2868x (437x) |
| up_proj | 372x (15x) | 2847x (379x) | 4098x (668x) |
| down_proj | 1059x (39x) | 8053x (661x) | 10801x (810x) |

- No proven-bound violations. Real activations come closer to the bound than any synthetic
  case (minimum 10x, in `gate_proj`, against 25x for synthetic outliers), but never cross it.
- Block tightness matches the synthetic study: 1,500-10,800x (median), far looser than
  V-ABFT, as expected for a threshold that may never false-alarm.
- The Gaussian-calibrated tolerance raised **no** false alarms on this model's decode
  activations. That is a negative result for the "why not tests" argument here, with a
  caveat: the calibration was re-tuned to each input's own RMS, which is generous. A fixed
  per-layer tolerance, as a test harness would use, is the fairer comparison and is still to
  be run; the synthetic mixed-scale case shows the failure mode it guards against.
