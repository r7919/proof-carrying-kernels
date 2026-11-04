// Trusted top-k / top-p sampling checker (GPU version of pck.sampling).
//
// One CUDA block per row of logits. For row b with selection sel[b, :len[b]]:
//   1. Insert every selected index into a shared-memory bitmap with atomicOr; an index
//      out of range or already present marks the selection malformed.
//   2. One pass over the vocabulary: exact FP32 min over selected logits, max over
//      rejected logits and over all logits, and a NaN flag.
//   3. top-p only: FP64 softmax masses of the selection and of the selection without its
//      smallest logit, using CUDA's exp (documented max error 1 ulp in FP64).
// top-k passes iff the selection is well formed, has length k, no logit is NaN, and
// min(selected) >= max(rejected). top-p additionally needs mass >= p - slack and
// mass_without_smallest < p + slack, where slack (from the host) covers the sampler's
// declared probability error plus this kernel's own FP64 rounding.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace pck {
namespace {

constexpr int kThreads = 1024;
constexpr int kWarp = 32;
constexpr unsigned kFullMask = 0xffffffffu;

struct MinMax {
  float min_sel;
  float max_rej;
  float max_all;
};

__device__ float warp_min(float v) {
  for (int o = kWarp / 2; o > 0; o /= 2) v = fminf(v, __shfl_down_sync(kFullMask, v, o));
  return v;
}
__device__ float warp_max(float v) {
  for (int o = kWarp / 2; o > 0; o /= 2) v = fmaxf(v, __shfl_down_sync(kFullMask, v, o));
  return v;
}
__device__ double warp_sum(double v) {
  for (int o = kWarp / 2; o > 0; o /= 2) v = __dadd_rn(v, __shfl_down_sync(kFullMask, v, o));
  return v;
}

// Block-wide reductions; the result is valid in every thread.
__device__ MinMax block_minmax(MinMax m, float* scratch) {
  const int lane = threadIdx.x % kWarp, warp = threadIdx.x / kWarp;
  m.min_sel = warp_min(m.min_sel);
  m.max_rej = warp_max(m.max_rej);
  m.max_all = warp_max(m.max_all);
  if (lane == 0) {
    scratch[3 * warp] = m.min_sel;
    scratch[3 * warp + 1] = m.max_rej;
    scratch[3 * warp + 2] = m.max_all;
  }
  __syncthreads();
  if (warp == 0) {
    const bool live = lane < kThreads / kWarp;
    m.min_sel = warp_min(live ? scratch[3 * lane] : INFINITY);
    m.max_rej = warp_max(live ? scratch[3 * lane + 1] : -INFINITY);
    m.max_all = warp_max(live ? scratch[3 * lane + 2] : -INFINITY);
    if (lane == 0) {
      scratch[0] = m.min_sel;
      scratch[1] = m.max_rej;
      scratch[2] = m.max_all;
    }
  }
  __syncthreads();
  MinMax out{scratch[0], scratch[1], scratch[2]};
  __syncthreads();
  return out;
}

__device__ double block_sum(double v, double* scratch) {
  const int lane = threadIdx.x % kWarp, warp = threadIdx.x / kWarp;
  v = warp_sum(v);
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = warp_sum(lane < kThreads / kWarp ? scratch[lane] : 0.0);
    if (lane == 0) scratch[0] = v;
  }
  __syncthreads();
  const double out = scratch[0];
  __syncthreads();
  return out;
}

__global__ void __launch_bounds__(kThreads)
    sampling_check_kernel(const float* __restrict__ logits, const int64_t* __restrict__ sel,
                          const int64_t* __restrict__ len, int64_t vocab, int64_t max_len,
                          int64_t k, bool top_p, double p, double temperature, double slack,
                          bool* __restrict__ passed, float* __restrict__ min_sel_out,
                          float* __restrict__ max_rej_out, double* __restrict__ mass_out,
                          double* __restrict__ without_out) {
  extern __shared__ unsigned int bitmap[];
  __shared__ float fscratch[3 * kThreads / kWarp];
  __shared__ double dscratch[kThreads / kWarp];
  __shared__ int malformed;
  __shared__ int has_nan;

  const int t = static_cast<int>(threadIdx.x);
  const int64_t row = blockIdx.x;
  const float* l = logits + row * vocab;
  const int64_t* s = sel + row * max_len;
  const int64_t n_sel = len[row];
  const int64_t words = (vocab + 31) / 32;

  for (int64_t w = t; w < words; w += kThreads) bitmap[w] = 0u;
  if (t == 0) {
    malformed = (n_sel < 1 || n_sel > max_len || n_sel > vocab || (!top_p && n_sel != k)) ? 1 : 0;
    has_nan = 0;
  }
  __syncthreads();

  // 1. Selection bitmap; duplicates and out-of-range indices are malformed.
  for (int64_t i = t; i < n_sel && i < max_len; i += kThreads) {
    const int64_t idx = s[i];
    if (idx < 0 || idx >= vocab) {
      atomicOr(&malformed, 1);
    } else {
      const unsigned bit = 1u << (idx & 31);
      if (atomicOr(&bitmap[idx >> 5], bit) & bit) atomicOr(&malformed, 1);
    }
  }
  __syncthreads();

  // 2. Exact min over selected, max over rejected and over all logits.
  MinMax m{INFINITY, -INFINITY, -INFINITY};
  bool nan_seen = false;
  for (int64_t j = t; j < vocab; j += kThreads) {
    const float v = l[j];
    nan_seen |= isnan(v);
    m.max_all = fmaxf(m.max_all, v);
    if (bitmap[j >> 5] & (1u << (j & 31))) {
      m.min_sel = fminf(m.min_sel, v);
    } else {
      m.max_rej = fmaxf(m.max_rej, v);
    }
  }
  if (nan_seen) atomicOr(&has_nan, 1);
  m = block_minmax(m, fscratch);
  bool ok = malformed == 0 && has_nan == 0 && m.min_sel >= m.max_rej;  // -inf if none rejected

  // 3. top-p: FP64 softmax masses.
  double mass = 0.0, without = 0.0;
  if (top_p && ok) {
    const double mx = static_cast<double>(m.max_all);
    double z = 0.0, z_sel = 0.0;
    for (int64_t j = t; j < vocab; j += kThreads) {
      const double e = exp((static_cast<double>(l[j]) - mx) / temperature);
      z = __dadd_rn(z, e);
      if (bitmap[j >> 5] & (1u << (j & 31))) z_sel = __dadd_rn(z_sel, e);
    }
    z = block_sum(z, dscratch);
    z_sel = block_sum(z_sel, dscratch);
    const double e_min = exp((static_cast<double>(m.min_sel) - mx) / temperature);
    mass = z_sel / z;
    without = (z_sel - e_min) / z;
    ok = mass >= p - slack && without < p + slack;
  }

  if (t == 0) {
    passed[row] = ok;
    min_sel_out[row] = m.min_sel;
    max_rej_out[row] = m.max_rej;
    mass_out[row] = mass;
    without_out[row] = without;
  }
}

}  // namespace

constexpr int64_t kMaxVocab = 48 * 1024 * 8;  // bitmap must fit in 48 KB of shared memory

cudaError_t sampling_check(const float* logits, const int64_t* sel, const int64_t* len,
                           int64_t rows, int64_t vocab, int64_t max_len, int64_t k, bool top_p,
                           double p, double temperature, double slack, bool* passed,
                           float* min_sel, float* max_rej, double* mass, double* without,
                           cudaStream_t stream) {
  if (vocab < 1 || vocab > kMaxVocab || rows < 1) return cudaErrorInvalidValue;
  const size_t shared = static_cast<size_t>((vocab + 31) / 32) * sizeof(unsigned int);
  sampling_check_kernel<<<static_cast<unsigned int>(rows), kThreads, shared, stream>>>(
      logits, sel, len, vocab, max_len, k, top_p, p, temperature, slack, passed, min_sel,
      max_rej, mass, without);
  return cudaGetLastError();
}

}  // namespace pck
