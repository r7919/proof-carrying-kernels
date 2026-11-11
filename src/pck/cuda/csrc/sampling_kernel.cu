// Trusted top-k / top-p sampling checker (GPU version of pck.sampling).
//
// For row b with selection sel[b, :len[b]] over a vocabulary of V logits:
//   kernel 1 (one thread per selected entry): insert each index into a zeroed bitmap with
//            atomicOr; an index out of range or already present marks the row malformed.
//   kernel 2 (rows x segments blocks): each segment computes the exact FP32 min over its
//            selected logits, max over its rejected logits and its own max m_s, a NaN flag,
//            and (top-p) z_s = sum exp((l - m_s) / T) over all and over selected logits,
//            summed in FP64. Each term uses FP32 expf (max error 2 ulp), since consumer
//            GPUs run FP64 at 1/64 rate: the argument d = (l - m_s) / T is formed in FP64,
//            split as d = hi + lo with hi = fp32(d), and e = expf(hi) * (1 + lo). The split
//            keeps the error relative to e at about 6 u32 for any |d|; an FP32 argument
//            alone would add |d| u32, up to 100x more. The last segment of the row to finish
//            (atomic ticket) merges in a fixed order: M = max m_s,
//            Z = sum z_s * exp((m_s - M) / T), and decides.
// top-k passes iff the selection is well formed with length k, no logit is NaN, and
// min(selected) >= max(rejected). top-p additionally needs mass >= p - slack and
// mass_without_smallest < p + slack; slack (from the host) covers the sampler's declared
// probability error plus this kernel's own FP64 rounding, including the merge.
//
// Scratch (bitmap and flags) must be zero on entry; the host allocates it zeroed per call.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace pck {
namespace {

constexpr int kThreads = 256;
constexpr int kWarp = 32;
constexpr int kWarps = kThreads / kWarp;
constexpr unsigned kFullMask = 0xffffffffu;
// Per-row flag words after the bitmap: malformed, NaN seen, finished-segment ticket.
constexpr int kMalformed = 0, kNan = 1, kTicket = 2, kFlags = 3;

__global__ void insert_kernel(const int64_t* __restrict__ sel, const int64_t* __restrict__ len,
                              int64_t vocab, int64_t max_len, int64_t words,
                              unsigned int* __restrict__ scratch) {
  const int64_t row = blockIdx.y;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t n = min(len[row], max_len);
  if (i >= n) return;
  unsigned int* bitmap = scratch + row * (words + kFlags);
  unsigned int* flags = bitmap + words;
  const int64_t idx = sel[row * max_len + i];
  if (idx < 0 || idx >= vocab) {
    atomicOr(&flags[kMalformed], 1u);
    return;
  }
  const unsigned bit = 1u << (idx & 31);
  if (atomicOr(&bitmap[idx >> 5], bit) & bit) atomicOr(&flags[kMalformed], 1u);
}

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

// Block-wide reductions; results are valid in every thread.
template <int kN>
__device__ void block_minmax(float (&v)[kN], const bool (&is_min)[kN], float* scratch) {
  const int lane = threadIdx.x % kWarp, warp = threadIdx.x / kWarp;
  for (int q = 0; q < kN; ++q) {
    v[q] = is_min[q] ? warp_min(v[q]) : warp_max(v[q]);
    if (lane == 0) scratch[q * kWarps + warp] = v[q];
  }
  __syncthreads();
  for (int q = 0; q < kN; ++q) {
    float r = is_min[q] ? INFINITY : -INFINITY;
    for (int w = 0; w < kWarps; ++w) {
      r = is_min[q] ? fminf(r, scratch[q * kWarps + w]) : fmaxf(r, scratch[q * kWarps + w]);
    }
    v[q] = r;
  }
  __syncthreads();
}

__device__ double block_sum(double v, double* scratch) {
  const int lane = threadIdx.x % kWarp, warp = threadIdx.x / kWarp;
  v = warp_sum(v);
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  double r = 0.0;
  for (int w = 0; w < kWarps; ++w) r = __dadd_rn(r, scratch[w]);  // same order in every thread
  __syncthreads();
  return r;
}

struct Partial {
  float min_sel, max_rej, max_all;
  double z, z_sel;
};

__global__ void __launch_bounds__(kThreads)
    segment_kernel(const float* __restrict__ logits, const int64_t* __restrict__ len, int64_t vocab,
                   int64_t max_len, int64_t words, int64_t seg_len, int64_t k, bool top_p, double p,
                   double temperature, double slack, unsigned int* __restrict__ scratch,
                   Partial* __restrict__ partials, bool* __restrict__ passed,
                   float* __restrict__ min_sel_out, float* __restrict__ max_rej_out,
                   double* __restrict__ mass_out, double* __restrict__ without_out) {
  __shared__ float fscratch[3 * kWarps];
  __shared__ double dscratch[kWarps];
  __shared__ bool is_last;
  const int t = static_cast<int>(threadIdx.x);
  const int64_t row = blockIdx.y, seg = blockIdx.x, segments = gridDim.x;
  const float* l = logits + row * vocab;
  const unsigned int* bitmap = scratch + row * (words + kFlags);
  unsigned int* flags = scratch + row * (words + kFlags) + words;
  const int64_t j0 = seg * seg_len, j1 = min(vocab, j0 + seg_len);

  // Exact min over selected, max over rejected and over all logits in this segment.
  float v[3] = {INFINITY, -INFINITY, -INFINITY};
  const bool is_min[3] = {true, false, false};
  bool nan_seen = false;
  for (int64_t j = j0 + t; j < j1; j += kThreads) {
    const float x = l[j];
    nan_seen |= isnan(x);
    v[2] = fmaxf(v[2], x);
    if (bitmap[j >> 5] & (1u << (j & 31))) {
      v[0] = fminf(v[0], x);
    } else {
      v[1] = fmaxf(v[1], x);
    }
  }
  if (nan_seen) atomicOr(&flags[kNan], 1u);
  block_minmax(v, is_min, fscratch);

  // top-p: softmax partial sums relative to this segment's own max.
  double z = 0.0, z_sel = 0.0;
  if (top_p && v[2] > -INFINITY) {
    const double m = static_cast<double>(v[2]);
    const double inv_t = 1.0 / temperature;
    for (int64_t j = j0 + t; j < j1; j += kThreads) {
      const double d = __dmul_rn(static_cast<double>(l[j]) - m, inv_t);
      const float hi = __double2float_rn(d);
      const float lo = __double2float_rn(d - static_cast<double>(hi));  // |lo| <= u32 |d|
      const float eh = expf(hi);
      // Terms with d < -128 (masked -inf logits, or tiny temperatures) are below e^-128,
      // within the host's V 2^-148 absolute allowance; dropping them also keeps -inf from
      // turning into NaN (hi = -inf makes lo NaN). A NaN logit is flagged above.
      const double e = d >= -128.0 ? static_cast<double>(__fmaf_rn(eh, lo, eh)) : 0.0;
      z = __dadd_rn(z, e);
      if (bitmap[j >> 5] & (1u << (j & 31))) z_sel = __dadd_rn(z_sel, e);
    }
  }
  if (top_p) {
    z = block_sum(z, dscratch);
    z_sel = block_sum(z_sel, dscratch);
  }

  if (t == 0) {
    partials[row * segments + seg] = Partial{v[0], v[1], v[2], z, z_sel};
    __threadfence();  // publish the partial before taking a ticket
    is_last = atomicAdd(&flags[kTicket], 1u) == static_cast<unsigned>(segments) - 1;
  }
  __syncthreads();
  if (!is_last || t != 0) return;

  // Last segment of the row: merge in a fixed order and decide.
  __threadfence();
  const volatile Partial* ps = partials + row * segments;
  float min_sel = INFINITY, max_rej = -INFINITY, mx = -INFINITY;
  for (int64_t s = 0; s < segments; ++s) {
    min_sel = fminf(min_sel, ps[s].min_sel);
    max_rej = fmaxf(max_rej, ps[s].max_rej);
    mx = fmaxf(mx, ps[s].max_all);
  }
  const int64_t n_sel = len[row];
  const bool well_formed = flags[kMalformed] == 0 && n_sel >= 1 && n_sel <= max_len &&
                           n_sel <= vocab && (top_p || n_sel == k);
  bool ok = well_formed && flags[kNan] == 0 && min_sel >= max_rej;  // -inf if none rejected
  double mass = 0.0, without = 0.0;
  if (top_p && ok) {
    const double big_m = static_cast<double>(mx);
    double big_z = 0.0, big_z_sel = 0.0;
    for (int64_t s = 0; s < segments; ++s) {
      if (ps[s].max_all > -INFINITY) {
        const double scale = exp((static_cast<double>(ps[s].max_all) - big_m) / temperature);
        big_z = __dadd_rn(big_z, ps[s].z * scale);
        big_z_sel = __dadd_rn(big_z_sel, ps[s].z_sel * scale);
      }
    }
    const double e_min = exp((static_cast<double>(min_sel) - big_m) / temperature);
    mass = big_z_sel / big_z;
    without = (big_z_sel - e_min) / big_z;
    ok = mass >= p - slack && without < p + slack;
  }
  passed[row] = ok;
  min_sel_out[row] = min_sel;
  max_rej_out[row] = max_rej;
  mass_out[row] = mass;
  without_out[row] = without;
}

}  // namespace

int64_t sampling_scratch_words(int64_t vocab) { return (vocab + 31) / 32 + kFlags; }

int64_t sampling_partial_bytes() { return static_cast<int64_t>(sizeof(Partial)); }

cudaError_t sampling_check(const float* logits, const int64_t* sel, const int64_t* len,
                           int64_t rows, int64_t vocab, int64_t max_len, int64_t k, bool top_p,
                           double p, double temperature, double slack, int64_t segments,
                           unsigned int* scratch, void* partials, bool* passed, float* min_sel,
                           float* max_rej, double* mass, double* without, cudaStream_t stream) {
  if (vocab < 1 || rows < 1 || max_len < 1 || segments < 1 || segments > vocab) {
    return cudaErrorInvalidValue;
  }
  const int64_t words = (vocab + 31) / 32;
  const dim3 insert_grid(static_cast<unsigned int>((max_len + kThreads - 1) / kThreads),
                         static_cast<unsigned int>(rows));
  insert_kernel<<<insert_grid, kThreads, 0, stream>>>(sel, len, vocab, max_len, words, scratch);
  const int64_t seg_len = (vocab + segments - 1) / segments;
  const dim3 grid(static_cast<unsigned int>(segments), static_cast<unsigned int>(rows));
  segment_kernel<<<grid, kThreads, 0, stream>>>(
      logits, len, vocab, max_len, words, seg_len, k, top_p, p, temperature, slack, scratch,
      static_cast<Partial*>(partials), passed, min_sel, max_rej, mass, without);
  return cudaGetLastError();
}

}  // namespace pck
