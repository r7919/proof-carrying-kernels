// Diagnostic batched GEMV variants (timing only, not trusted code): Y = X W^T for a batch
// of B activation vectors, BF16 W and X, one block per row of W, coalesced 16-byte loads.
// The variants differ only in the inner multiply-accumulate:
//   0  FP32 multiply then add (__fmul_rn, __fadd_rn), as in the agent's template
//   1  FP32 FMA (bitwise identical to 0: a BF16 x BF16 product is exact in FP32)
//   2  packed BF16 FMA (__hfma2 on bfloat162 accumulators), the precision shortcut
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>

namespace pckb {
namespace {

constexpr int kVec = 8;
constexpr int kWarp = 32;
constexpr unsigned kFullMask = 0xffffffffu;

__device__ __forceinline__ void load8(const __nv_bfloat16* p, __nv_bfloat162 (&v)[4]) {
  const uint4 raw = *reinterpret_cast<const uint4*>(p);
  memcpy(v, &raw, sizeof(raw));
}

template <int B, int MODE>
__global__ void batched_kernel(const __nv_bfloat16* __restrict__ W,
                               const __nv_bfloat16* __restrict__ X, float* __restrict__ Y,
                               int64_t m, int64_t n) {
  __shared__ float warp_sums[B][kWarp];
  const int t = static_cast<int>(threadIdx.x);
  const int threads = static_cast<int>(blockDim.x);
  const __nv_bfloat16* w_row = W + static_cast<int64_t>(blockIdx.x) * n;
  float acc[B];
  __nv_bfloat162 acc2[B];
#pragma unroll
  for (int b = 0; b < B; ++b) {
    acc[b] = 0.0f;
    acc2[b] = __floats2bfloat162_rn(0.0f, 0.0f);
  }
  for (int64_t base = static_cast<int64_t>(t) * kVec; base < n;
       base += static_cast<int64_t>(threads) * kVec) {
    __nv_bfloat162 w2[4];
    load8(w_row + base, w2);
#pragma unroll
    for (int b = 0; b < B; ++b) {
      __nv_bfloat162 x2[4];
      load8(X + b * n + base, x2);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        if (MODE == 2) {
          acc2[b] = __hfma2(w2[k], x2[k], acc2[b]);
        } else {
          const float2 wf = __bfloat1622float2(w2[k]);
          const float2 xf = __bfloat1622float2(x2[k]);
          if (MODE == 0) {
            acc[b] = __fadd_rn(acc[b], __fmul_rn(wf.x, xf.x));
            acc[b] = __fadd_rn(acc[b], __fmul_rn(wf.y, xf.y));
          } else {
            acc[b] = __fmaf_rn(wf.x, xf.x, acc[b]);
            acc[b] = __fmaf_rn(wf.y, xf.y, acc[b]);
          }
        }
      }
    }
  }
  const int lane = t % kWarp, warp = t / kWarp, warps = threads / kWarp;
#pragma unroll
  for (int b = 0; b < B; ++b) {
    float v = MODE == 2 ? __low2float(acc2[b]) + __high2float(acc2[b]) : acc[b];
    for (int off = kWarp / 2; off > 0; off /= 2) v += __shfl_down_sync(kFullMask, v, off);
    if (lane == 0) warp_sums[b][warp] = v;
  }
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int b = 0; b < B; ++b) {
      float s = lane < warps ? warp_sums[b][lane] : 0.0f;
      for (int off = kWarp / 2; off > 0; off /= 2) s += __shfl_down_sync(kFullMask, s, off);
      if (lane == 0) Y[b * m + blockIdx.x] = s;
    }
  }
}

template <int B>
void launch_b(int mode, const __nv_bfloat16* W, const __nv_bfloat16* X, float* Y, int64_t m,
              int64_t n, int threads, cudaStream_t s) {
  const auto grid = static_cast<unsigned>(m);
  if (mode == 0) batched_kernel<B, 0><<<grid, threads, 0, s>>>(W, X, Y, m, n);
  if (mode == 1) batched_kernel<B, 1><<<grid, threads, 0, s>>>(W, X, Y, m, n);
  if (mode == 2) batched_kernel<B, 2><<<grid, threads, 0, s>>>(W, X, Y, m, n);
}

}  // namespace

cudaError_t batched_gemv(int mode, const void* W, const void* X, float* Y, int64_t m, int64_t n,
                         int64_t batch, int threads, cudaStream_t s) {
  const auto* w = static_cast<const __nv_bfloat16*>(W);
  const auto* x = static_cast<const __nv_bfloat16*>(X);
  if (mode < 0 || mode > 2 || n % kVec != 0) return cudaErrorInvalidValue;
  switch (batch) {
    case 1:
      launch_b<1>(mode, w, x, Y, m, n, threads, s);
      break;
    case 2:
      launch_b<2>(mode, w, x, Y, m, n, threads, s);
      break;
    case 4:
      launch_b<4>(mode, w, x, Y, m, n, threads, s);
      break;
    case 8:
      launch_b<8>(mode, w, x, Y, m, n, threads, s);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

}  // namespace pckb
