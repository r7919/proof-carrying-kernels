// Diagnostic variants of the checker's phase-1 loop (NOT trusted code, timing only).
//
// Same grid (K * nb, segments), 256 threads, segment layout and block reductions as
// src/pck/cuda/csrc/checker_kernel.cu; only the per-element arithmetic differs:
//   0  loads only (FP32 sum of c + a + x, to keep the loads live)
//   1  production: FP64 FMA for c . x, FP32 round-up for a . |x|
//   2  FP32 FMA for c . x (unsound), FP32 round-up for a . |x|
//   3  compensated FP32 for c . x (TwoProduct + TwoSum), FP32 round-up for a . |x|
//   4  FP64 FMA for c . x only (no a . |x|)
// Each block writes one double per output so the compiler cannot drop the work.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace pckv {
namespace {

constexpr int kThreads = 256;
constexpr int kWarp = 32;
constexpr unsigned kFullMask = 0xffffffffu;

template <bool kRoundUp>
__device__ double block_sum(double v, double* shared) {
  const int lane = static_cast<int>(threadIdx.x) % kWarp;
  const int warp = static_cast<int>(threadIdx.x) / kWarp;
  for (int offset = kWarp / 2; offset > 0; offset /= 2) {
    const double o = __shfl_down_sync(kFullMask, v, offset);
    v = kRoundUp ? __dadd_ru(v, o) : __dadd_rn(v, o);
  }
  if (lane == 0) {
    shared[warp] = v;
  }
  __syncthreads();
  double total = 0.0;
  if (warp == 0) {
    total = lane < kThreads / kWarp ? shared[lane] : 0.0;
    for (int offset = kWarp / 2; offset > 0; offset /= 2) {
      const double o = __shfl_down_sync(kFullMask, total, offset);
      total = kRoundUp ? __dadd_ru(total, o) : __dadd_rn(total, o);
    }
  }
  __syncthreads();
  return total;
}

template <int kVariant>
__global__ void __launch_bounds__(kThreads)
    phase1(const float* __restrict__ c, const __nv_bfloat16* __restrict__ a,
           const __nv_bfloat16* __restrict__ x, int64_t n, int64_t seg_len, int num_blocks,
           double* __restrict__ out) {
  __shared__ double shared[kWarp];
  const int t = static_cast<int>(threadIdx.x);
  const int64_t check_block = blockIdx.x;
  const int64_t seg = blockIdx.y;
  const int64_t beta = check_block % num_blocks;
  const float* c_row = c + check_block * n;
  const __nv_bfloat16* a_row = a + beta * n;
  const int64_t j_end = min(n, (seg + 1) * seg_len);

  double cx64 = 0.0;
  float cx32 = 0.0f, lo = 0.0f, s = 0.0f;
#pragma unroll 4
  for (int64_t j = seg * seg_len + t; j < j_end; j += kThreads) {
    const float xj = __bfloat162float(x[j]);
    const float cj = c_row[j];
    if constexpr (kVariant == 0) {
      cx32 = __fadd_rn(cx32, __fadd_rn(cj, __fadd_rn(__bfloat162float(a_row[j]), xj)));
    } else if constexpr (kVariant == 1 || kVariant == 4) {
      cx64 = __fma_rn(static_cast<double>(cj), static_cast<double>(xj), cx64);
    } else if constexpr (kVariant == 2) {
      cx32 = __fmaf_rn(cj, xj, cx32);
    } else if constexpr (kVariant == 3) {
      const float p = __fmul_rn(cj, xj);
      const float e = __fmaf_rn(cj, xj, -p);  // exact: c * x = p + e
      const float sum = __fadd_rn(cx32, p);
      const float z = __fsub_rn(sum, cx32);
      const float err = __fadd_rn(__fsub_rn(cx32, __fsub_rn(sum, z)), __fsub_rn(p, z));
      cx32 = sum;
      lo = __fadd_rn(lo, __fadd_rn(err, e));
    }
    if constexpr (kVariant >= 1 && kVariant <= 3) {
      s = __fadd_ru(s, __fmul_ru(__bfloat162float(a_row[j]), fabsf(xj)));
    }
  }
  double cx;
  if constexpr (kVariant == 1 || kVariant == 4) {
    cx = cx64;
  } else if constexpr (kVariant == 3) {
    cx = static_cast<double>(cx32) + static_cast<double>(lo);
  } else {
    cx = static_cast<double>(cx32);
  }
  cx = block_sum<false>(cx, shared);
  const double s_up = block_sum<true>(static_cast<double>(s), shared);
  if (t == 0) {
    out[check_block * gridDim.y + seg] = cx + s_up;
  }
}

}  // namespace

cudaError_t run_variant(int variant, const float* c, const void* a, const void* x, int64_t n,
                        int num_blocks, int checks, int segments, double* out,
                        cudaStream_t stream) {
  const int64_t seg_len = (n + segments - 1) / segments;
  const dim3 grid(static_cast<unsigned int>(checks) * num_blocks,
                  static_cast<unsigned int>(segments));
  const auto* ab = static_cast<const __nv_bfloat16*>(a);
  const auto* xb = static_cast<const __nv_bfloat16*>(x);
  switch (variant) {
    case 0:
      phase1<0><<<grid, kThreads, 0, stream>>>(c, ab, xb, n, seg_len, num_blocks, out);
      break;
    case 1:
      phase1<1><<<grid, kThreads, 0, stream>>>(c, ab, xb, n, seg_len, num_blocks, out);
      break;
    case 2:
      phase1<2><<<grid, kThreads, 0, stream>>>(c, ab, xb, n, seg_len, num_blocks, out);
      break;
    case 3:
      phase1<3><<<grid, kThreads, 0, stream>>>(c, ab, xb, n, seg_len, num_blocks, out);
      break;
    case 4:
      phase1<4><<<grid, kThreads, 0, stream>>>(c, ab, xb, n, seg_len, num_blocks, out);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

}  // namespace pckv
