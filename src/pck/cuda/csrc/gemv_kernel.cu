// GEMV kernels: y = W x with BF16 (or FP8 E4M3) weights, BF16 x, FP32 accumulation and
// FP32 output.
//
// Each kernel matches a CPU model in pck.gemv bit for bit, so the summation tree
// (and hence the proven error bound) is known exactly.
//
// Naive kernel (pck.gemv.emulate_gemv): one block per output row, `threads` threads
// per block. Thread t sums the contiguous chunk [t * chunk, (t + 1) * chunk)
// sequentially, treating columns past n as +0, then a pairwise tree combines the
// threads. Tree height: (chunk - 1) + log2(threads).
//
// Explicit round-to-nearest intrinsics keep the compiler from contracting or
// reordering operations. (A BF16 x BF16 product is exact in FP32, so an FMA
// would give the same result, but the intrinsics make that independent of
// compiler flags.)

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>

namespace pck {
namespace {

__global__ void gemv_bf16_naive_kernel(const __nv_bfloat16* __restrict__ W,
                                       const __nv_bfloat16* __restrict__ x, float* __restrict__ y,
                                       int64_t n, int64_t chunk) {
  extern __shared__ float partial[];
  const int t = static_cast<int>(threadIdx.x);
  const __nv_bfloat16* w_row = W + static_cast<int64_t>(blockIdx.x) * n;
  const int64_t begin = t * chunk;

  auto product = [&](int64_t j) {
    return __fmul_rn(__bfloat162float(w_row[j]), __bfloat162float(x[j]));
  };

  float acc = begin < n ? product(begin) : 0.0f;
  for (int64_t j = begin + 1; j < begin + chunk; ++j) {
    acc = __fadd_rn(acc, j < n ? product(j) : 0.0f);
  }
  partial[t] = acc;
  __syncthreads();

  for (int stride = 1; stride < static_cast<int>(blockDim.x); stride *= 2) {
    if (t % (2 * stride) == 0) {
      partial[t] = __fadd_rn(partial[t], partial[t + stride]);
    }
    __syncthreads();
  }
  if (t == 0) {
    y[blockIdx.x] = partial[0];
  }
}

// Coalesced kernel (pck.gemv.emulate_gemv_coalesced): one block per output row.
// Columns are grouped into 16-byte vectors of 8 BF16 values. Thread t owns vectors
// t, t + T, t + 2T, ..., so a warp reads 512 contiguous bytes per step. Each thread
// sums its products sequentially (vector by vector, columns past n as +0), each warp
// reduces by __shfl_down_sync halving, and warp 0 reduces the warp sums the same way.
constexpr int kVec = 8;
constexpr int kWarp = 32;
constexpr unsigned kFullMask = 0xffffffffu;

__device__ __forceinline__ void products(const __nv_bfloat16* __restrict__ w,
                                         const __nv_bfloat16* __restrict__ x, int64_t base,
                                         int64_t n, bool vector_loads, float (&p)[kVec]) {
  if (vector_loads && base + kVec <= n) {
    const uint4 wv = *reinterpret_cast<const uint4*>(w + base);
    const uint4 xv = *reinterpret_cast<const uint4*>(x + base);
    __nv_bfloat162 w2[kVec / 2];
    __nv_bfloat162 x2[kVec / 2];
    memcpy(w2, &wv, sizeof(wv));
    memcpy(x2, &xv, sizeof(xv));
#pragma unroll
    for (int k = 0; k < kVec / 2; ++k) {
      const float2 wf = __bfloat1622float2(w2[k]);
      const float2 xf = __bfloat1622float2(x2[k]);
      p[2 * k] = __fmul_rn(wf.x, xf.x);
      p[2 * k + 1] = __fmul_rn(wf.y, xf.y);
    }
  } else {
#pragma unroll
    for (int k = 0; k < kVec; ++k) {
      const int64_t j = base + k;
      p[k] = j < n ? __fmul_rn(__bfloat162float(w[j]), __bfloat162float(x[j])) : 0.0f;
    }
  }
}

// FP8 (E4M3) weights: 16 per 16-byte load, with two 16-byte loads of x. An E4M3 value has
// 4 significand bits and a BF16 value 8, so each product is exact in FP32 and the
// summation tree is the BF16 kernel's with vectors of 16.
constexpr int kVecFp8 = 16;

__device__ __forceinline__ float fp8_to_float(uint8_t bits) {
  __nv_fp8_e4m3 v;
  v.__x = bits;
  return static_cast<float>(v);
}

__device__ __forceinline__ void products(const uint8_t* __restrict__ w,
                                         const __nv_bfloat16* __restrict__ x, int64_t base,
                                         int64_t n, bool vector_loads, float (&p)[kVecFp8]) {
  if (vector_loads && base + kVecFp8 <= n) {
    const uint4 wv = *reinterpret_cast<const uint4*>(w + base);
    const uint4 xa = *reinterpret_cast<const uint4*>(x + base);
    const uint4 xb = *reinterpret_cast<const uint4*>(x + base + kVecFp8 / 2);
    __nv_fp8x2_e4m3 w2[kVecFp8 / 2];
    __nv_bfloat162 x2[kVecFp8 / 2];
    memcpy(w2, &wv, sizeof(wv));
    memcpy(x2, &xa, sizeof(xa));
    memcpy(x2 + kVecFp8 / 4, &xb, sizeof(xb));
#pragma unroll
    for (int k = 0; k < kVecFp8 / 2; ++k) {
      const float2 wf = static_cast<float2>(w2[k]);
      const float2 xf = __bfloat1622float2(x2[k]);
      p[2 * k] = __fmul_rn(wf.x, xf.x);
      p[2 * k + 1] = __fmul_rn(wf.y, xf.y);
    }
  } else {
#pragma unroll
    for (int k = 0; k < kVecFp8; ++k) {
      const int64_t j = base + k;
      p[k] = j < n ? __fmul_rn(fp8_to_float(w[j]), __bfloat162float(x[j])) : 0.0f;
    }
  }
}

// Halving reduction over the first `width` lanes (a power of two); lane 0 gets the sum.
__device__ __forceinline__ float halving_sum(float v, int width) {
  for (int offset = width / 2; offset > 0; offset /= 2) {
    v = __fadd_rn(v, __shfl_down_sync(kFullMask, v, offset));
  }
  return v;
}

template <typename WT, int V>
__global__ void gemv_coalesced_kernel(const WT* __restrict__ W, const __nv_bfloat16* __restrict__ x,
                                      float* __restrict__ y, int64_t n, int64_t per_thread,
                                      bool vector_loads) {
  __shared__ float warp_sums[kWarp];
  const int t = static_cast<int>(threadIdx.x);
  const int threads = static_cast<int>(blockDim.x);
  const WT* w_row = W + static_cast<int64_t>(blockIdx.x) * n;

  float p[V];
  products(w_row, x, static_cast<int64_t>(t) * V, n, vector_loads, p);
  float acc = p[0];
#pragma unroll
  for (int k = 1; k < V; ++k) {
    acc = __fadd_rn(acc, p[k]);
  }
  for (int64_t v = 1; v < per_thread; ++v) {
    products(w_row, x, (v * threads + t) * V, n, vector_loads, p);
#pragma unroll
    for (int k = 0; k < V; ++k) {
      acc = __fadd_rn(acc, p[k]);
    }
  }

  const int lane = t % kWarp;
  const int warp = t / kWarp;
  const int num_warps = threads / kWarp;
  acc = halving_sum(acc, kWarp);
  if (lane == 0) {
    warp_sums[warp] = acc;
  }
  __syncthreads();
  if (warp == 0) {
    float s = lane < num_warps ? warp_sums[lane] : 0.0f;
    s = halving_sum(s, num_warps);
    if (lane == 0) {
      y[blockIdx.x] = s;
    }
  }
}

}  // namespace

cudaError_t gemv_bf16_coalesced(const void* W, const void* x, float* y, int64_t m, int64_t n,
                                int threads, cudaStream_t stream) {
  const int64_t per_thread = ((n + kVec - 1) / kVec + threads - 1) / threads;
  const bool vector_loads = n % kVec == 0 && reinterpret_cast<uintptr_t>(W) % 16 == 0 &&
                            reinterpret_cast<uintptr_t>(x) % 16 == 0;
  gemv_coalesced_kernel<__nv_bfloat16, kVec><<<static_cast<unsigned int>(m), threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(W), static_cast<const __nv_bfloat16*>(x), y, n, per_thread,
      vector_loads);
  return cudaGetLastError();
}

cudaError_t gemv_fp8_coalesced(const void* W, const void* x, float* y, int64_t m, int64_t n,
                               int threads, cudaStream_t stream) {
  const int64_t per_thread = ((n + kVecFp8 - 1) / kVecFp8 + threads - 1) / threads;
  const bool vector_loads = n % kVecFp8 == 0 && reinterpret_cast<uintptr_t>(W) % 16 == 0 &&
                            reinterpret_cast<uintptr_t>(x) % 16 == 0;
  gemv_coalesced_kernel<uint8_t, kVecFp8><<<static_cast<unsigned int>(m), threads, 0, stream>>>(
      static_cast<const uint8_t*>(W), static_cast<const __nv_bfloat16*>(x), y, n, per_thread,
      vector_loads);
  return cudaGetLastError();
}

cudaError_t gemv_bf16_naive(const void* W, const void* x, float* y, int64_t m, int64_t n,
                            int threads, cudaStream_t stream) {
  const int64_t chunk = (n + threads - 1) / threads;
  const size_t shared = static_cast<size_t>(threads) * sizeof(float);
  gemv_bf16_naive_kernel<<<static_cast<unsigned int>(m), threads, shared, stream>>>(
      static_cast<const __nv_bfloat16*>(W), static_cast<const __nv_bfloat16*>(x), y, n, chunk);
  return cudaGetLastError();
}

}  // namespace pck
