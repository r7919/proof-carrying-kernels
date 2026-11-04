// Naive BF16 GEMV: y = W x with FP32 accumulation and FP32 output.
//
// One block per output row, `threads` threads per block. Thread t sums the
// contiguous chunk [t * chunk, (t + 1) * chunk) sequentially, treating columns
// past n as +0, then a pairwise tree combines the threads. This is exactly the
// summation tree of pck.gemv.emulate_gemv, so outputs match it bit for bit and
// the tree height is (chunk - 1) + log2(threads).
//
// Explicit round-to-nearest intrinsics keep the compiler from contracting or
// reordering operations. (A BF16 x BF16 product is exact in FP32, so an FMA
// would give the same result, but the intrinsics make that independent of
// compiler flags.)

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace pck {
namespace {

__global__ void gemv_bf16_naive_kernel(const __nv_bfloat16* __restrict__ W,
                                       const __nv_bfloat16* __restrict__ x,
                                       float* __restrict__ y, int64_t n, int64_t chunk) {
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

}  // namespace

cudaError_t gemv_bf16_naive(const void* W, const void* x, float* y, int64_t m, int64_t n,
                            int threads, cudaStream_t stream) {
  const int64_t chunk = (n + threads - 1) / threads;
  const size_t shared = static_cast<size_t>(threads) * sizeof(float);
  gemv_bf16_naive_kernel<<<static_cast<unsigned int>(m), threads, shared, stream>>>(
      static_cast<const __nv_bfloat16*>(W), static_cast<const __nv_bfloat16*>(x), y, n, chunk);
  return cudaGetLastError();
}

}  // namespace pck
