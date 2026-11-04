// Trusted blocked checksum checker for y = W x (production mode).
//
// Inputs are precomputed at weight load by trusted code (pck.cuda.checker):
//   c[k, beta, :] = r[k, beta, :]^T W_beta     FP32 (computed in FP64, rounded to nearest)
//   a[beta, :]    = sum_{i in beta} |W_i|       FP32, rounded UP
//   r[k, beta, :] in {-1, +1}                   int8, secret
// For every check k and row block beta, after the GEMV has finished:
//   rho = c . x - r . y_beta                                          (FP64, nearest)
//   tau = coef * (a . |x|) + geps * (|c| . |x| + sum |y_beta|)        (FP64, rounded up)
// where coef >= gamma_h + u32 and geps >= gamma_{n+b+1}(u64) are rounded up on the host.
// The block passes only if every y in it is finite and |rho| <= tau, so NaN/Inf always
// raise the alarm.
//
// Products c_j * x_j and a_j * |x_j| (FP32 x BF16: 24 + 8 significand bits) are exact in
// FP64. Sums for rho round to nearest (their error is what geps covers); sums for tau
// round up, so tau is never below its exact value.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace pck {
namespace {

constexpr int kThreads = 256;
constexpr int kWarp = 32;
constexpr unsigned kFullMask = 0xffffffffu;

template <bool kRoundUp>
__device__ __forceinline__ double add(double a, double b) {
  return kRoundUp ? __dadd_ru(a, b) : __dadd_rn(a, b);
}

// Sum over the thread block; the result is valid in thread 0. Reuses `shared`.
template <bool kRoundUp>
__device__ double block_sum(double v, double* shared) {
  const int lane = static_cast<int>(threadIdx.x) % kWarp;
  const int warp = static_cast<int>(threadIdx.x) / kWarp;
  for (int offset = kWarp / 2; offset > 0; offset /= 2) {
    v = add<kRoundUp>(v, __shfl_down_sync(kFullMask, v, offset));
  }
  if (lane == 0) {
    shared[warp] = v;
  }
  __syncthreads();
  double total = 0.0;
  if (warp == 0) {
    total = lane < kThreads / kWarp ? shared[lane] : 0.0;
    for (int offset = kWarp / 2; offset > 0; offset /= 2) {
      total = add<kRoundUp>(total, __shfl_down_sync(kFullMask, total, offset));
    }
  }
  __syncthreads();
  return total;
}

__global__ void __launch_bounds__(kThreads)
    blocked_check_kernel(const float* __restrict__ c, const float* __restrict__ a,
                         const int8_t* __restrict__ r, const __nv_bfloat16* __restrict__ x,
                         const float* __restrict__ y, int64_t n, int block, int num_blocks,
                         double coef, double geps, bool* __restrict__ passed,
                         double* __restrict__ rho_out, double* __restrict__ tau_out) {
  __shared__ double shared[kWarp];
  __shared__ int nonfinite;
  const int t = static_cast<int>(threadIdx.x);
  const int64_t check_block = blockIdx.x;  // = k * num_blocks + beta
  const int64_t beta = check_block % num_blocks;
  const float* c_row = c + check_block * n;
  const float* a_row = a + beta * n;
  const int8_t* r_row = r + check_block * block;
  const float* y_blk = y + beta * block;

  if (t == 0) {
    nonfinite = 0;
  }
  __syncthreads();

  double cx = 0.0, cabs = 0.0, s = 0.0;
  for (int64_t j = t; j < n; j += kThreads) {
    const double xj = static_cast<double>(__bfloat162float(x[j]));
    const double p = static_cast<double>(c_row[j]) * xj;  // exact
    cx = __dadd_rn(cx, p);
    cabs = __dadd_ru(cabs, fabs(p));
    s = __dadd_ru(s, static_cast<double>(a_row[j]) * fabs(xj));  // exact product
  }
  double ry = 0.0, yabs = 0.0;
  bool bad = false;
  for (int i = t; i < block; i += kThreads) {
    const double yi = static_cast<double>(y_blk[i]);
    bad |= !isfinite(yi);
    ry = __dadd_rn(ry, r_row[i] > 0 ? yi : -yi);
    yabs = __dadd_ru(yabs, fabs(yi));
  }
  if (bad) {
    atomicOr(&nonfinite, 1);
  }

  cx = block_sum<false>(cx, shared);
  ry = block_sum<false>(ry, shared);
  cabs = block_sum<true>(cabs, shared);
  yabs = block_sum<true>(yabs, shared);
  s = block_sum<true>(s, shared);

  if (t == 0) {
    const double rho = __dsub_rn(cx, ry);
    const double tau = __dadd_ru(__dmul_ru(coef, s), __dmul_ru(geps, __dadd_ru(cabs, yabs)));
    passed[check_block] = nonfinite == 0 && fabs(rho) <= tau;  // NaN compares false
    rho_out[check_block] = rho;
    tau_out[check_block] = tau;
  }
}

}  // namespace

cudaError_t blocked_check(const float* c, const float* a, const int8_t* r, const void* x,
                          const float* y, int64_t n, int block, int num_blocks, int checks,
                          double coef, double geps, bool* passed, double* rho, double* tau,
                          cudaStream_t stream) {
  const unsigned int grid = static_cast<unsigned int>(checks) * num_blocks;
  blocked_check_kernel<<<grid, kThreads, 0, stream>>>(
      c, a, r, static_cast<const __nv_bfloat16*>(x), y, n, block, num_blocks, coef, geps, passed,
      rho, tau);
  return cudaGetLastError();
}

}  // namespace pck
