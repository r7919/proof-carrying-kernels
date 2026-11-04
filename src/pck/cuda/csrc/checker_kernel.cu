// Trusted blocked checksum checker for y = W x (production mode).
//
// Inputs are precomputed at weight load by trusted code (pck.cuda.checker):
//   c[k, beta, :] = r[k, beta, :]^T W_beta     FP32 (computed in FP64, rounded to nearest)
//   a[beta, :]    = sum_{i in beta} |W_i|       FP32, rounded UP
//   r[k, beta, :] in {-1, +1}                   int8, secret
// For every check k and row block beta, after the GEMV has finished:
//   rho = c . x - r . y_beta                       FP64, round to nearest
//   s   = a . |x|                                  FP32, every operation rounded UP
//   tau = coef * s + geps * ((1 + u32) * s + sum |y_beta|)    FP64, rounded UP
// where coef >= gamma_h + u32 and geps >= gamma_{n+b+1}(u64) are rounded up on the host.
// The block passes only if every y in it is finite and |rho| <= tau, so NaN/Inf always
// raise the alarm.
//
// Why this is sound and cheap:
//  * c_j * x_j (FP32 x BF16: 24 + 8 significand bits) is exact in FP64, so one FP64 FMA
//    per element computes rho with only the summation rounding, which geps covers.
//  * s is an upper bound on a . |x| because every FP32 operation rounds up; it is looser
//    than exact by at most ~(n + 1) * 2^-24 relative.
//  * The FP64 rounding of rho is at most gamma * (sum |c_j x_j| + sum |y|), and
//    |c_j| <= (1 + u32) * a_j because |r^T W_j| <= sum |W_ij| and c is rounded to nearest.
// Only one FP64 operation per weight column runs in the hot loop, which matters on GPUs
// whose FP64 rate is 1/64 of FP32.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace pck {
namespace {

constexpr int kThreads = 256;
constexpr int kWarp = 32;
constexpr unsigned kFullMask = 0xffffffffu;
constexpr double kOnePlusU32 = 1.0 + 0x1p-24;  // exact in FP64

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

  double cx = 0.0;
  float s = 0.0f;
  for (int64_t j = t; j < n; j += kThreads) {
    const float xj = __bfloat162float(x[j]);
    cx = __fma_rn(static_cast<double>(c_row[j]), static_cast<double>(xj), cx);  // exact product
    s = __fadd_ru(s, __fmul_ru(a_row[j], fabsf(xj)));
  }
  double ry = 0.0;
  float yabs = 0.0f;
  bool bad = false;
  for (int i = t; i < block; i += kThreads) {
    const float yi = y_blk[i];
    bad |= !isfinite(yi);
    ry = __dadd_rn(ry, static_cast<double>(r_row[i] > 0 ? yi : -yi));
    yabs = __fadd_ru(yabs, fabsf(yi));
  }
  if (bad) {
    atomicOr(&nonfinite, 1);
  }

  cx = block_sum<false>(cx, shared);
  ry = block_sum<false>(ry, shared);
  // FP32 round-up partial sums are exact in FP64, and the FP64 tree also rounds up.
  const double s_up = block_sum<true>(static_cast<double>(s), shared);
  const double yabs_up = block_sum<true>(static_cast<double>(yabs), shared);

  if (t == 0) {
    const double rho = __dsub_rn(cx, ry);
    const double cabs_up = __dmul_ru(kOnePlusU32, s_up);
    const double tau =
        __dadd_ru(__dmul_ru(coef, s_up), __dmul_ru(geps, __dadd_ru(cabs_up, yabs_up)));
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
