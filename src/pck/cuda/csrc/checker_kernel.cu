// Trusted blocked checksum checker for y = W x (production mode).
//
// Inputs are precomputed at weight load by trusted code (pck.cuda.checker):
//   c[k, beta, :] = r[k, beta, :]^T W_beta     FP32 (computed in FP64, rounded to nearest)
//   a[beta, :]    = sum_{i in beta} |W_i|       BF16, rounded UP
//   r[k, beta, :] in {-1, +1}                   int8, secret
// For every check k and row block beta, after the GEMV has finished:
//   rho = c . x - r . y_beta                       FP64, round to nearest
//   s   = a . |x|                                  FP32, every operation rounded UP
//   tau = coef * s + geps * ((1 + u32) * s + sum |y_beta|)    FP64, rounded UP
// where coef >= gamma_h + u32 and geps >= gamma_{n+b+1}(u64) are rounded up on the host.
// The block passes only if every y in it is finite and |rho| <= tau, so NaN/Inf always
// raise the alarm.
//
// Traffic per block of b BF16 weight rows: K FP32 c rows + one BF16 a row, i.e. about
// (2K + 1) / b of the GEMV's.
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
//
// Parallelism: the n columns of every (check, block) pair are split into `segments`
// CUDA blocks so the grid fills the GPU even when there are few row blocks. Each writes
// partial sums; the last to finish (atomic ticket) adds them in a fixed order, handles the
// y terms and writes the verdict. The error bounds hold for any summation order. The
// counters are reset to zero by the last block, so a checker must not be used from two
// streams at once.

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
    blocked_check_kernel(const float* __restrict__ c, const __nv_bfloat16* __restrict__ a,
                         const int8_t* __restrict__ r, const __nv_bfloat16* __restrict__ x,
                         const float* __restrict__ y, int64_t n, int64_t seg_len, int block,
                         int num_blocks, double coef, double geps, double* partial_cx,
                         double* partial_s, unsigned int* counters, bool* __restrict__ passed,
                         double* __restrict__ rho_out, double* __restrict__ tau_out) {
  __shared__ double shared[kWarp];
  __shared__ int nonfinite;
  __shared__ bool is_last;
  const int t = static_cast<int>(threadIdx.x);
  const int64_t check_block = blockIdx.x;  // = k * num_blocks + beta
  const int64_t segments = gridDim.y;
  const int64_t seg = blockIdx.y;
  const int64_t beta = check_block % num_blocks;
  const float* c_row = c + check_block * n;
  const __nv_bfloat16* a_row = a + beta * n;

  if (t == 0) {
    nonfinite = 0;
  }
  __syncthreads();

  // Phase 1 (every segment): partial c . x and a . |x| over this segment's columns.
  const int64_t j_end = min(n, (seg + 1) * seg_len);
  double cx = 0.0;
  float s = 0.0f;
#pragma unroll 4
  for (int64_t j = seg * seg_len + t; j < j_end; j += kThreads) {
    const float xj = __bfloat162float(x[j]);
    cx = __fma_rn(static_cast<double>(c_row[j]), static_cast<double>(xj), cx);  // exact product
    s = __fadd_ru(s, __fmul_ru(__bfloat162float(a_row[j]), fabsf(xj)));
  }
  cx = block_sum<false>(cx, shared);
  // FP32 round-up partial sums are exact in FP64, and the FP64 tree also rounds up.
  const double s_up = block_sum<true>(static_cast<double>(s), shared);
  if (t == 0) {
    partial_cx[check_block * segments + seg] = cx;
    partial_s[check_block * segments + seg] = s_up;
    __threadfence();  // publish partials before taking a ticket
    is_last = atomicAdd(&counters[check_block], 1u) == segments - 1;
  }
  __syncthreads();
  if (!is_last) {
    return;
  }

  // Phase 2 (last segment to finish): the y terms, then combine and decide.
  __threadfence();
  const int8_t* r_row = r + check_block * block;
  const float* y_blk = y + beta * block;
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
  ry = block_sum<false>(ry, shared);
  const double yabs_up = block_sum<true>(static_cast<double>(yabs), shared);

  if (t == 0) {
    const volatile double* pcx = partial_cx + check_block * segments;
    const volatile double* ps = partial_s + check_block * segments;
    double cx_total = 0.0;
    double s_total = 0.0;
    for (int64_t i = 0; i < segments; ++i) {  // fixed order: deterministic result
      cx_total = __dadd_rn(cx_total, pcx[i]);
      s_total = __dadd_ru(s_total, ps[i]);
    }
    counters[check_block] = 0;  // ready for the next call

    const double rho = __dsub_rn(cx_total, ry);
    const double cabs_up = __dmul_ru(kOnePlusU32, s_total);
    const double tau =
        __dadd_ru(__dmul_ru(coef, s_total), __dmul_ru(geps, __dadd_ru(cabs_up, yabs_up)));
    passed[check_block] = nonfinite == 0 && fabs(rho) <= tau;  // NaN compares false
    rho_out[check_block] = rho;
    tau_out[check_block] = tau;
  }
}

}  // namespace

cudaError_t blocked_check(const float* c, const void* a, const int8_t* r, const void* x,
                          const float* y, int64_t n, int block, int num_blocks, int checks,
                          int segments, double coef, double geps, double* partial_cx,
                          double* partial_s, unsigned int* counters, bool* passed, double* rho,
                          double* tau, cudaStream_t stream) {
  const int64_t seg_len = (n + segments - 1) / segments;
  const dim3 grid(static_cast<unsigned int>(checks) * num_blocks,
                  static_cast<unsigned int>(segments));
  blocked_check_kernel<<<grid, kThreads, 0, stream>>>(
      c, static_cast<const __nv_bfloat16*>(a), r, static_cast<const __nv_bfloat16*>(x), y, n,
      seg_len, block, num_blocks, coef,
      geps, partial_cx, partial_s, counters, passed, rho, tau);
  return cudaGetLastError();
}

}  // namespace pck
