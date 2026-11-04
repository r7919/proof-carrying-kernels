// Trusted blocked checksum checker for y = W x (production mode).
//
// Inputs are precomputed at weight load by trusted code (pck.cuda.checker):
//   c[k, beta, :] = r[k, beta, :]^T W_beta     FP32 (computed in FP64, rounded to nearest)
//   a[beta, :]    = sum_{i in beta} |W_i|       BF16, rounded UP
//   r[k, beta, :] in {-1, +1}                   int8, secret
// For every check k and row block beta, after the GEMV has finished:
//   rho = c . x - r . y_beta        c . x in compensated FP32 per thread, then FP64
//   s   = a . |x|                   FP32, every operation rounded UP
//   tau = coef * s + geps * ((1 + u32) * s + sum |y_beta|) + tau_floor  rounded UP
// where coef >= gamma_h + u32, geps >= gH + g2 + gH * g2 (gH = gamma_H(u64) for the FP64
// reduction height H, g2 = gamma_{2L}(u32)^2 for L terms per thread chain) and
// tau_floor >= n * 2^-148 are rounded up on the host. The block passes
// only if every y in it is finite and |rho| <= tau, so NaN/Inf always raise the alarm.
//
// Traffic per block of b BF16 weight rows: K FP32 c rows + one BF16 a row, i.e. about
// (2K + 1) / b of the GEMV's.
//
// Why this is sound and cheap:
//  * Each thread computes its part of c . x with Dot2 (Ogita, Rump, Oishi 2005):
//    TwoProduct (p + e = c_j * x_j exactly, via FMA) and TwoSum (hi + err = hi + p
//    exactly), accumulating the error terms in lo. hi + lo is then within
//    gamma_{2L}(u32)^2 * sum |c_j x_j| of the exact chain sum, before any rounding back to
//    FP32; hi and lo are combined in FP64. All of this is FP32 except one FP64 add per
//    thread, which matters on GPUs whose FP64 rate is 1/64 of FP32.
//  * Underflow can make TwoProduct inexact by at most 2^-149 per product; tau_floor covers
//    it.
//  * The FP64 reductions across threads and segments, and r . y, round to nearest; their
//    error is at most gamma_H(u64) * (sum |c_j x_j| + sum |y|), H <= 21 + segments +
//    ceil(b / 256). The Dot2 and FP64 errors compound, hence the gH * g2 term.
//  * s is an upper bound on a . |x| because every FP32 operation rounds up; it is looser
//    than exact by at most ~(n + 1) * 2^-24 relative.
//  * |c_j| <= (1 + u32) * a_j because |r^T W_j| <= sum |W_ij|, c is rounded to nearest
//    and a is rounded up, so sum |c_j x_j| <= (1 + u32) * s.
//
// Parallelism and batching: one launch checks up to kMaxBatch GEMV outputs (e.g. every
// GEMV of a decode layer), which pays the fixed launch cost once. The n columns of every
// (check, block) pair are split into `segments` CUDA blocks so the grid fills the GPU. Each
// writes partial sums; the last to finish (atomic ticket) adds them in a fixed order,
// handles the y terms and writes the verdict. The error bounds hold for any summation
// order. Counters are reset to zero by the last block, so a checker must not be used from
// two streams at once.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "checker.h"

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
    blocked_check_kernel(const __grid_constant__ CheckBatch batch) {
  __shared__ double shared[kWarp];
  __shared__ int nonfinite;
  __shared__ bool is_last;
  const int t = static_cast<int>(threadIdx.x);

  // Which check (GEMV) and which (check k, row block beta, segment) this block handles.
  const int64_t item = blockIdx.x;
  int g = 0;
  while (g + 1 < batch.count && item >= batch.start[g + 1]) {
    ++g;
  }
  const CheckDesc& d = batch.desc[g];
  const int64_t local = item - batch.start[g];
  const int64_t check_block = local / d.segments;  // = k * num_blocks + beta
  const int64_t seg = local % d.segments;
  const int64_t beta = check_block % d.num_blocks;
  const int64_t n = d.n;
  const float* c_row = d.c + check_block * n;
  const auto* a_row = static_cast<const __nv_bfloat16*>(d.a) + beta * n;
  const auto* x = static_cast<const __nv_bfloat16*>(d.x);

  if (t == 0) {
    nonfinite = 0;
  }
  __syncthreads();

  // Phase 1 (every segment): partial c . x and a . |x| over this segment's columns.
  const int64_t j_end = min(n, (seg + 1) * d.seg_len);
  float hi = 0.0f, lo = 0.0f, s = 0.0f;
#pragma unroll 4
  for (int64_t j = seg * d.seg_len + t; j < j_end; j += kThreads) {
    const float xj = __bfloat162float(x[j]);
    const float cj = c_row[j];
    const float p = __fmul_rn(cj, xj);
    const float e = __fmaf_rn(cj, xj, -p);  // TwoProduct: cj * xj = p + e
    const float sum = __fadd_rn(hi, p);     // TwoSum: hi + p = sum + err
    const float z = __fsub_rn(sum, hi);
    const float err = __fadd_rn(__fsub_rn(hi, __fsub_rn(sum, z)), __fsub_rn(p, z));
    hi = sum;
    lo = __fadd_rn(lo, __fadd_rn(err, e));
    s = __fadd_ru(s, __fmul_ru(__bfloat162float(a_row[j]), fabsf(xj)));
  }
  double cx = __dadd_rn(static_cast<double>(hi), static_cast<double>(lo));
  cx = block_sum<false>(cx, shared);
  // FP32 round-up partial sums are exact in FP64, and the FP64 tree also rounds up.
  const double s_up = block_sum<true>(static_cast<double>(s), shared);
  if (t == 0) {
    d.partial_cx[check_block * d.segments + seg] = cx;
    d.partial_s[check_block * d.segments + seg] = s_up;
    __threadfence();  // publish partials before taking a ticket
    is_last = atomicAdd(&d.counters[check_block], 1u) == static_cast<unsigned>(d.segments) - 1;
  }
  __syncthreads();
  if (!is_last) {
    return;
  }

  // Phase 2 (last segment to finish): the y terms, then combine and decide.
  __threadfence();
  const int8_t* r_row = d.r + check_block * d.block;
  const float* y_blk = d.y + beta * d.block;
  double ry = 0.0;
  float yabs = 0.0f;
  bool bad = false;
  for (int i = t; i < d.block; i += kThreads) {
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
    const volatile double* pcx = d.partial_cx + check_block * d.segments;
    const volatile double* ps = d.partial_s + check_block * d.segments;
    double cx_total = 0.0;
    double s_total = 0.0;
    for (int i = 0; i < d.segments; ++i) {  // fixed order: deterministic result
      cx_total = __dadd_rn(cx_total, pcx[i]);
      s_total = __dadd_ru(s_total, ps[i]);
    }
    d.counters[check_block] = 0;  // ready for the next call

    const double rho = __dsub_rn(cx_total, ry);
    const double cabs_up = __dmul_ru(kOnePlusU32, s_total);
    const double eps = __dmul_ru(d.geps, __dadd_ru(cabs_up, yabs_up));
    const double tau = __dadd_ru(__dadd_ru(__dmul_ru(d.coef, s_total), eps), d.tau_floor);
    d.passed[check_block] = nonfinite == 0 && fabs(rho) <= tau;  // NaN compares false
    d.rho[check_block] = rho;
    d.tau[check_block] = tau;
  }
}

}  // namespace

cudaError_t blocked_check(const CheckBatch& batch, cudaStream_t stream) {
  if (batch.count < 1 || batch.count > kMaxBatch) {
    return cudaErrorInvalidValue;
  }
  const int64_t items = batch.start[batch.count];
  blocked_check_kernel<<<static_cast<unsigned int>(items), kThreads, 0, stream>>>(batch);
  return cudaGetLastError();
}

}  // namespace pck
