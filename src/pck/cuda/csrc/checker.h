// Host/device interface of the trusted blocked checker (see checker_kernel.cu).

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace pck {

// Everything needed to check one GEMV output y = W x. Pointers are device pointers; a and x
// point to BF16 data.
struct CheckDesc {
  const float* c;          // (K, nb, n) FP32 checksums
  const void* a;           // (nb, n) BF16, rounded up
  const int8_t* r;         // (K, nb, b) signs
  const void* x;           // (n,) BF16
  const float* y;          // (nb * b,) FP32 output to check
  double* partial_cx;      // (K, nb, segments) scratch
  double* partial_s;       // (K, nb, segments) scratch
  unsigned int* counters;  // (K, nb) zero between calls
  bool* passed;            // (K, nb) outputs
  double* rho;
  double* tau;
  int64_t n;
  int64_t seg_len;
  int block;
  int num_blocks;
  int checks;
  int segments;
  double coef;
  double geps;
  double tau_floor;
};

constexpr int kMaxBatch = 8;

// Up to kMaxBatch checks in one launch. Work item i belongs to check g when
// start[g] <= i < start[g + 1]; each check has checks * num_blocks * segments items.
struct CheckBatch {
  CheckDesc desc[kMaxBatch];
  int64_t start[kMaxBatch + 1];
  int count;
};

cudaError_t blocked_check(const CheckBatch& batch, cudaStream_t stream);

}  // namespace pck
