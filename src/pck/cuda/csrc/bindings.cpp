// PyTorch bindings for the GEMV kernels and the trusted GEMV and sampling checkers.

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "checker.h"

#include <cstdint>
#include <functional>
#include <limits>
#include <tuple>
#include <vector>

namespace pck {
cudaError_t gemv_bf16_naive(const void* W, const void* x, float* y, int64_t m, int64_t n,
                            int threads, cudaStream_t stream);
cudaError_t gemv_bf16_coalesced(const void* W, const void* x, float* y, int64_t m, int64_t n,
                                int threads, cudaStream_t stream);
cudaError_t gemv_template(const void* W, const void* x, float* y, int64_t m, int64_t n,
                          int threads, int rows_per_block, int vec, int unroll, int flags,
                          cudaStream_t stream);
cudaError_t gemv_column(const void* WT, const void* x, float* y, float* scratch, int64_t m,
                        int64_t n, int threads, int vec, int splits, float skip_below,
                        cudaStream_t stream);
int64_t sampling_scratch_words(int64_t vocab);
int64_t sampling_partial_bytes();
cudaError_t sampling_check(const float* logits, const int64_t* sel, const int64_t* len,
                           int64_t rows, int64_t vocab, int64_t max_len, int64_t k, bool top_p,
                           double p, double temperature, double slack, int64_t segments,
                           unsigned int* scratch, void* partials, bool* passed, float* min_sel,
                           float* max_rej, double* mass, double* without, cudaStream_t stream);
}  // namespace pck

namespace {

using Launcher = std::function<cudaError_t(const void*, const void*, float*, int64_t, int64_t,
                                           int, cudaStream_t)>;

torch::Tensor launch_gemv(const torch::Tensor& W, const torch::Tensor& x, int64_t threads,
                          int64_t min_threads, const Launcher& launcher) {
  TORCH_CHECK(W.is_cuda() && x.is_cuda(), "W and x must be CUDA tensors");
  TORCH_CHECK(W.device() == x.device(), "W and x must be on the same device");
  TORCH_CHECK(W.scalar_type() == torch::kBFloat16 && x.scalar_type() == torch::kBFloat16,
              "W and x must be bfloat16");
  TORCH_CHECK(W.dim() == 2 && x.dim() == 1 && x.size(0) == W.size(1),
              "expected W of shape (m, n) and x of shape (n,), got ", W.sizes(), " and ",
              x.sizes());
  TORCH_CHECK(W.is_contiguous() && x.is_contiguous(), "W and x must be contiguous");
  TORCH_CHECK(threads >= min_threads && threads <= 1024 && (threads & (threads - 1)) == 0,
              "threads must be a power of two in [", min_threads, ", 1024], got ", threads);
  TORCH_CHECK(W.size(0) <= std::numeric_limits<int32_t>::max(), "too many rows");

  const c10::cuda::CUDAGuard guard(W.device());
  auto y = torch::empty({W.size(0)}, W.options().dtype(torch::kFloat32));
  if (W.size(0) == 0) {
    return y;
  }
  TORCH_CHECK(W.size(1) > 0, "n must be positive");
  const cudaError_t err =
      launcher(W.data_ptr(), x.data_ptr(), y.data_ptr<float>(), W.size(0), W.size(1),
               static_cast<int>(threads), c10::cuda::getCurrentCUDAStream().stream());
  TORCH_CHECK(err == cudaSuccess, "gemv kernel launch failed: ", cudaGetErrorString(err));
  return y;
}

torch::Tensor gemv_naive(const torch::Tensor& W, const torch::Tensor& x, int64_t threads) {
  return launch_gemv(W, x, threads, 1, pck::gemv_bf16_naive);
}

torch::Tensor gemv_coalesced(const torch::Tensor& W, const torch::Tensor& x, int64_t threads) {
  return launch_gemv(W, x, threads, 32, pck::gemv_bf16_coalesced);
}

torch::Tensor gemv_template(const torch::Tensor& W, const torch::Tensor& x, int64_t threads,
                            int64_t rows_per_block, int64_t vec, int64_t unroll,
                            int64_t flags) {
  TORCH_CHECK(rows_per_block > 0 && threads % rows_per_block == 0,
              "rows_per_block must divide threads");
  const auto launcher = [&](const void* w, const void* xp, float* y, int64_t m, int64_t n,
                            int t, cudaStream_t stream) {
    return pck::gemv_template(w, xp, y, m, n, t, static_cast<int>(rows_per_block),
                              static_cast<int>(vec), static_cast<int>(unroll),
                              static_cast<int>(flags), stream);
  };
  return launch_gemv(W, x, threads, 32, launcher);
}

torch::Tensor gemv_column(const torch::Tensor& WT, const torch::Tensor& x, int64_t threads,
                          int64_t vec, int64_t splits, double skip_below) {
  TORCH_CHECK(WT.is_cuda() && x.is_cuda() && WT.device() == x.device(),
              "WT and x must be CUDA tensors on one device");
  TORCH_CHECK(WT.scalar_type() == torch::kBFloat16 && x.scalar_type() == torch::kBFloat16,
              "WT and x must be bfloat16");
  TORCH_CHECK(WT.dim() == 2 && x.dim() == 1 && x.size(0) == WT.size(0),
              "expected WT of shape (n, m) and x of shape (n,)");
  TORCH_CHECK(WT.is_contiguous() && x.is_contiguous(), "WT and x must be contiguous");
  const c10::cuda::CUDAGuard guard(WT.device());
  const int64_t n = WT.size(0), m = WT.size(1);
  auto y = torch::empty({m}, WT.options().dtype(torch::kFloat32));
  auto scratch = torch::empty({splits > 1 ? splits * m : 0}, y.options());
  if (m == 0) return y;
  const cudaError_t err = pck::gemv_column(
      WT.data_ptr(), x.data_ptr(), y.data_ptr<float>(), scratch.data_ptr<float>(), m, n,
      static_cast<int>(threads), static_cast<int>(vec), static_cast<int>(splits),
      static_cast<float>(skip_below), c10::cuda::getCurrentCUDAStream().stream());
  TORCH_CHECK(err == cudaSuccess, "column gemv launch failed: ", cudaGetErrorString(err));
  return y;
}

void check_cuda(const torch::Tensor& t, const char* name, torch::ScalarType dtype,
                const torch::Device& device) {
  TORCH_CHECK(t.is_cuda() && t.device() == device, name, " must be on ", device);
  TORCH_CHECK(t.scalar_type() == dtype, name, " must be ", dtype, ", got ", t.scalar_type());
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

using CheckOutputs = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>;

// Validates one check's tensors and fills its descriptor; returns its outputs.
CheckOutputs describe(const torch::Tensor& c, const torch::Tensor& a, const torch::Tensor& r,
                      const torch::Tensor& x, const torch::Tensor& y, torch::Tensor& partial_cx,
                      torch::Tensor& partial_s, torch::Tensor& counters, double coef,
                      double geps, double tau_floor, const torch::Device& device,
                      pck::CheckDesc& d) {
  check_cuda(c, "c", torch::kFloat32, device);
  check_cuda(a, "a", torch::kBFloat16, device);
  check_cuda(r, "r", torch::kInt8, device);
  check_cuda(x, "x", torch::kBFloat16, device);
  check_cuda(y, "y", torch::kFloat32, device);
  check_cuda(partial_cx, "partial_cx", torch::kFloat64, device);
  check_cuda(partial_s, "partial_s", torch::kFloat64, device);
  check_cuda(counters, "counters", torch::kInt32, device);
  TORCH_CHECK(c.dim() == 3 && a.dim() == 2 && r.dim() == 3 && x.dim() == 1 && y.dim() == 1,
              "expected c (K, nb, n), a (nb, n), r (K, nb, b), x (n,), y (m,)");
  const int64_t checks = c.size(0), num_blocks = c.size(1), n = c.size(2), block = r.size(2);
  TORCH_CHECK(a.size(0) == num_blocks && a.size(1) == n, "a must have shape (nb, n)");
  TORCH_CHECK(r.size(0) == checks && r.size(1) == num_blocks, "r must have shape (K, nb, b)");
  TORCH_CHECK(x.size(0) == n, "x must have shape (n,)");
  TORCH_CHECK(y.size(0) == num_blocks * block, "y must have shape (nb * b,)");
  TORCH_CHECK(checks >= 1 && num_blocks >= 1 && n >= 1 && block >= 1, "empty checker");
  TORCH_CHECK(partial_cx.dim() == 3 && partial_cx.size(0) == checks &&
                  partial_cx.size(1) == num_blocks && partial_s.sizes() == partial_cx.sizes(),
              "partial sums must have shape (K, nb, segments)");
  TORCH_CHECK(counters.dim() == 2 && counters.size(0) == checks && counters.size(1) == num_blocks,
              "counters must have shape (K, nb)");
  const int64_t segments = partial_cx.size(2);
  TORCH_CHECK(segments >= 1 && segments <= n, "segments must be in [1, n], got ", segments);
  TORCH_CHECK(coef > 0 && geps > 0 && tau_floor > 0, "coef, geps and floor must be positive");

  const auto shape = std::vector<int64_t>{checks, num_blocks};
  auto passed = torch::empty(shape, y.options().dtype(torch::kBool));
  auto rho = torch::empty(shape, y.options().dtype(torch::kFloat64));
  auto tau = torch::empty(shape, y.options().dtype(torch::kFloat64));
  d = pck::CheckDesc{c.data_ptr<float>(),
                     a.data_ptr(),
                     r.data_ptr<int8_t>(),
                     x.data_ptr(),
                     y.data_ptr<float>(),
                     partial_cx.data_ptr<double>(),
                     partial_s.data_ptr<double>(),
                     reinterpret_cast<unsigned int*>(counters.data_ptr<int32_t>()),
                     passed.data_ptr<bool>(),
                     rho.data_ptr<double>(),
                     tau.data_ptr<double>(),
                     n,
                     (n + segments - 1) / segments,
                     static_cast<int>(block),
                     static_cast<int>(num_blocks),
                     static_cast<int>(checks),
                     static_cast<int>(segments),
                     coef,
                     geps,
                     tau_floor};
  return {passed, rho, tau};
}

// Checks up to kMaxBatch GEMV outputs in one launch; returns (passed, rho, tau) per check.
std::vector<CheckOutputs> blocked_check(
    const std::vector<torch::Tensor>& c, const std::vector<torch::Tensor>& a,
    const std::vector<torch::Tensor>& r, const std::vector<torch::Tensor>& x,
    const std::vector<torch::Tensor>& y, std::vector<torch::Tensor>& partial_cx,
    std::vector<torch::Tensor>& partial_s, std::vector<torch::Tensor>& counters,
    const std::vector<double>& coef, const std::vector<double>& geps,
    const std::vector<double>& tau_floor) {
  const size_t count = c.size();
  TORCH_CHECK(count >= 1 && count <= static_cast<size_t>(pck::kMaxBatch),
              "batch size must be in [1, ", pck::kMaxBatch, "], got ", count);
  TORCH_CHECK(a.size() == count && r.size() == count && x.size() == count &&
                  y.size() == count && partial_cx.size() == count &&
                  partial_s.size() == count && counters.size() == count &&
                  coef.size() == count && geps.size() == count && tau_floor.size() == count,
              "all argument lists must have the same length");
  TORCH_CHECK(y[0].is_cuda(), "y must be a CUDA tensor");
  const auto device = y[0].device();
  const c10::cuda::CUDAGuard guard(device);

  pck::CheckBatch batch{};
  batch.count = static_cast<int>(count);
  std::vector<CheckOutputs> outputs;
  outputs.reserve(count);
  for (size_t g = 0; g < count; ++g) {
    TORCH_CHECK(y[g].is_cuda(), "y must be a CUDA tensor");
    outputs.push_back(describe(c[g], a[g], r[g], x[g], y[g], partial_cx[g], partial_s[g],
                               counters[g], coef[g], geps[g], tau_floor[g], device,
                               batch.desc[g]));
    const auto& d = batch.desc[g];
    batch.start[g + 1] =
        batch.start[g] + static_cast<int64_t>(d.checks) * d.num_blocks * d.segments;
  }
  TORCH_CHECK(batch.start[count] <= std::numeric_limits<int32_t>::max(), "too many blocks");
  const cudaError_t err =
      pck::blocked_check(batch, c10::cuda::getCurrentCUDAStream().stream());
  TORCH_CHECK(err == cudaSuccess, "checker kernel launch failed: ", cudaGetErrorString(err));
  return outputs;
}

// Checks sampler output for every row of logits; returns (passed, min_sel, max_rej, mass,
// mass_without_smallest), each of shape (rows,).
std::vector<torch::Tensor> sampling_check(const torch::Tensor& logits, const torch::Tensor& sel,
                                          const torch::Tensor& len, int64_t k, bool top_p,
                                          double p, double temperature, double slack,
                                          int64_t segments) {
  TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
  const auto device = logits.device();
  check_cuda(logits, "logits", torch::kFloat32, device);
  check_cuda(sel, "selected", torch::kInt64, device);
  check_cuda(len, "lengths", torch::kInt64, device);
  TORCH_CHECK(logits.dim() == 2 && sel.dim() == 2 && len.dim() == 1,
              "expected logits (B, V), selected (B, L), lengths (B,)");
  const int64_t rows = logits.size(0), vocab = logits.size(1), max_len = sel.size(1);
  TORCH_CHECK(sel.size(0) == rows && len.size(0) == rows, "batch sizes differ");
  TORCH_CHECK(rows >= 1 && vocab >= 1 && max_len >= 1, "empty input");
  TORCH_CHECK(rows <= 65535, "too many rows");
  TORCH_CHECK(segments >= 1 && segments <= vocab && segments <= std::numeric_limits<int32_t>::max(),
              "segments must be in [1, V], got ", segments);
  TORCH_CHECK(temperature > 0 && slack >= 0, "temperature must be positive, slack >= 0");

  const c10::cuda::CUDAGuard guard(device);
  auto opts = logits.options();
  auto scratch = torch::zeros({rows, pck::sampling_scratch_words(vocab)}, opts.dtype(torch::kInt32));
  auto partials =
      torch::empty({rows * segments * pck::sampling_partial_bytes()}, opts.dtype(torch::kUInt8));
  auto passed = torch::empty({rows}, opts.dtype(torch::kBool));
  auto min_sel = torch::empty({rows}, opts.dtype(torch::kFloat32));
  auto max_rej = torch::empty({rows}, opts.dtype(torch::kFloat32));
  auto mass = torch::empty({rows}, opts.dtype(torch::kFloat64));
  auto without = torch::empty({rows}, opts.dtype(torch::kFloat64));
  const cudaError_t err = pck::sampling_check(
      logits.data_ptr<float>(), sel.data_ptr<int64_t>(), len.data_ptr<int64_t>(), rows, vocab,
      max_len, k, top_p, p, temperature, slack, segments,
      reinterpret_cast<unsigned int*>(scratch.data_ptr<int32_t>()), partials.data_ptr(),
      passed.data_ptr<bool>(), min_sel.data_ptr<float>(), max_rej.data_ptr<float>(),
      mass.data_ptr<double>(), without.data_ptr<double>(),
      c10::cuda::getCurrentCUDAStream().stream());
  TORCH_CHECK(err == cudaSuccess, "sampling checker launch failed: ", cudaGetErrorString(err));
  return {passed, min_sel, max_rej, mass, without};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemv_naive", &gemv_naive, "Naive BF16 GEMV with FP32 accumulation (y in FP32)",
        pybind11::arg("W"), pybind11::arg("x"), pybind11::arg("threads"));
  m.def("gemv_coalesced",
        &gemv_coalesced, "Coalesced BF16 GEMV with 16-byte loads and warp-shuffle reduction",
        pybind11::arg("W"), pybind11::arg("x"), pybind11::arg("threads"));
  m.def("gemv_template", &gemv_template,
        "The agent's GEMV template; flags: 1 acc BF16, 2 partials BF16, 4 output BF16, "
        "8 skip tail",
        pybind11::arg("W"), pybind11::arg("x"), pybind11::arg("threads"),
        pybind11::arg("rows_per_block"), pybind11::arg("vec"), pybind11::arg("unroll"),
        pybind11::arg("flags"));
  m.def("gemv_column", &gemv_column,
        "The agent's column-major GEMV template over WT = W^T (n, m)", pybind11::arg("WT"),
        pybind11::arg("x"), pybind11::arg("threads"), pybind11::arg("vec"),
        pybind11::arg("splits"), pybind11::arg("skip_below"));
  m.def("blocked_check", &blocked_check,
        "Trusted blocked checksum checker for up to 8 GEMV outputs in one launch; returns "
        "(passed, rho, tau) per check, each of shape (K, nb)",
        pybind11::arg("c"), pybind11::arg("a"), pybind11::arg("r"), pybind11::arg("x"),
        pybind11::arg("y"), pybind11::arg("partial_cx"), pybind11::arg("partial_s"),
        pybind11::arg("counters"), pybind11::arg("coef"), pybind11::arg("geps"),
        pybind11::arg("tau_floor"));
  m.def("sampling_check", &sampling_check, "Trusted top-k / top-p sampling checker",
        pybind11::arg("logits"), pybind11::arg("selected"), pybind11::arg("lengths"),
        pybind11::arg("k"), pybind11::arg("top_p"), pybind11::arg("p"),
        pybind11::arg("temperature"), pybind11::arg("slack"), pybind11::arg("segments"));
}
