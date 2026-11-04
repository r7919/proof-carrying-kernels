// PyTorch bindings for the GEMV kernels and the trusted checker.

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

namespace pck {
cudaError_t gemv_bf16_naive(const void* W, const void* x, float* y, int64_t m, int64_t n,
                            int threads, cudaStream_t stream);
cudaError_t gemv_bf16_coalesced(const void* W, const void* x, float* y, int64_t m, int64_t n,
                                int threads, cudaStream_t stream);
cudaError_t blocked_check(const float* c, const float* a, const int8_t* r, const void* x,
                          const float* y, int64_t n, int block, int num_blocks, int checks,
                          int segments, double coef, double geps, double* partial_cx,
                          double* partial_s, unsigned int* counters, bool* passed, double* rho,
                          double* tau, cudaStream_t stream);
}  // namespace pck

namespace {

using Launcher = cudaError_t (*)(const void*, const void*, float*, int64_t, int64_t, int,
                                 cudaStream_t);

torch::Tensor launch_gemv(const torch::Tensor& W, const torch::Tensor& x, int64_t threads,
                          int64_t min_threads, Launcher launcher) {
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

void check_cuda(const torch::Tensor& t, const char* name, torch::ScalarType dtype,
                const torch::Device& device) {
  TORCH_CHECK(t.is_cuda() && t.device() == device, name, " must be on ", device);
  TORCH_CHECK(t.scalar_type() == dtype, name, " must be ", dtype, ", got ", t.scalar_type());
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> blocked_check(
    const torch::Tensor& c, const torch::Tensor& a, const torch::Tensor& r,
    const torch::Tensor& x, const torch::Tensor& y, double coef, double geps,
    torch::Tensor& partial_cx, torch::Tensor& partial_s, torch::Tensor& counters) {
  TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
  const auto device = y.device();
  check_cuda(c, "c", torch::kFloat32, device);
  check_cuda(a, "a", torch::kFloat32, device);
  check_cuda(r, "r", torch::kInt8, device);
  check_cuda(x, "x", torch::kBFloat16, device);
  check_cuda(y, "y", torch::kFloat32, device);
  TORCH_CHECK(c.dim() == 3 && a.dim() == 2 && r.dim() == 3 && x.dim() == 1 && y.dim() == 1,
              "expected c (K, nb, n), a (nb, n), r (K, nb, b), x (n,), y (m,)");
  const int64_t checks = c.size(0), num_blocks = c.size(1), n = c.size(2), block = r.size(2);
  TORCH_CHECK(a.size(0) == num_blocks && a.size(1) == n, "a must have shape (nb, n)");
  TORCH_CHECK(r.size(0) == checks && r.size(1) == num_blocks, "r must have shape (K, nb, b)");
  TORCH_CHECK(x.size(0) == n, "x must have shape (n,)");
  TORCH_CHECK(y.size(0) == num_blocks * block, "y must have shape (nb * b,)");
  TORCH_CHECK(checks >= 1 && num_blocks >= 1 && n >= 1 && block >= 1, "empty checker");
  TORCH_CHECK(checks * num_blocks <= std::numeric_limits<int32_t>::max(), "too many blocks");
  TORCH_CHECK(coef > 0 && geps > 0, "coef and geps must be positive");
  check_cuda(partial_cx, "partial_cx", torch::kFloat64, device);
  check_cuda(partial_s, "partial_s", torch::kFloat64, device);
  check_cuda(counters, "counters", torch::kInt32, device);
  TORCH_CHECK(partial_cx.dim() == 3 && partial_cx.size(0) == checks &&
                  partial_cx.size(1) == num_blocks && partial_s.sizes() == partial_cx.sizes(),
              "partial sums must have shape (K, nb, segments)");
  TORCH_CHECK(counters.dim() == 2 && counters.size(0) == checks && counters.size(1) == num_blocks,
              "counters must have shape (K, nb)");
  const int64_t segments = partial_cx.size(2);
  TORCH_CHECK(segments >= 1 && segments <= 65535 && segments <= n,
              "segments must be in [1, min(n, 65535)], got ", segments);

  const c10::cuda::CUDAGuard guard(device);
  const auto shape = std::vector<int64_t>{checks, num_blocks};
  auto passed = torch::empty(shape, y.options().dtype(torch::kBool));
  auto rho = torch::empty(shape, y.options().dtype(torch::kFloat64));
  auto tau = torch::empty(shape, y.options().dtype(torch::kFloat64));
  const cudaError_t err = pck::blocked_check(
      c.data_ptr<float>(), a.data_ptr<float>(), r.data_ptr<int8_t>(), x.data_ptr(),
      y.data_ptr<float>(), n, static_cast<int>(block), static_cast<int>(num_blocks),
      static_cast<int>(checks), static_cast<int>(segments), coef, geps,
      partial_cx.data_ptr<double>(), partial_s.data_ptr<double>(),
      reinterpret_cast<unsigned int*>(counters.data_ptr<int32_t>()), passed.data_ptr<bool>(),
      rho.data_ptr<double>(), tau.data_ptr<double>(), c10::cuda::getCurrentCUDAStream().stream());
  TORCH_CHECK(err == cudaSuccess, "checker kernel launch failed: ", cudaGetErrorString(err));
  return {passed, rho, tau};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemv_naive", &gemv_naive, "Naive BF16 GEMV with FP32 accumulation (y in FP32)",
        pybind11::arg("W"), pybind11::arg("x"), pybind11::arg("threads"));
  m.def("gemv_coalesced",
        &gemv_coalesced, "Coalesced BF16 GEMV with 16-byte loads and warp-shuffle reduction",
        pybind11::arg("W"), pybind11::arg("x"), pybind11::arg("threads"));
  m.def("blocked_check", &blocked_check,
        "Trusted blocked checksum checker; returns (passed, rho, tau), each (K, nb)",
        pybind11::arg("c"), pybind11::arg("a"), pybind11::arg("r"), pybind11::arg("x"),
        pybind11::arg("y"), pybind11::arg("coef"), pybind11::arg("geps"),
        pybind11::arg("partial_cx"), pybind11::arg("partial_s"), pybind11::arg("counters"));
}
