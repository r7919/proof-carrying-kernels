// PyTorch bindings for the GEMV kernels.

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>

namespace pck {
cudaError_t gemv_bf16_naive(const void* W, const void* x, float* y, int64_t m, int64_t n,
                            int threads, cudaStream_t stream);
cudaError_t gemv_bf16_coalesced(const void* W, const void* x, float* y, int64_t m, int64_t n,
                                int threads, cudaStream_t stream);
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

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemv_naive", &gemv_naive, "Naive BF16 GEMV with FP32 accumulation (y in FP32)",
        pybind11::arg("W"), pybind11::arg("x"), pybind11::arg("threads"));
  m.def("gemv_coalesced",
        &gemv_coalesced, "Coalesced BF16 GEMV with 16-byte loads and warp-shuffle reduction",
        pybind11::arg("W"), pybind11::arg("x"), pybind11::arg("threads"));
}
