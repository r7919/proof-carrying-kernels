// Bindings for the diagnostic batched GEMV variants (timing only, not trusted code).

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cstdint>

namespace pckb {
cudaError_t batched_gemv(int mode, const void* W, const void* X, float* Y, int64_t m,
                         int64_t n, int64_t batch, int threads, cudaStream_t s);
}  // namespace pckb

namespace {

torch::Tensor run(int64_t mode, const torch::Tensor& W, const torch::Tensor& X,
                  int64_t threads) {
  TORCH_CHECK(W.is_cuda() && X.is_cuda() && W.scalar_type() == torch::kBFloat16 &&
                  X.scalar_type() == torch::kBFloat16,
              "W and X must be BF16 CUDA tensors");
  TORCH_CHECK(W.dim() == 2 && X.dim() == 2 && X.size(1) == W.size(1), "shape mismatch");
  TORCH_CHECK(W.is_contiguous() && X.is_contiguous(), "inputs must be contiguous");
  const c10::cuda::CUDAGuard guard(W.device());
  auto Y = torch::empty({X.size(0), W.size(0)}, W.options().dtype(torch::kFloat32));
  const cudaError_t err =
      pckb::batched_gemv(static_cast<int>(mode), W.data_ptr(), X.data_ptr(), Y.data_ptr<float>(),
                         W.size(0), W.size(1), X.size(0), static_cast<int>(threads),
                         c10::cuda::getCurrentCUDAStream().stream());
  TORCH_CHECK(err == cudaSuccess, "batched gemv failed: ", cudaGetErrorString(err));
  return Y;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "Diagnostic batched GEMV variant", pybind11::arg("mode"),
        pybind11::arg("W"), pybind11::arg("X"), pybind11::arg("threads"));
}
