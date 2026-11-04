// Bindings for the diagnostic checker variants (timing only, not trusted code).

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cstdint>

namespace pckv {
cudaError_t run_variant(int variant, const float* c, const void* a, const void* x, int64_t n,
                        int num_blocks, int checks, int segments, double* out,
                        cudaStream_t stream);
}  // namespace pckv

namespace {

void run(int64_t variant, const torch::Tensor& c, const torch::Tensor& a,
         const torch::Tensor& x, torch::Tensor& out) {
  TORCH_CHECK(c.is_cuda() && c.scalar_type() == torch::kFloat32 && c.dim() == 3, "bad c");
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == torch::kBFloat16 && a.dim() == 2, "bad a");
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.dim() == 1, "bad x");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat64 && out.dim() == 3,
              "bad out");
  TORCH_CHECK(c.is_contiguous() && a.is_contiguous() && x.is_contiguous() && out.is_contiguous(),
              "inputs must be contiguous");
  const int64_t checks = c.size(0), num_blocks = c.size(1), n = c.size(2);
  TORCH_CHECK(a.size(0) == num_blocks && a.size(1) == n && x.size(0) == n, "shape mismatch");
  TORCH_CHECK(out.size(0) == checks && out.size(1) == num_blocks, "out must be (K, nb, S)");
  const c10::cuda::CUDAGuard guard(c.device());
  const cudaError_t err = pckv::run_variant(
      static_cast<int>(variant), c.data_ptr<float>(), a.data_ptr(), x.data_ptr(), n,
      static_cast<int>(num_blocks), static_cast<int>(checks), static_cast<int>(out.size(2)),
      out.data_ptr<double>(), c10::cuda::getCurrentCUDAStream().stream());
  TORCH_CHECK(err == cudaSuccess, "variant launch failed: ", cudaGetErrorString(err));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "Run checker phase-1 variant", pybind11::arg("variant"),
        pybind11::arg("c"), pybind11::arg("a"), pybind11::arg("x"), pybind11::arg("out"));
}
