#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include <vector>

extern THCState *state;

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int feat_distance_wrapper(int b, int n, int m, int c, at::Tensor feat_a_tensor,
                          at::Tensor feat_b_tensor, at::Tensor distance_tensor);

void feat_distance_kernel_launcher(int b, int n, int m, int c,
                                   const float *feat_a, const float *feat_b,
                                   float *distance, cudaStream_t stream);

int feat_distance_wrapper(int b, int n, int m, int c, at::Tensor feat_a_tensor,
                          at::Tensor feat_b_tensor,
                          at::Tensor distance_tensor) {
  CHECK_INPUT(feat_a_tensor);
  CHECK_INPUT(feat_b_tensor);
  const float *feat_a = feat_a_tensor.data_ptr<float>();
  const float *feat_b = feat_b_tensor.data_ptr<float>();
  float *distance = distance_tensor.data_ptr<float>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  feat_distance_kernel_launcher(b, n, m, c, feat_a, feat_b, distance, stream);
  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("feat_distance_wrapper", &feat_distance_wrapper,
        "feat_distance_wrapper");
}
