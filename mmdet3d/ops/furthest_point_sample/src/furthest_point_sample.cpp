// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/sampling.cpp

#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include <vector>

extern THCState *state;

int furthest_point_sampling_wrapper(int b, int n, int m,
                                    at::Tensor points_tensor,
                                    at::Tensor temp_tensor,
                                    at::Tensor idx_tensor);

void furthest_point_sampling_kernel_launcher(int b, int n, int m,
                                             const float *dataset, float *temp,
                                             int *idxs, cudaStream_t stream);

int furthest_point_sampling_with_dist_wrapper(int b, int n, int m,
                                              at::Tensor points_tensor,
                                              at::Tensor temp_tensor,
                                              at::Tensor idx_tensor);

void furthest_point_sampling_with_dist_kernel_launcher(int b, int n, int m,
                                                       const float *dataset,
                                                       float *temp, int *idxs,
                                                       cudaStream_t stream);

int furthest_point_sampling_wrapper(int b, int n, int m,
                                    at::Tensor points_tensor,
                                    at::Tensor temp_tensor,
                                    at::Tensor idx_tensor) {
  const float *points = points_tensor.data_ptr<float>();
  float *temp = temp_tensor.data_ptr<float>();
  int *idx = idx_tensor.data_ptr<int>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  furthest_point_sampling_kernel_launcher(b, n, m, points, temp, idx, stream);
  return 1;
}

int furthest_point_sampling_with_dist_wrapper(int b, int n, int m,
                                              at::Tensor points_tensor,
                                              at::Tensor temp_tensor,
                                              at::Tensor idx_tensor) {

  const float *points = points_tensor.data<float>();
  float *temp = temp_tensor.data<float>();
  int *idx = idx_tensor.data<int>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  furthest_point_sampling_with_dist_kernel_launcher(b, n, m, points, temp, idx, stream);
  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("furthest_point_sampling_wrapper", &furthest_point_sampling_wrapper,
        "furthest_point_sampling_wrapper");
  m.def("furthest_point_sampling_with_dist_wrapper",
        &furthest_point_sampling_with_dist_wrapper,
        "furthest_point_sampling_with_dist_wrapper");
}
