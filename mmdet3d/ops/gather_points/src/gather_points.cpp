#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include <vector>

extern THCState *state;

int gather_points_wrapper(int b, int c, int n, int npoints,
                          at::Tensor points_tensor, at::Tensor idx_tensor,
                          at::Tensor out_tensor);

void gather_points_kernel_launcher(int b, int c, int n, int npoints,
                                   const float *points, const int *idx,
                                   float *out, cudaStream_t stream);

int gather_points_grad_wrapper(int b, int c, int n, int npoints,
                               at::Tensor grad_out_tensor,
                               at::Tensor idx_tensor,
                               at::Tensor grad_points_tensor);

void gather_points_grad_kernel_launcher(int b, int c, int n, int npoints,
                                        const float *grad_out, const int *idx,
                                        float *grad_points,
                                        cudaStream_t stream);

int gather_points_wrapper(int b, int c, int n, int npoints,
                          at::Tensor points_tensor, at::Tensor idx_tensor,
                          at::Tensor out_tensor) {
  const float *points = points_tensor.data_ptr<float>();
  const int *idx = idx_tensor.data_ptr<int>();
  float *out = out_tensor.data_ptr<float>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  gather_points_kernel_launcher(b, c, n, npoints, points, idx, out, stream);
  return 1;
}

int gather_points_grad_wrapper(int b, int c, int n, int npoints,
                               at::Tensor grad_out_tensor,
                               at::Tensor idx_tensor,
                               at::Tensor grad_points_tensor) {
  const float *grad_out = grad_out_tensor.data_ptr<float>();
  const int *idx = idx_tensor.data_ptr<int>();
  float *grad_points = grad_points_tensor.data_ptr<float>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  gather_points_grad_kernel_launcher(b, c, n, npoints, grad_out, idx,
                                     grad_points, stream);
  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather_points_wrapper", &gather_points_wrapper,
        "gather_points_wrapper");
  m.def("gather_points_grad_wrapper", &gather_points_grad_wrapper,
        "gather_points_grad_wrapper");
}
