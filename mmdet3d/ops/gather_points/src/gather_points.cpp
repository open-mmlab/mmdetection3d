#include <ATen/cuda/CUDAContext.h>
#include <ATen/TensorUtils.h>
#include <THC/THC.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include <vector>


extern THCState *state;

int gather_points_wrapper(int b, int c, int n, int npoints,
                          at::Tensor& points_tensor, at::Tensor& idx_tensor,
                          at::Tensor& out_tensor);

void gather_points_kernel_launcher(int b, int c, int n, int npoints,
                                   const at::Tensor& points_tensor,
                                   const at::Tensor& idx_tensor,
                                   at::Tensor& out_tensor);

int gather_points_grad_wrapper(int b, int c, int n, int npoints,
                               at::Tensor& grad_out_tensor,
                               at::Tensor& idx_tensor,
                               at::Tensor& grad_points_tensor);

void gather_points_grad_kernel_launcher(int b, int c, int n, int npoints,
                                        const at::Tensor& grad_out_tensor,
                                        const at::Tensor& idx_tensor,
                                        at::Tensor& grad_points_tensor);

int gather_points_wrapper(int b, int c, int n, int npoints,
                          at::Tensor& points_tensor, at::Tensor& idx_tensor,
                          at::Tensor& out_tensor)
{
  gather_points_kernel_launcher(b, c, n, npoints, points_tensor, idx_tensor, out_tensor);
  return 1;
}

int gather_points_grad_wrapper(int b, int c, int n, int npoints,
                               at::Tensor& grad_out_tensor,
                               at::Tensor& idx_tensor,
                               at::Tensor& grad_points_tensor)
{
  gather_points_grad_kernel_launcher(b, c, n, npoints, grad_out_tensor, idx_tensor,
                                     grad_points_tensor);
  return 1;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("gather_points_wrapper", &gather_points_wrapper,
        "gather_points_wrapper");
  m.def("gather_points_grad_wrapper", &gather_points_grad_wrapper,
        "gather_points_grad_wrapper");
}
