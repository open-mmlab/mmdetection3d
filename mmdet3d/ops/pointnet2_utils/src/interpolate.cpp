#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "interpolate_gpu.h"

extern THCState *state;

void three_nn_wrapper_fast(int n, int m, at::Tensor unknown_tensor,
    at::Tensor known_tensor, at::Tensor dist2_tensor, at::Tensor idx_tensor) {
    const float *unknown = unknown_tensor.data<float>();
    const float *known = known_tensor.data<float>();
    float *dist2 = dist2_tensor.data<float>();
    int *idx = idx_tensor.data<int>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();;
    three_nn_kernel_launcher_fast(n, m, unknown, known, dist2, idx, stream);
}


void three_interpolate_wrapper_fast(int c, int m, int n,
                         at::Tensor points_tensor,
                         at::Tensor idx_tensor,
                         at::Tensor weight_tensor,
                         at::Tensor out_tensor) {

    const float *points = points_tensor.data<float>();
    const float *weight = weight_tensor.data<float>();
    float *out = out_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();;
    three_interpolate_kernel_launcher_fast(c, m, n, points, idx, weight, out, stream);
}

void three_interpolate_grad_wrapper_fast(int c, int n, int m,
                            at::Tensor grad_out_tensor,
                            at::Tensor idx_tensor,
                            at::Tensor weight_tensor,
                            at::Tensor grad_points_tensor) {

    const float *grad_out = grad_out_tensor.data<float>();
    const float *weight = weight_tensor.data<float>();
    float *grad_points = grad_points_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();;
    three_interpolate_grad_kernel_launcher_fast(c, n, m, grad_out, idx, weight, grad_points, stream);
}
