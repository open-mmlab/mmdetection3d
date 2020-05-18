#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <THC/THC.h>
#include <torch/extension.h>

extern THCState *state;

int furthest_point_sampling_wrapper(int b, int n, int m,
    at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor);

void furthest_point_sampling_kernel_launcher(int b, int n, int m,
    const float *dataset, float *temp, int *idxs, cudaStream_t stream);


int furthest_point_sampling_wrapper(int b, int n, int m,
    at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor) {

    const float *points = points_tensor.data<float>();
    float *temp = temp_tensor.data<float>();
    int *idx = idx_tensor.data<int>();

    cudaStream_t stream = THCState_getCurrentStream(state);
    furthest_point_sampling_kernel_launcher(b, n, m, points, temp, idx, stream);
    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("furthest_point_sampling_wrapper", &furthest_point_sampling_wrapper, "furthest_point_sampling_wrapper");
}
