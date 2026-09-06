//
// Created by zhang on 20-1-14.
//

#include "localAttention.h"

torch::Tensor similar_forward(
        const torch::Tensor &x_ori,
        const torch::Tensor &x_loc,
        const int kH,
        const int kW) {
    return similar_cuda_forward(
            x_ori, x_loc,
            kH, kW);
}

torch::Tensor similar_backward(
        const torch::Tensor &x,
        const torch::Tensor &grad_out,
        const int kH,
        const int kW,
        const bool is_ori) {
    return similar_cuda_backward(
            x, grad_out,
            kH, kW,
            is_ori);
}


torch::Tensor weighting_forward(
        const torch::Tensor &x_ori,
        const torch::Tensor &x_weight,
        const int kH,
        const int kW) {
    return weighting_cuda_forward(
            x_ori, x_weight,
            kH, kW);
}

torch::Tensor weighting_backward_ori(
        const torch::Tensor &x_weight,
        const torch::Tensor &grad_out,
        const int kH,
        const int kW) {
    return weighting_cuda_backward_ori(
            x_weight, grad_out,
            kH, kW);
}

torch::Tensor weighting_backward_weight(
        const torch::Tensor &x_ori,
        const torch::Tensor &grad_out,
        const int kH,
        const int kW) {
    return weighting_cuda_backward_weight(
            x_ori, grad_out,
            kH, kW);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("similar_forward", &similar_forward,
            "similar_forward (CUDA)");
    m.def("similar_backward", &similar_backward,
            "similar_backward (CUDA)");

    m.def("weighting_forward", &weighting_forward,
            "weighting_forward (CUDA)");
    m.def("weighting_backward_ori", &weighting_backward_ori,
            "weighting_backward_ori (CUDA)");
    m.def("weighting_backward_weight", &weighting_backward_weight,
            "weighting_backward_weight (CUDA)");
}
