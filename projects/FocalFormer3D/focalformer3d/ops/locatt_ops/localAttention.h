//
// Created by zhang on 20-1-14.
//

#ifndef LOCALATTENTION_LOCALATTENTION_H
#define LOCALATTENTION_LOCALATTENTION_H

#pragma once
#include <torch/extension.h>

torch::Tensor similar_cuda_forward(
        const torch::Tensor &x_ori,
        const torch::Tensor &x_loc,
        const int kH,
        const int kW);

torch::Tensor similar_cuda_backward(
        const torch::Tensor &x,
        const torch::Tensor &grad_out,
        const int kH,
        const int kW,
        const bool is_ori);

torch::Tensor weighting_cuda_forward(
        const torch::Tensor &x_ori,
        const torch::Tensor &x_weight,
        const int kH,
        const int kW);

torch::Tensor weighting_cuda_backward_ori(
        const torch::Tensor &x_weight,
        const torch::Tensor &grad_out,
        const int kH,
        const int kW);

torch::Tensor weighting_cuda_backward_weight(
        const torch::Tensor &x_ori,
        const torch::Tensor &grad_out,
        const int kH,
        const int kW);

#endif //LOCALATTENTION_LOCALATTENTION_H
