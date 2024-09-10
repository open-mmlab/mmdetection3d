# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.registry import MODELS
from torch import Tensor
from torch import distributed as dist
from torch import nn as nn
from torch.autograd.function import Function


class AllReduce(Function):

    @staticmethod
    def forward(ctx, input: Tensor) -> Tensor:
        input_list = [
            torch.zeros_like(input) for k in range(dist.get_world_size())
        ]
        # Use allgather instead of allreduce in-place operations is unreliable
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        dist.all_reduce(grad_output, async_op=False)
        return grad_output


@MODELS.register_module('naiveSyncBN1d')
class NaiveSyncBatchNorm1d(nn.BatchNorm1d):
    """Synchronized Batch Normalization for 3D Tensors.

    Note:
        This implementation is modified from
        https://github.com/facebookresearch/detectron2/

        `torch.nn.SyncBatchNorm` has known unknown bugs.
        It produces significantly worse AP (and sometimes goes NaN)
        when the batch size on each worker is quite different
        (e.g., when scale augmentation is used).
        In 3D detection, different workers has points of different shapes,
        which also cause instability.

        Use this implementation before `nn.SyncBatchNorm` is fixed.
        It is slower than `nn.SyncBatchNorm`.
    """

    def __init__(self, *args: list, **kwargs: dict) -> None:
        super(NaiveSyncBatchNorm1d, self).__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor): Has shape (N, C) or (N, C, L), where N is
                the batch size, C is the number of features or
                channels, and L is the sequence length

        Returns:
            Tensor: Has shape (N, C) or (N, C, L), same shape as input.
        """
        using_dist = dist.is_available() and dist.is_initialized()
        if (not using_dist) or dist.get_world_size() == 1 \
                or not self.training:
            return super().forward(input)
        assert input.shape[0] > 0, 'SyncBN does not support empty inputs'
        is_two_dim = input.dim() == 2
        if is_two_dim:
            input = input.unsqueeze(2)

        C = input.shape[1]
        mean = torch.mean(input, dim=[0, 2])
        meansqr = torch.mean(input * input, dim=[0, 2])

        vec = torch.cat([mean, meansqr], dim=0)
        vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())

        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        self.running_mean += self.momentum * (
            mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1)
        bias = bias.reshape(1, -1, 1)
        output = input * scale + bias
        if is_two_dim:
            output = output.squeeze(2)
        return output


@MODELS.register_module('naiveSyncBN2d')
class NaiveSyncBatchNorm2d(nn.BatchNorm2d):
    """Synchronized Batch Normalization for 4D Tensors.

    Note:
        This implementation is modified from
        https://github.com/facebookresearch/detectron2/

        `torch.nn.SyncBatchNorm` has known unknown bugs.
        It produces significantly worse AP (and sometimes goes NaN)
        when the batch size on each worker is quite different
        (e.g., when scale augmentation is used).
        This phenomenon also occurs when the multi-modality feature fusion
        modules of multi-modality detectors use SyncBN.

        Use this implementation before `nn.SyncBatchNorm` is fixed.
        It is slower than `nn.SyncBatchNorm`.
    """

    def __init__(self, *args: list, **kwargs: dict) -> None:
        super(NaiveSyncBatchNorm2d, self).__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            Input (Tensor): Feature has shape (N, C, H, W).

        Returns:
            Tensor: Has shape (N, C, H, W), same shape as input.
        """
        assert input.dtype == torch.float32, \
            f'input should be in float32 type, got {input.dtype}'
        using_dist = dist.is_available() and dist.is_initialized()
        if (not using_dist) or \
                dist.get_world_size() == 1 or \
                not self.training:
            return super().forward(input)

        assert input.shape[0] > 0, 'SyncBN does not support empty inputs'
        C = input.shape[1]
        mean = torch.mean(input, dim=[0, 2, 3])
        meansqr = torch.mean(input * input, dim=[0, 2, 3])

        vec = torch.cat([mean, meansqr], dim=0)
        vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())

        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        self.running_mean += self.momentum * (
            mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return input * scale + bias
