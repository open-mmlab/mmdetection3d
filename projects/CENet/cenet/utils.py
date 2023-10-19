# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mmdet3d.utils import ConfigType


def get_gaussian_kernel(kernel_size: int = 3, sigma: int = 2) -> Tensor:
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(-torch.sum(
        (xy_grid - mean)**2., dim=-1) / (2 * variance))

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(kernel_size, kernel_size)

    return gaussian_kernel


class KNN(nn.Module):

    def __init__(self, test_cfg: ConfigType, num_classes: int,
                 ignore_index: int) -> None:
        super(KNN, self).__init__()
        self.knn = test_cfg.knn
        self.search = test_cfg.search
        self.sigma = test_cfg.sigma
        self.cutoff = test_cfg.cutoff
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, proj_range: Tensor, unproj_range: Tensor,
                proj_argmax: Tensor, px: Tensor, py: Tensor) -> Tensor:

        # sizes of projection scan
        H, W = proj_range.shape

        # number of points
        P = unproj_range.shape

        # check if size of kernel is odd and complain
        if self.search % 2 == 0:
            raise ValueError('Nearest neighbor kernel must be odd number')

        # calculate padding
        pad = int((self.search - 1) / 2)

        # unfold neighborhood to get nearest neighbors for each pixel
        # (range image)
        proj_unfold_k_rang = F.unfold(
            proj_range[None, None, ...],
            kernel_size=(self.search, self.search),
            padding=(pad, pad))

        # index with px, py to get ALL the pcld points
        idx_list = py * W + px
        unproj_unfold_k_rang = proj_unfold_k_rang[:, :, idx_list]

        # WARNING, THIS IS A HACK
        # Make non valid (<0) range points extremely big so that there is no
        # screwing up the nn self.search
        unproj_unfold_k_rang[unproj_unfold_k_rang < 0] = float('inf')

        # now the matrix is unfolded TOTALLY, replace the middle points with
        # the actual range points
        center = int(((self.search * self.search) - 1) / 2)
        unproj_unfold_k_rang[:, center, :] = unproj_range

        # now compare range
        k2_distances = torch.abs(unproj_unfold_k_rang - unproj_range)

        # make a kernel to weigh the ranges according to distance in (x,y)
        # I make this 1 - kernel because I want distances that are close
        # in (x,y) to matter more
        inv_gauss_k = (1 - get_gaussian_kernel(self.search, self.sigma)).view(
            1, -1, 1)
        inv_gauss_k = inv_gauss_k.to(proj_range.device).type(proj_range.type())

        # apply weighing
        k2_distances = k2_distances * inv_gauss_k

        # find nearest neighbors
        _, knn_idx = k2_distances.topk(
            self.knn, dim=1, largest=False, sorted=False)

        # do the same unfolding with the argmax
        proj_unfold_1_argmax = F.unfold(
            proj_argmax[None, None, ...].float(),
            kernel_size=(self.search, self.search),
            padding=(pad, pad)).long()
        unproj_unfold_1_argmax = proj_unfold_1_argmax[:, :, idx_list]

        # get the top k logits from the knn at each pixel
        knn_argmax = torch.gather(
            input=unproj_unfold_1_argmax, dim=1, index=knn_idx)

        # fake an invalid argmax of classes + 1 for all cutoff items
        if self.cutoff > 0:
            knn_distances = torch.gather(
                input=k2_distances, dim=1, index=knn_idx)
            knn_invalid_idx = knn_distances > self.cutoff
            knn_argmax[knn_invalid_idx] = self.num_classes

        # now vote
        # argmax onehot has an extra class for objects after cutoff
        knn_argmax_onehot = torch.zeros(
            (1, self.num_classes + 1, P[0]),
            device=proj_range.device).type(proj_range.type())
        ones = torch.ones_like(knn_argmax).type(proj_range.type())
        knn_argmax_onehot = knn_argmax_onehot.scatter_add_(1, knn_argmax, ones)

        # now vote (as a sum over the onehot shit)
        # (don't let it choose unlabeled OR invalid)
        if self.ignore_index == self.num_classes - 1:
            knn_argmax_out = knn_argmax_onehot[:, :-2].argmax(dim=1)
        elif self.ignore_index == 0:
            knn_argmax_out = knn_argmax_onehot[:, 1:-1].argmax(dim=1) + 1
        else:
            knn_argmax_out = knn_argmax_onehot[:, :-1].argmax(dim=1)

        # reshape again
        knn_argmax_out = knn_argmax_out.view(P)

        return knn_argmax_out
