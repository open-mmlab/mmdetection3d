# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.models import Det3DDataPreprocessor
from mmdet3d.models.data_preprocessors.voxelize import dynamic_scatter_3d
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList


@MODELS.register_module()
class TPVFormerDataPreprocessor(Det3DDataPreprocessor):

    @torch.no_grad()
    def voxelize(self, points: List[Tensor],
                 data_samples: SampleList) -> List[Tensor]:
        """Apply voxelization to point cloud. In TPVFormer, it will get voxel-
        wise segmentation label and voxel/point coordinates.

        Args:
            points (List[Tensor]): Point cloud in one data batch.
            data_samples: (List[:obj:`Det3DDataSample`]): The annotation data
                of every samples. Add voxel-wise annotation for segmentation.

        Returns:
            List[Tensor]: Coordinates of voxels, shape is Nx3,
        """
        for point, data_sample in zip(points, data_samples):
            min_bound = point.new_tensor(
                self.voxel_layer.point_cloud_range[:3])
            max_bound = point.new_tensor(
                self.voxel_layer.point_cloud_range[3:])
            point_clamp = torch.clamp(point, min_bound, max_bound + 1e-6)
            coors = torch.floor(
                (point_clamp - min_bound) /
                point_clamp.new_tensor(self.voxel_layer.voxel_size)).int()
            self.get_voxel_seg(coors, data_sample)
            data_sample.point_coors = coors

    def get_voxel_seg(self, res_coors: Tensor, data_sample: SampleList):
        """Get voxel-wise segmentation label and point2voxel map.

        Args:
            res_coors (Tensor): The voxel coordinates of points, Nx3.
            data_sample: (:obj:`Det3DDataSample`): The annotation data of
                every samples. Add voxel-wise annotation forsegmentation.
        """

        if self.training:
            pts_semantic_mask = data_sample.gt_pts_seg.pts_semantic_mask
            pts_semantic_mask = F.one_hot(pts_semantic_mask.long()).float()
            voxel_semantic_mask, voxel_coors, point2voxel_map = \
                dynamic_scatter_3d(pts_semantic_mask, res_coors, 'mean', True)
            voxel_semantic_mask = torch.argmax(voxel_semantic_mask, dim=-1)
            data_sample.gt_pts_seg.voxel_semantic_mask = voxel_semantic_mask
            data_sample.point2voxel_map = point2voxel_map
            data_sample.voxel_coors = voxel_coors
        else:
            pseudo_tensor = res_coors.new_ones([res_coors.shape[0], 1]).float()
            _, _, point2voxel_map = dynamic_scatter_3d(pseudo_tensor,
                                                       res_coors, 'mean', True)
            data_sample.point2voxel_map = point2voxel_map


@MODELS.register_module()
class GridMask(nn.Module):
    """GridMask data augmentation.

        Modified from https://github.com/dvlab-research/GridMask.

    Args:
        use_h (bool): Whether to mask on height dimension. Defaults to True.
        use_w (bool): Whether to mask on width dimension. Defaults to True.
        rotate (int): Rotation degree. Defaults to 1.
        offset (bool): Whether to mask offset. Defaults to False.
        ratio (float): Mask ratio. Defaults to 0.5.
        mode (int): Mask mode. if mode == 0, mask with square grid.
            if mode == 1, mask the rest. Defaults to 0
        prob (float): Probability of applying the augmentation.
            Defaults to 1.0.
    """

    def __init__(self,
                 use_h: bool = True,
                 use_w: bool = True,
                 rotate: int = 1,
                 offset: bool = False,
                 ratio: float = 0.5,
                 mode: int = 0,
                 prob: float = 1.0):
        super().__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.prob = prob

    def forward(self, inputs: Tensor,
                data_samples: SampleList) -> Tuple[Tensor, SampleList]:
        if np.random.rand() > self.prob:
            return inputs, data_samples
        height, width = inputs.shape[-2:]
        mask_height = int(1.5 * height)
        mask_width = int(1.5 * width)
        distance = np.random.randint(2, min(height, width))
        length = min(max(int(distance * self.ratio + 0.5), 1), distance - 1)
        mask = np.ones((mask_height, mask_width), np.float32)
        stride_on_height = np.random.randint(distance)
        stride_on_width = np.random.randint(distance)
        if self.use_h:
            for i in range(mask_height // distance):
                start = distance * i + stride_on_height
                end = min(start + length, mask_height)
                mask[start:end, :] *= 0
        if self.use_w:
            for i in range(mask_width // distance):
                start = distance * i + stride_on_width
                end = min(start + length, mask_width)
                mask[:, start:end] *= 0

        # NOTE: r is the rotation radian, here is a random counterclockwise
        # rotation of 1° or remain unchanged, which follows the implementation
        # of the official detection version.
        # https://github.com/dvlab-research/GridMask.
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))

        mask = mask.rotate(r)
        mask = np.array(mask)
        mask = mask[int(0.25 * height):int(0.25 * height) + height,
                    int(0.25 * width):int(0.25 * width) + width]

        mask = inputs.new_tensor(mask)
        if self.mode == 1:
            mask = 1 - mask
        mask = mask.expand_as(inputs)
        if self.offset:
            offset = inputs.new_tensor(2 *
                                       (np.random.rand(height, width) - 0.5))
            inputs = inputs * mask + offset * (1 - mask)
        else:
            inputs = inputs * mask

        return inputs, data_samples
