# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS
from mmdet3d.structures.bbox_3d import rotation_3d_in_axis


@MODELS.register_module()
class Batch3DRoIGridExtractor(BaseModule):
    """Grid point wise roi-aware Extractor.

    Extract grid point wise roi features.

    Args:
        grid_size (int): The number of grid points in roi bbox.
        sa_module_cfg (dict): Config of sa module to get grid points features.
            Defaults to None.
        init_cfg (dict, optional): Initialize config of
            model. Defaults to None.
    """

    def __init__(self,
                 grid_size: int = 6,
                 sa_module_cfg: dict = None,
                 init_cfg: dict = None) -> None:
        super(Batch3DRoIGridExtractor, self).__init__(init_cfg=init_cfg)
        self.roi_grid_pool_layer = MODELS.build(sa_module_cfg)
        self.grid_size = grid_size

    def forward(self, feats: torch.Tensor, coordinate: torch.Tensor,
                batch_inds: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats (torch.Tensor): Key points features.
            coordinate (torch.Tensor): Key points coordinates.
            batch_inds (torch.Tensor): Input batch indexes.
            rois (torch.Tensor): Detection results of rpn head.

        Returns:
            torch.Tensor: Grid points features.
        """
        batch_size = int(batch_inds.max()) + 1

        xyz = coordinate
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_inds == k).sum()

        rois_batch_inds = rois[:, 0].int()
        # (N1+N2+..., 6x6x6, 3)
        roi_grid = self.get_dense_grid_points(rois[:, 1:])

        new_xyz = roi_grid.view(-1, 3)
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int()
        for k in range(batch_size):
            new_xyz_batch_cnt[k] = ((rois_batch_inds == k).sum() *
                                    roi_grid.size(1))
        pooled_points, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz.contiguous(),
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=feats.contiguous())  # (M1 + M2 ..., C)

        pooled_features = pooled_features.view(-1, self.grid_size,
                                               self.grid_size, self.grid_size,
                                               pooled_features.shape[-1])
        # (BxN, 6, 6, 6, C)
        return pooled_features

    def get_dense_grid_points(self, rois: torch.Tensor) -> torch.Tensor:
        """Get dense grid points from rois.

        Args:
            rois (torch.Tensor): Detection results of rpn head.

        Returns:
            torch.Tensor: Grid points coordinates.
        """
        rois[:, 2] += rois[:, 5] / 2
        faked_features = rois.new_ones(
            (self.grid_size, self.grid_size, self.grid_size))
        dense_idx = faked_features.nonzero()
        dense_idx = dense_idx.repeat(rois.size(0), 1, 1).float()
        dense_idx = ((dense_idx + 0.5) / self.grid_size)
        dense_idx[..., :3] -= 0.5

        roi_ctr = rois[:, :3]
        roi_dim = rois[:, 3:6]
        roi_grid_points = dense_idx * roi_dim.view(-1, 1, 3)
        roi_grid_points = rotation_3d_in_axis(
            roi_grid_points, rois[:, 6], axis=2)
        roi_grid_points += roi_ctr.view(-1, 1, 3)

        return roi_grid_points
