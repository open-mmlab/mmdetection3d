# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class Voxel2PointScatterNeck(nn.Module):
    """
    A memory-efficient voxel2point with torch_scatter
    """

    def __init__(
        self,
        point_cloud_range=None,
        voxel_size=None,
        with_xyz=True,
        normalize_local_xyz=False,
        ):
        super().__init__()
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.with_xyz = with_xyz
        self.normalize_local_xyz = normalize_local_xyz

    def forward(self, points, pts_coors, voxel_feats, voxel2point_inds, voxel_padding=-1):
        """Forward function.

        Args:
            points (torch.Tensor): of shape (N, C_point).
            pts_coors (torch.Tensor): of shape (N, 4).
            voxel_feats (torch.Tensor): of shape (M, C_feature), should be padded and reordered.
            voxel2point_inds: (N,)

        Returns:
            torch.Tensor: of shape (N, C_feature+C_point).
        """
        assert points.size(0) == pts_coors.size(0) == voxel2point_inds.size(-1)
        dtype = voxel_feats.dtype
        device = voxel_feats.device
        pts_feats = voxel_feats[voxel2point_inds] # voxel_feats must be the output of torch_scatter, voxel2point_inds is the input of torch_scatter
        pts_mask = ~((pts_feats == voxel_padding).all(1)) # some dropped voxels are padded
        if self.with_xyz:
            pts_feats = pts_feats[pts_mask]
            pts_coors = pts_coors[pts_mask]
            points = points[pts_mask]

            voxel_size = torch.tensor(self.voxel_size, dtype=dtype, device=device).reshape(1,3)
            pc_min_range = torch.tensor(self.point_cloud_range[:3], dtype=dtype, device=device).reshape(1,3)
            voxel_center_each_pts = (pts_coors[:, [3,2,1]].to(dtype).to(device) + 0.5) * voxel_size + pc_min_range# x y z order
            local_xyz = points[:, :3] - voxel_center_each_pts
            if self.normalize_local_xyz:
                local_xyz = local_xyz / (voxel_size / 2)

            if self.training and not self.normalize_local_xyz:
                assert (local_xyz.abs() < voxel_size / 2 + 1e-3).all(), 'Holds in training. However, in test, this is not always True because of lack of point range clip'
            results = torch.cat([pts_feats, local_xyz], 1)
        else:
            results = pts_feats[pts_mask]
        
        return results, pts_mask