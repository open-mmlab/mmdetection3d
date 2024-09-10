# modified from https://github.com/Haiyang-W/DSVT
import torch
import torch.nn as nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class PointPillarsScatter3D(nn.Module):
    """The difference between `PointPillarsScatter3D` and `PointPillarsScatter`
    is that the voxel in this module is along 3 dims: (x, y, z)."""

    def __init__(self, output_shape, num_bev_feats, **kwargs):
        super().__init__()
        self.nx, self.ny, self.nz = output_shape
        self.num_bev_feats = num_bev_feats
        self.num_bev_feats_ori = num_bev_feats // self.nz

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict[
            'voxel_coords']

        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_feats_ori,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] * self.ny * self.nx + \
                this_coords[:, 2] * self.nx + this_coords[:,  3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(
            batch_size, self.num_bev_feats_ori * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict
