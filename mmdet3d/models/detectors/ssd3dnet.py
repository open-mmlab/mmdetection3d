import numpy as np
import torch

from mmdet3d.ops import Voxelization
from mmdet.models import DETECTORS
from .votenet import VoteNet


@DETECTORS.register_module()
class SSD3DNet(VoteNet):
    """3DSSDNet model.

    https://arxiv.org/abs/2002.10187.pdf
    """

    def __init__(self,
                 backbone,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SSD3DNet, self).__init__(
            backbone=backbone,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        if 'use_voxel_sample' in self.test_cfg.keys() and \
                self.test_cfg.use_voxel_sample:
            self.cur_sweep_sampler = Voxelization(
                **test_cfg.voxel_sampler_cfg[0])
            self.prev_sweep_sampler = Voxelization(
                **test_cfg.voxel_sampler_cfg[1])

    def _points_voxel_sampling(self, cur_points, sweeps_points):
        voxels, coors, num_points_per_voxel = self.cur_sweep_sampler(
            cur_points)
        max_voxels = self.test_cfg.voxel_sampler_cfg[0].max_voxels
        if voxels.shape[0] >= max_voxels:
            cur_points = voxels[:max_voxels]
        else:
            choices = np.random.choice(
                voxels.shape[0], max_voxels, replace=True)
            cur_points = voxels[choices]
        voxels, coors, num_points_per_voxel = self.prev_sweep_sampler(
            sweeps_points)

        max_voxels = self.test_cfg.voxel_sampler_cfg[1].max_voxels
        if voxels.shape[0] >= max_voxels:
            sweeps_points = voxels[:max_voxels]
        else:
            choices = np.random.choice(
                voxels.shape[0], max_voxels, replace=True)
            sweeps_points = voxels[choices]
        points = torch.cat([cur_points, sweeps_points], 0).squeeze(1)
        return points

    def _prepare_input(self, points, img_metas):
        """Prepare network inputs."""
        if 'use_voxel_sample' in self.test_cfg.keys() and \
                self.test_cfg.use_voxel_sample:
            points_cat = []
            for idx in range(len(points)):
                points_cat.append(
                    self._points_voxel_sampling(
                        points[idx][:img_metas[idx]['cur_points_num']],
                        points[idx][img_metas[idx]['cur_points_num']:]))
            points = points_cat
            points_cat = torch.stack(points_cat)
        else:
            points_cat = torch.stack(points)
        return points_cat, points
