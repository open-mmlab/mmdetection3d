# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmdet3d.models.roi_heads.roi_extractors import (Single3DRoIAwareExtractor,
                                                     Single3DRoIPointExtractor)


def test_single_roiaware_extractor():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')

    roi_layer_cfg = dict(
        type='RoIAwarePool3d', out_size=4, max_pts_per_voxel=128, mode='max')

    self = Single3DRoIAwareExtractor(roi_layer=roi_layer_cfg)
    feats = torch.tensor(
        [[1, 2, 3.3], [1.2, 2.5, 3.0], [0.8, 2.1, 3.5], [1.6, 2.6, 3.6],
         [0.8, 1.2, 3.9], [-9.2, 21.0, 18.2], [3.8, 7.9, 6.3],
         [4.7, 3.5, -12.2], [3.8, 7.6, -2], [-10.6, -12.9, -20], [-16, -18, 9],
         [-21.3, -52, -5], [0, 0, 0], [6, 7, 8], [-2, -3, -4]],
        dtype=torch.float32).cuda()
    coordinate = feats.clone()
    batch_inds = torch.zeros(feats.shape[0]).cuda()
    rois = torch.tensor([[0, 1.0, 2.0, 3.0, 5.0, 4.0, 6.0, -0.3 - np.pi / 2],
                         [0, -10.0, 23.0, 16.0, 20, 10, 20, -0.5 - np.pi / 2]],
                        dtype=torch.float32).cuda()
    # test forward
    pooled_feats = self(feats, coordinate, batch_inds, rois)
    assert pooled_feats.shape == torch.Size([2, 4, 4, 4, 3])
    assert torch.allclose(pooled_feats.sum(),
                          torch.tensor(51.100).cuda(), 1e-3)


def test_single_roipoint_extractor():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')

    roi_layer_cfg = dict(
        type='RoIPointPool3d', num_sampled_points=512, pool_extra_width=0)

    self = Single3DRoIPointExtractor(roi_layer=roi_layer_cfg)

    feats = torch.tensor(
        [[1, 2, 3.3], [1.2, 2.5, 3.0], [0.8, 2.1, 3.5], [1.6, 2.6, 3.6],
         [0.8, 1.2, 3.9], [-9.2, 21.0, 18.2], [3.8, 7.9, 6.3],
         [4.7, 3.5, -12.2], [3.8, 7.6, -2], [-10.6, -12.9, -20], [-16, -18, 9],
         [-21.3, -52, -5], [0, 0, 0], [6, 7, 8], [-2, -3, -4]],
        dtype=torch.float32).unsqueeze(0).cuda()
    points = feats.clone()
    batch_inds = feats.shape[0]
    rois = torch.tensor([[0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.3],
                         [0, -10.0, 23.0, 16.0, 10, 20, 20, 0.5]],
                        dtype=torch.float32).cuda()
    pooled_feats = self(feats, points, batch_inds, rois)
    assert pooled_feats.shape == torch.Size([2, 512, 6])
