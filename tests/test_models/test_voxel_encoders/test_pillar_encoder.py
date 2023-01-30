# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmdet3d.registry import MODELS


def test_pillar_feature_net():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    pillar_feature_net_cfg = dict(
        type='PillarFeatureNet',
        in_channels=5,
        feat_channels=[64],
        with_distance=False,
        voxel_size=(0.2, 0.2, 8),
        point_cloud_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01))
    pillar_feature_net = MODELS.build(pillar_feature_net_cfg)

    features = torch.rand([97297, 20, 5])
    num_voxels = torch.randint(1, 100, [97297])
    coors = torch.randint(0, 100, [97297, 4])

    features = pillar_feature_net(features, num_voxels, coors)
    assert features.shape == torch.Size([97297, 64])
