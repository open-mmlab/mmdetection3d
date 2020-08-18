import torch

from mmdet3d.models.builder import build_voxel_encoder


def test_pillar_feature_net():
    pillar_feature_net_cfg = dict(
        type='PillarFeatureNet',
        in_channels=5,
        feat_channels=[64],
        with_distance=False,
        voxel_size=(0.2, 0.2, 8),
        point_cloud_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01))

    pillar_feature_net = build_voxel_encoder(pillar_feature_net_cfg)

    features = torch.rand([97297, 20, 5])
    num_voxels = torch.randint(1, 100, [97297])
    coors = torch.randint(0, 100, [97297, 4])

    features = pillar_feature_net(features, num_voxels, coors)
    assert features.shape == torch.Size([97297, 64])


def test_hard_simple_VFE():
    hard_simple_VFE_cfg = dict(type='HardSimpleVFE', num_features=5)
    hard_simple_VFE = build_voxel_encoder(hard_simple_VFE_cfg)
    features = torch.rand([240000, 10, 5])
    num_voxels = torch.randint(1, 10, [240000])

    outputs = hard_simple_VFE(features, num_voxels, None)
    assert outputs.shape == torch.Size([240000, 5])
