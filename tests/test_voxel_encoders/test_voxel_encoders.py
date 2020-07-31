import numpy as np
import torch

from mmdet3d.models.builder import build_voxel_encoder


def _set_seed():
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)


def test_pillar_feature_net():
    _set_seed()
    pillar_feature_net_cfg = dict(
        type='PillarFeatureNet',
        in_channels=5,
        feat_channels=[64],
        with_distance=False,
        voxel_size=(0.2, 0.2, 8),
        point_cloud_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
    )

    pillar_feature_net = build_voxel_encoder(pillar_feature_net_cfg)

    norm_layer_weight = torch.from_numpy(
        np.load('tests/test_voxel_encoders/norm_weight.npy'))
    pillar_feature_net.pfn_layers[0].norm.weight = torch.nn.Parameter(
        norm_layer_weight)
    features = torch.from_numpy(
        np.load('tests/test_voxel_encoders/input_features.npy'))
    num_voxels = torch.from_numpy(
        np.load('tests/test_voxel_encoders/num_voxels.npy'))
    coors = torch.from_numpy(np.load('tests/test_voxel_encoders/coors.npy'))

    expected_features = torch.from_numpy(
        np.load('tests/test_voxel_encoders/expected_features.npy'))

    features = pillar_feature_net(features, num_voxels, coors)
    assert torch.allclose(features, expected_features)


def test_hard_simple_VFE():
    hard_simple_VFE_cfg = dict(type='HardSimpleVFE', num_features=5)
    hard_simple_VFE = build_voxel_encoder(hard_simple_VFE_cfg)
    features = torch.rand([240000, 10, 5])
    num_voxels = torch.randint(0, 100, [240000])

    outputs = hard_simple_VFE(features, num_voxels, None)
    assert outputs.shape == torch.Size([240000, 5])
