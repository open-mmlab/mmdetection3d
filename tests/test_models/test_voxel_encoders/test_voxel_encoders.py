# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn.functional as F

from mmdet3d.registry import MODELS


def test_hard_simple_VFE():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    hard_simple_VFE_cfg = dict(type='HardSimpleVFE', num_features=5)
    hard_simple_VFE = MODELS.build(hard_simple_VFE_cfg)
    features = torch.rand([240000, 10, 5])
    num_voxels = torch.randint(1, 10, [240000])

    outputs = hard_simple_VFE(features, num_voxels, None)
    assert outputs.shape == torch.Size([240000, 5])


def test_seg_VFE():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    seg_VFE_cfg = dict(
        type='SegVFE',
        feat_channels=[64, 128, 256, 256],
        grid_shape=[480, 360, 32],
        with_voxel_center=True,
        feat_compression=16,
        return_point_feats=True)
    seg_VFE = MODELS.build(seg_VFE_cfg)
    seg_VFE = seg_VFE.cuda()
    features = torch.rand([240000, 6]).cuda()
    coors = []
    for i in range(4):
        coor = torch.randint(0, 10, (60000, 3))
        coor = F.pad(coor, (1, 0), mode='constant', value=i)
        coors.append(coor)
    coors = torch.cat(coors, dim=0).cuda()
    out_features, out_coors, out_point_features = seg_VFE(features, coors)
    assert out_features.shape[0] == out_coors.shape[0]
    assert len(out_point_features) == 4
    assert out_point_features[0].shape == torch.Size([240000, 64])
    assert out_point_features[1].shape == torch.Size([240000, 128])
    assert out_point_features[2].shape == torch.Size([240000, 256])
    assert out_point_features[3].shape == torch.Size([240000, 256])
