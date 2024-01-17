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
        feat_compression=16)
    seg_VFE = MODELS.build(seg_VFE_cfg)
    seg_VFE = seg_VFE.cuda()
    features = torch.rand([240000, 6]).cuda()
    coors = []
    for i in range(4):
        coor = torch.randint(0, 10, (60000, 3))
        coor = F.pad(coor, (1, 0), mode='constant', value=i)
        coors.append(coor)
    coors = torch.cat(coors, dim=0).cuda()
    feat_dict = dict(voxels=features, coors=coors)
    feat_dict = seg_VFE(feat_dict)
    assert feat_dict['voxel_feats'].shape[0] == feat_dict['voxel_coors'].shape[
        0]
    assert len(feat_dict['point_feats']) == 4
    assert feat_dict['point_feats'][0].shape == torch.Size([240000, 64])
    assert feat_dict['point_feats'][1].shape == torch.Size([240000, 128])
    assert feat_dict['point_feats'][2].shape == torch.Size([240000, 256])
    assert feat_dict['point_feats'][3].shape == torch.Size([240000, 256])
    assert len(feat_dict['point2voxel_maps']) == 4
