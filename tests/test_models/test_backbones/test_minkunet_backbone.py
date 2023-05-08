# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn.functional as F

from mmdet3d.registry import MODELS


def test_minkunet_backbone():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')

    try:
        import torchsparse  # noqa: F401
    except ImportError:
        pytest.skip('test requires Torchsparse installation')

    coordinates, features = [], []
    for i in range(2):
        c = torch.randint(0, 10, (100, 3)).int()
        c = F.pad(c, (0, 1), mode='constant', value=i)
        coordinates.append(c)
        f = torch.rand(100, 4)
        features.append(f)
    features = torch.cat(features, dim=0).cuda()
    coordinates = torch.cat(coordinates, dim=0).cuda()

    cfg = dict(
        type='MinkUNetBackbone',
        input_conv_module=dict(
            type='SparseConvModule',
            conv_type='SubMConv3d',
            norm_cfg=dict(type='BN1d')),
        downsample_module=dict(
            type='SparseConvModule',
            conv_type='SparseConv3d',
            norm_cfg=dict(type='BN1d')),
        upsample_module=dict(
            type='SparseConvModule',
            conv_type='SparseInverseConv3d',
            norm_cfg=dict(type='BN1d')),
        residual_block=dict(
            type='SparseBasicBlock',
            conv_cfg=dict(type='SubMConv3d'),
            norm_cfg=dict(type='BN1d')),
        sparseconv_backends='spconv')
    self = MODELS.build(cfg).cuda()
    self.init_weights()

    y = self(features, coordinates)
    assert y.F.shape == torch.Size([200, 96])
    assert y.C.shape == torch.Size([200, 4])
