# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmdet3d.registry import MODELS


def test_mink_resnet():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')

    try:
        import MinkowskiEngine as ME
    except ImportError:
        pytest.skip('test requires MinkowskiEngine installation')

    coordinates, features = [], []
    np.random.seed(42)
    # batch of 2 point clouds
    for i in range(2):
        c = torch.from_numpy(np.random.rand(500, 3) * 100)
        coordinates.append(c.float().cuda())
        f = torch.from_numpy(np.random.rand(500, 3))
        features.append(f.float().cuda())
    tensor_coordinates, tensor_features = ME.utils.sparse_collate(
        coordinates, features)
    x = ME.SparseTensor(
        features=tensor_features, coordinates=tensor_coordinates)

    # MinkResNet34 with 4 outputs
    cfg = dict(type='MinkResNet', depth=34, in_channels=3)
    self = MODELS.build(cfg).cuda()
    self.init_weights()

    y = self(x)
    assert len(y) == 4
    assert y[0].F.shape == torch.Size([900, 64])
    assert y[0].tensor_stride[0] == 8
    assert y[1].F.shape == torch.Size([472, 128])
    assert y[1].tensor_stride[0] == 16
    assert y[2].F.shape == torch.Size([105, 256])
    assert y[2].tensor_stride[0] == 32
    assert y[3].F.shape == torch.Size([16, 512])
    assert y[3].tensor_stride[0] == 64

    # MinkResNet50 with 2 outputs
    cfg = dict(
        type='MinkResNet', depth=34, in_channels=3, num_stages=2, pool=False)
    self = MODELS.build(cfg).cuda()
    self.init_weights()

    y = self(x)
    assert len(y) == 2
    assert y[0].F.shape == torch.Size([985, 64])
    assert y[0].tensor_stride[0] == 4
    assert y[1].F.shape == torch.Size([900, 128])
    assert y[1].tensor_stride[0] == 8
