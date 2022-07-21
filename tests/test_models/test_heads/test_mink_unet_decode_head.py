# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmdet3d.models.builder import build_head


def test_minkunet_decode_head_loss():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')

    try:
        import MinkowskiEngine as ME
    except ImportError:
        pytest.skip('test requires MinkowskiEngine installation')

    # build head with input feature channels 96
    # from MinkUNet18 backbone as an example
    minkunet_decode_head_cfg = dict(
        type='MinkUNetHead',
        channels=96 * 1,
        num_classes=20,
        ignore_index=20,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=None,
            loss_weight=1.0,
        ))

    self = build_head(minkunet_decode_head_cfg)
    self.cuda()

    assert isinstance(self.final, ME.MinkowskiConvolution)
    assert self.final.in_channels == 96
    assert self.final.out_channels == 20

    # test forward
    coordinates = [
        torch.from_numpy(np.random.rand(100, 3)).float().cuda(),
    ]
    features = [
        torch.from_numpy(np.random.rand(100, 3)).float().cuda(),
    ]
    tensor_coordinates, tensor_features = ME.utils.sparse_collate(
        coordinates, features)
    x = ME.SparseTensor(
        features=tensor_features, coordinates=tensor_coordinates)

    input_dict = {'features': x}
    seg_logits = self(input_dict)
    assert seg_logits.shape == torch.Size([1, 20])

    # test loss
    pts_semantic_mask = torch.randint(0, 20, (1, )).long().cuda()
    losses = self.losses(seg_logits, pts_semantic_mask)
    assert losses['loss_sem_seg'].item() > 0

    # test loss with ignore_index
    ignore_index_mask = torch.ones_like(pts_semantic_mask) * 20
    losses = self.losses(seg_logits, ignore_index_mask)
    assert losses['loss_sem_seg'].item() == 0

    # test loss with class_weight
    minkunet_decode_head_cfg['loss_decode'] = dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        class_weight=np.random.rand(20),
        loss_weight=1.0)
    self = build_head(minkunet_decode_head_cfg)
    self.cuda()
    losses = self.losses(seg_logits, pts_semantic_mask)
    assert losses['loss_sem_seg'].item() > 0
