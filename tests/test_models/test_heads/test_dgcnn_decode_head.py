# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch
from mmcv.cnn.bricks import ConvModule

from mmdet3d.models.builder import build_head


def test_dgcnn_decode_head_loss():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    dgcnn_decode_head_cfg = dict(
        type='DGCNNHead',
        fp_channels=(1024, 512),
        channels=256,
        num_classes=13,
        dropout_ratio=0.5,
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=None,
            loss_weight=1.0),
        ignore_index=13)

    self = build_head(dgcnn_decode_head_cfg)
    self.cuda()
    assert isinstance(self.conv_seg, torch.nn.Conv1d)
    assert self.conv_seg.in_channels == 256
    assert self.conv_seg.out_channels == 13
    assert self.conv_seg.kernel_size == (1, )
    assert isinstance(self.pre_seg_conv, ConvModule)
    assert isinstance(self.pre_seg_conv.conv, torch.nn.Conv1d)
    assert self.pre_seg_conv.conv.in_channels == 512
    assert self.pre_seg_conv.conv.out_channels == 256
    assert self.pre_seg_conv.conv.kernel_size == (1, )
    assert isinstance(self.pre_seg_conv.bn, torch.nn.BatchNorm1d)
    assert self.pre_seg_conv.bn.num_features == 256

    # test forward
    fa_points = torch.rand(2, 4096, 1024).float().cuda()
    input_dict = dict(fa_points=fa_points)
    seg_logits = self(input_dict)
    assert seg_logits.shape == torch.Size([2, 13, 4096])

    # test loss
    pts_semantic_mask = torch.randint(0, 13, (2, 4096)).long().cuda()
    losses = self.losses(seg_logits, pts_semantic_mask)
    assert losses['loss_sem_seg'].item() > 0

    # test loss with ignore_index
    ignore_index_mask = torch.ones_like(pts_semantic_mask) * 13
    losses = self.losses(seg_logits, ignore_index_mask)
    assert losses['loss_sem_seg'].item() == 0

    # test loss with class_weight
    dgcnn_decode_head_cfg['loss_decode'] = dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        class_weight=np.random.rand(13),
        loss_weight=1.0)
    self = build_head(dgcnn_decode_head_cfg)
    self.cuda()
    losses = self.losses(seg_logits, pts_semantic_mask)
    assert losses['loss_sem_seg'].item() > 0
