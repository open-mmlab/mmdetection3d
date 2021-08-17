# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch
from mmcv.cnn.bricks import ConvModule

from mmdet3d.models.builder import build_head


def test_paconv_decode_head_loss():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    paconv_decode_head_cfg = dict(
        type='PAConvHead',
        fp_channels=((768, 256, 256), (384, 256, 256), (320, 256, 128),
                     (128 + 6, 128, 128, 128)),
        channels=128,
        num_classes=20,
        dropout_ratio=0.5,
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        act_cfg=dict(type='ReLU'),
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=None,
            loss_weight=1.0),
        ignore_index=20)

    self = build_head(paconv_decode_head_cfg)
    self.cuda()
    assert isinstance(self.conv_seg, torch.nn.Conv1d)
    assert self.conv_seg.in_channels == 128
    assert self.conv_seg.out_channels == 20
    assert self.conv_seg.kernel_size == (1, )
    assert isinstance(self.pre_seg_conv, ConvModule)
    assert isinstance(self.pre_seg_conv.conv, torch.nn.Conv1d)
    assert self.pre_seg_conv.conv.in_channels == 128
    assert self.pre_seg_conv.conv.out_channels == 128
    assert self.pre_seg_conv.conv.kernel_size == (1, )
    assert isinstance(self.pre_seg_conv.bn, torch.nn.BatchNorm1d)
    assert self.pre_seg_conv.bn.num_features == 128
    assert isinstance(self.pre_seg_conv.activate, torch.nn.ReLU)

    # test forward
    sa_xyz = [
        torch.rand(2, 4096, 3).float().cuda(),
        torch.rand(2, 1024, 3).float().cuda(),
        torch.rand(2, 256, 3).float().cuda(),
        torch.rand(2, 64, 3).float().cuda(),
        torch.rand(2, 16, 3).float().cuda(),
    ]
    sa_features = [
        torch.rand(2, 6, 4096).float().cuda(),
        torch.rand(2, 64, 1024).float().cuda(),
        torch.rand(2, 128, 256).float().cuda(),
        torch.rand(2, 256, 64).float().cuda(),
        torch.rand(2, 512, 16).float().cuda(),
    ]
    input_dict = dict(sa_xyz=sa_xyz, sa_features=sa_features)
    seg_logits = self(input_dict)
    assert seg_logits.shape == torch.Size([2, 20, 4096])

    # test loss
    pts_semantic_mask = torch.randint(0, 20, (2, 4096)).long().cuda()
    losses = self.losses(seg_logits, pts_semantic_mask)
    assert losses['loss_sem_seg'].item() > 0

    # test loss with ignore_index
    ignore_index_mask = torch.ones_like(pts_semantic_mask) * 20
    losses = self.losses(seg_logits, ignore_index_mask)
    assert losses['loss_sem_seg'].item() == 0

    # test loss with class_weight
    paconv_decode_head_cfg['loss_decode'] = dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        class_weight=np.random.rand(20),
        loss_weight=1.0)
    self = build_head(paconv_decode_head_cfg)
    self.cuda()
    losses = self.losses(seg_logits, pts_semantic_mask)
    assert losses['loss_sem_seg'].item() > 0
