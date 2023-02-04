# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmdet3d.registry import MODELS


def test_secfpn():
    neck_cfg = dict(
        type='SECONDFPN',
        in_channels=[2, 3],
        upsample_strides=[1, 2],
        out_channels=[4, 6],
    )
    neck = MODELS.build(neck_cfg)
    assert neck.deblocks[0][0].in_channels == 2
    assert neck.deblocks[1][0].in_channels == 3
    assert neck.deblocks[0][0].out_channels == 4
    assert neck.deblocks[1][0].out_channels == 6
    assert neck.deblocks[0][0].stride == (1, 1)
    assert neck.deblocks[1][0].stride == (2, 2)
    assert neck is not None

    neck_cfg = dict(
        type='SECONDFPN',
        in_channels=[2, 2],
        upsample_strides=[1, 2, 4],
        out_channels=[2, 2],
    )
    with pytest.raises(AssertionError):
        MODELS.build(neck_cfg)

    neck_cfg = dict(
        type='SECONDFPN',
        in_channels=[2, 2, 4],
        upsample_strides=[1, 2, 4],
        out_channels=[2, 2],
    )
    with pytest.raises(AssertionError):
        MODELS.build(neck_cfg)


def test_centerpoint_fpn():

    second_cfg = dict(
        type='SECOND',
        in_channels=2,
        out_channels=[2, 2, 2],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False))

    second = MODELS.build(second_cfg)

    # centerpoint usage of fpn
    centerpoint_fpn_cfg = dict(
        type='SECONDFPN',
        in_channels=[2, 2, 2],
        out_channels=[2, 2, 2],
        upsample_strides=[0.5, 1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True)

    # original usage of fpn
    fpn_cfg = dict(
        type='SECONDFPN',
        in_channels=[2, 2, 2],
        upsample_strides=[1, 2, 4],
        out_channels=[2, 2, 2])

    second_fpn = MODELS.build(fpn_cfg)

    centerpoint_second_fpn = MODELS.build(centerpoint_fpn_cfg)

    input = torch.rand([2, 2, 32, 32])
    sec_output = second(input)
    centerpoint_output = centerpoint_second_fpn(sec_output)
    second_output = second_fpn(sec_output)
    assert centerpoint_output[0].shape == torch.Size([2, 6, 8, 8])
    assert second_output[0].shape == torch.Size([2, 6, 16, 16])
