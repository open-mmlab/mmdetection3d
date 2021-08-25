# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmdet3d.models.builder import build_backbone, build_neck


def test_centerpoint_fpn():

    second_cfg = dict(
        type='SECOND',
        in_channels=64,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False))

    second = build_backbone(second_cfg)

    # centerpoint usage of fpn
    centerpoint_fpn_cfg = dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[0.5, 1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True)

    # original usage of fpn
    fpn_cfg = dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128])

    second_fpn = build_neck(fpn_cfg)

    centerpoint_second_fpn = build_neck(centerpoint_fpn_cfg)

    input = torch.rand([4, 64, 512, 512])
    sec_output = second(input)
    centerpoint_output = centerpoint_second_fpn(sec_output)
    second_output = second_fpn(sec_output)
    assert centerpoint_output[0].shape == torch.Size([4, 384, 128, 128])
    assert second_output[0].shape == torch.Size([4, 384, 256, 256])


def test_imvoxel_neck():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')

    neck_cfg = dict(
        type='OutdoorImVoxelNeck', in_channels=64, out_channels=256)
    neck = build_neck(neck_cfg).cuda()
    inputs = torch.rand([1, 64, 216, 248, 12], device='cuda')
    outputs = neck(inputs)
    assert outputs[0].shape == (1, 256, 248, 216)
