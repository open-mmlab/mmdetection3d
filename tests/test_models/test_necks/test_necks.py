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


def test_fp_neck():
    if not torch.cuda.is_available():
        pytest.skip()

    xyzs = [16384, 4096, 1024, 256, 64]
    feat_channels = [1, 96, 256, 512, 1024]
    channel_num = 5

    sa_xyz = [torch.rand(3, xyzs[i], 3) for i in range(channel_num)]
    sa_features = [
        torch.rand(3, feat_channels[i], xyzs[i]) for i in range(channel_num)
    ]

    neck_cfg = dict(
        type='PointNetFPNeck',
        fp_channels=((1536, 512, 512), (768, 512, 512), (608, 256, 256),
                     (257, 128, 128)))

    neck = build_neck(neck_cfg)
    neck.init_weights()

    if torch.cuda.is_available():
        sa_xyz = [x.cuda() for x in sa_xyz]
        sa_features = [x.cuda() for x in sa_features]
        neck.cuda()

    feats_sa = {'sa_xyz': sa_xyz, 'sa_features': sa_features}
    outputs = neck(feats_sa)
    assert outputs['fp_xyz'].cpu().numpy().shape == (3, 16384, 3)
    assert outputs['fp_features'].detach().cpu().numpy().shape == (3, 128,
                                                                   16384)


def test_dla_neck():

    s = 32
    in_channels = [16, 32, 64, 128, 256, 512]
    feat_sizes = [s // 2**i for i in range(6)]  # [32, 16, 8, 4, 2, 1]

    if torch.cuda.is_available():
        # Test DLA Neck with DCNv2 on GPU
        neck_cfg = dict(
            type='DLANeck',
            in_channels=[16, 32, 64, 128, 256, 512],
            start_level=2,
            end_level=5,
            norm_cfg=dict(type='GN', num_groups=32))
        neck = build_neck(neck_cfg)
        neck.init_weights()
        neck.cuda()
        feats = [
            torch.rand(4, in_channels[i], feat_sizes[i], feat_sizes[i]).cuda()
            for i in range(len(in_channels))
        ]
        outputs = neck(feats)
        assert outputs.shape == (4, 64, 8, 8)
    else:
        # Test DLA Neck without DCNv2 on CPU
        neck_cfg = dict(
            type='DLANeck',
            in_channels=[16, 32, 64, 128, 256, 512],
            start_level=2,
            end_level=5,
            norm_cfg=dict(type='GN', num_groups=32),
            use_dcn=False)
        neck = build_neck(neck_cfg)
        neck.init_weights()
        feats = [
            torch.rand(4, in_channels[i], feat_sizes[i], feat_sizes[i])
            for i in range(len(in_channels))
        ]
        outputs = neck(feats)
        assert outputs[0].shape == (4, 64, 8, 8)
