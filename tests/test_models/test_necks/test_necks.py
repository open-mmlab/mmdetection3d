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


def test_outdoor_imvoxel_neck():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')

    neck_cfg = dict(
        type='OutdoorImVoxelNeck', in_channels=64, out_channels=256)
    neck = build_neck(neck_cfg).cuda()
    inputs = torch.rand([1, 64, 216, 248, 12], device='cuda')
    outputs = neck(inputs)
    assert outputs[0].shape == (1, 256, 248, 216)


def test_indoor_imvoxel_neck():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')

    neck_cfg = dict(
        type='IndoorImVoxelNeck',
        in_channels=64,
        out_channels=256,
        n_blocks=[1, 1, 1])
    neck = build_neck(neck_cfg).cuda()
    inputs = torch.rand([1, 64, 40, 40, 16], device='cuda')
    outputs = neck(inputs)
    assert len(outputs) == 3
    assert outputs[0].shape == (1, 256, 40, 40, 16)
    assert outputs[1].shape == (1, 256, 20, 20, 8)
    assert outputs[2].shape == (1, 256, 10, 10, 4)


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
        assert outputs[0].shape == (4, 64, 8, 8)
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


def test_lss_view_transformer():
    feats_image_view = torch.rand(1, 2, 512, 44, 16)
    rots = torch.tensor([[[1.0000, 0.0067, 0.0017], [-0.0019, 0.0194, 0.9998],
                          [0.0067, -0.9998, 0.0194]],
                         [[0.5480, -0.0104, 0.8364], [-0.8360, 0.0250, 0.5481],
                          [-0.0266, -0.9996, 0.0050]]]).unsqueeze(0)
    trans = torch.tensor([[-0.0097, 0.4028, -0.3244],
                          [0.4984, 0.3233, -0.3376]]).unsqueeze(0)

    intrins = torch.tensor([[[1.2664e+03, 0.0000e+00, 8.1627e+02],
                             [0.0000e+00, 1.2664e+03, 4.9151e+02],
                             [0.0000e+00, 0.0000e+00,
                              1.0000e+00]]]).expand(1, 2, 3, 3)

    post_rots = torch.tensor([[[0.4800, 0.0000, 0.0000],
                               [0.0000, 0.4800, 0.0000],
                               [0.0000, 0.0000, 1.0000]],
                              [[0.4800, 0.0000, 0.0000],
                               [0.0000, 0.4800, 0.0000],
                               [0.0000, 0.0000, 1.0000]]]).unsqueeze(0)

    post_trans = torch.tensor([[-32., -176., 0.], [-32., -176.,
                                                   0.]]).unsqueeze(0)

    inputs = (feats_image_view, rots, trans, intrins, post_rots, post_trans)

    grid_config = {
        'x': [-51.2, 51.2, 0.8],
        'y': [-51.2, 51.2, 0.8],
        'z': [-10.0, 10.0, 20.0],
        'depth': [1.0, 60.0, 1.0],
    }
    neck_cfg = dict(
        type='LSSViewTransformer',
        grid_config=grid_config,
        input_size=(256, 704),
        downsample=16,
        in_channels=512,
        out_channels=64,
        accelerate=False)
    neck = build_neck(neck_cfg)
    neck.init_weights()

    if torch.cuda.is_available():
        inputs = tuple([item.cuda() for item in inputs])
        neck = neck.cuda()

    # for naive lift-splat-shoot view transformer
    feats_bev = neck(inputs)

    # for accelerated lift-splat-shoot view transformer
    neck.accelerate = True
    neck.max_voxel_points = 300
    feats_bev_acc = neck(inputs)
    assert feats_bev.shape == (1, 64, 128, 128)
    assert feats_bev_acc.shape == (1, 64, 128, 128)
    assert torch.sum(
        (feats_bev - feats_bev_acc).abs() < 0.0001).float() / (64 * 128 *
                                                               128) > 0.99
