import numpy as np
import pytest
import torch

from mmdet3d.models import build_backbone


def test_pointnet2_sa_ssg():
    if not torch.cuda.is_available():
        pytest.skip()

    cfg = dict(
        type='PointNet2SASSG',
        in_channels=6,
        num_points=(32, 16),
        radius=(0.8, 1.2),
        num_samples=(16, 8),
        sa_channels=((8, 16), (16, 16)),
        fp_channels=((16, 16), (16, 16)))
    self = build_backbone(cfg)
    self.cuda()
    assert self.SA_modules[0].mlps[0].layer0.conv.in_channels == 6
    assert self.SA_modules[0].mlps[0].layer0.conv.out_channels == 8
    assert self.SA_modules[0].mlps[0].layer1.conv.out_channels == 16
    assert self.SA_modules[1].mlps[0].layer1.conv.out_channels == 16
    assert self.FP_modules[0].mlps.layer0.conv.in_channels == 32
    assert self.FP_modules[0].mlps.layer0.conv.out_channels == 16
    assert self.FP_modules[1].mlps.layer0.conv.in_channels == 19

    xyz = np.fromfile('tests/data/sunrgbd/points/000001.bin', dtype=np.float32)
    xyz = torch.from_numpy(xyz).view(1, -1, 6).cuda()  # (B, N, 6)
    # test forward
    ret_dict = self(xyz)
    fp_xyz = ret_dict['fp_xyz']
    fp_features = ret_dict['fp_features']
    fp_indices = ret_dict['fp_indices']
    assert len(fp_xyz) == len(fp_features) == len(fp_indices) == 3
    assert fp_xyz[0].shape == torch.Size([1, 16, 3])
    assert fp_xyz[1].shape == torch.Size([1, 32, 3])
    assert fp_xyz[2].shape == torch.Size([1, 100, 3])
    assert fp_features[2].shape == torch.Size([1, 16, 100])
    assert fp_indices[2].shape == torch.Size([1, 100])


def test_h3d_backbone():
    if not torch.cuda.is_available():
        pytest.skip()

    cfg = dict(
        type='H3DBackbone',
        backbone=[
            dict(
                type='PointNet2SASSG',
                in_channels=4,
                num_points=(2048, 1024, 512, 256),
                radius=(0.2, 0.4, 0.8, 1.2),
                num_samples=(64, 32, 16, 16),
                sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                             (128, 128, 256)),
                fp_channels=((256, 256), (256, 256)),
                norm_cfg=dict(type='BN2d'),
                pool_mod='max',
                suffix=''),
            dict(
                type='PointNet2SASSG',
                in_channels=4,
                num_points=(2048, 1024, 512, 256),
                radius=(0.2, 0.4, 0.8, 1.2),
                num_samples=(64, 32, 16, 16),
                sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                             (128, 128, 256)),
                fp_channels=((256, 256), (256, 256)),
                norm_cfg=dict(type='BN2d'),
                pool_mod='max',
                suffix='net1'),
            dict(
                type='PointNet2SASSG',
                in_channels=4,
                num_points=(2048, 1024, 512, 256),
                radius=(0.2, 0.4, 0.8, 1.2),
                num_samples=(64, 32, 16, 16),
                sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                             (128, 128, 256)),
                fp_channels=((256, 256), (256, 256)),
                norm_cfg=dict(type='BN2d'),
                pool_mod='max',
                suffix='net2'),
            dict(
                type='PointNet2SASSG',
                in_channels=4,
                num_points=(2048, 1024, 512, 256),
                radius=(0.2, 0.4, 0.8, 1.2),
                num_samples=(64, 32, 16, 16),
                sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                             (128, 128, 256)),
                fp_channels=((256, 256), (256, 256)),
                norm_cfg=dict(type='BN2d'),
                pool_mod='max',
                suffix='net3')
        ])

    self = build_backbone(cfg)
    self.cuda()

    assert len(self.backbone_list) == 4

    xyz = np.fromfile('tests/data/sunrgbd/points/000001.bin', dtype=np.float32)
    xyz = torch.from_numpy(xyz).view(1, -1, 6).cuda()  # (B, N, 6)
    # test forward
    ret_dict = self(xyz[:, :, :4])
    hd_feature = ret_dict['hd_feature']
    assert hd_feature.shape == torch.Size([1, 256, 1024])
