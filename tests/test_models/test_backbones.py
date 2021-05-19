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
    sa_xyz = ret_dict['sa_xyz']
    sa_features = ret_dict['sa_features']
    sa_indices = ret_dict['sa_indices']
    assert len(fp_xyz) == len(fp_features) == len(fp_indices) == 3
    assert len(sa_xyz) == len(sa_features) == len(sa_indices) == 3
    assert fp_xyz[0].shape == torch.Size([1, 16, 3])
    assert fp_xyz[1].shape == torch.Size([1, 32, 3])
    assert fp_xyz[2].shape == torch.Size([1, 100, 3])
    assert fp_features[0].shape == torch.Size([1, 16, 16])
    assert fp_features[1].shape == torch.Size([1, 16, 32])
    assert fp_features[2].shape == torch.Size([1, 16, 100])
    assert fp_indices[0].shape == torch.Size([1, 16])
    assert fp_indices[1].shape == torch.Size([1, 32])
    assert fp_indices[2].shape == torch.Size([1, 100])
    assert sa_xyz[0].shape == torch.Size([1, 100, 3])
    assert sa_xyz[1].shape == torch.Size([1, 32, 3])
    assert sa_xyz[2].shape == torch.Size([1, 16, 3])
    assert sa_features[0].shape == torch.Size([1, 3, 100])
    assert sa_features[1].shape == torch.Size([1, 16, 32])
    assert sa_features[2].shape == torch.Size([1, 16, 16])
    assert sa_indices[0].shape == torch.Size([1, 100])
    assert sa_indices[1].shape == torch.Size([1, 32])
    assert sa_indices[2].shape == torch.Size([1, 16])

    # test only xyz input without features
    cfg['in_channels'] = 3
    self = build_backbone(cfg)
    self.cuda()
    ret_dict = self(xyz[..., :3])
    assert len(fp_xyz) == len(fp_features) == len(fp_indices) == 3
    assert len(sa_xyz) == len(sa_features) == len(sa_indices) == 3
    assert fp_features[0].shape == torch.Size([1, 16, 16])
    assert fp_features[1].shape == torch.Size([1, 16, 32])
    assert fp_features[2].shape == torch.Size([1, 16, 100])
    assert sa_features[0].shape == torch.Size([1, 3, 100])
    assert sa_features[1].shape == torch.Size([1, 16, 32])
    assert sa_features[2].shape == torch.Size([1, 16, 16])


def test_multi_backbone():
    if not torch.cuda.is_available():
        pytest.skip()

    # test list config
    cfg_list = dict(
        type='MultiBackbone',
        num_streams=4,
        suffixes=['net0', 'net1', 'net2', 'net3'],
        backbones=[
            dict(
                type='PointNet2SASSG',
                in_channels=4,
                num_points=(256, 128, 64, 32),
                radius=(0.2, 0.4, 0.8, 1.2),
                num_samples=(64, 32, 16, 16),
                sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                             (128, 128, 256)),
                fp_channels=((256, 256), (256, 256)),
                norm_cfg=dict(type='BN2d')),
            dict(
                type='PointNet2SASSG',
                in_channels=4,
                num_points=(256, 128, 64, 32),
                radius=(0.2, 0.4, 0.8, 1.2),
                num_samples=(64, 32, 16, 16),
                sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                             (128, 128, 256)),
                fp_channels=((256, 256), (256, 256)),
                norm_cfg=dict(type='BN2d')),
            dict(
                type='PointNet2SASSG',
                in_channels=4,
                num_points=(256, 128, 64, 32),
                radius=(0.2, 0.4, 0.8, 1.2),
                num_samples=(64, 32, 16, 16),
                sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                             (128, 128, 256)),
                fp_channels=((256, 256), (256, 256)),
                norm_cfg=dict(type='BN2d')),
            dict(
                type='PointNet2SASSG',
                in_channels=4,
                num_points=(256, 128, 64, 32),
                radius=(0.2, 0.4, 0.8, 1.2),
                num_samples=(64, 32, 16, 16),
                sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                             (128, 128, 256)),
                fp_channels=((256, 256), (256, 256)),
                norm_cfg=dict(type='BN2d'))
        ])

    self = build_backbone(cfg_list)
    self.cuda()

    assert len(self.backbone_list) == 4

    xyz = np.fromfile('tests/data/sunrgbd/points/000001.bin', dtype=np.float32)
    xyz = torch.from_numpy(xyz).view(1, -1, 6).cuda()  # (B, N, 6)
    # test forward
    ret_dict = self(xyz[:, :, :4])

    assert ret_dict['hd_feature'].shape == torch.Size([1, 256, 128])
    assert ret_dict['fp_xyz_net0'][-1].shape == torch.Size([1, 128, 3])
    assert ret_dict['fp_features_net0'][-1].shape == torch.Size([1, 256, 128])

    # test dict config
    cfg_dict = dict(
        type='MultiBackbone',
        num_streams=2,
        suffixes=['net0', 'net1'],
        aggregation_mlp_channels=[512, 128],
        backbones=dict(
            type='PointNet2SASSG',
            in_channels=4,
            num_points=(256, 128, 64, 32),
            radius=(0.2, 0.4, 0.8, 1.2),
            num_samples=(64, 32, 16, 16),
            sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                         (128, 128, 256)),
            fp_channels=((256, 256), (256, 256)),
            norm_cfg=dict(type='BN2d')))

    self = build_backbone(cfg_dict)
    self.cuda()

    assert len(self.backbone_list) == 2

    # test forward
    ret_dict = self(xyz[:, :, :4])

    assert ret_dict['hd_feature'].shape == torch.Size([1, 128, 128])
    assert ret_dict['fp_xyz_net0'][-1].shape == torch.Size([1, 128, 3])
    assert ret_dict['fp_features_net0'][-1].shape == torch.Size([1, 256, 128])

    # Length of backbone configs list should be equal to num_streams
    with pytest.raises(AssertionError):
        cfg_list['num_streams'] = 3
        build_backbone(cfg_list)

    # Length of suffixes list should be equal to num_streams
    with pytest.raises(AssertionError):
        cfg_dict['suffixes'] = ['net0', 'net1', 'net2']
        build_backbone(cfg_dict)

    # Type of 'backbones' should be Dict or List[Dict].
    with pytest.raises(AssertionError):
        cfg_dict['backbones'] = 'PointNet2SASSG'
        build_backbone(cfg_dict)


def test_pointnet2_sa_msg():
    if not torch.cuda.is_available():
        pytest.skip()

    # PN2MSG used in 3DSSD
    cfg = dict(
        type='PointNet2SAMSG',
        in_channels=4,
        num_points=(256, 64, (32, 32)),
        radii=((0.2, 0.4, 0.8), (0.4, 0.8, 1.6), (1.6, 3.2, 4.8)),
        num_samples=((8, 8, 16), (8, 8, 16), (8, 8, 8)),
        sa_channels=(((8, 8, 16), (8, 8, 16),
                      (8, 8, 16)), ((16, 16, 32), (16, 16, 32), (16, 24, 32)),
                     ((32, 32, 64), (32, 24, 64), (32, 64, 64))),
        aggregation_channels=(16, 32, 64),
        fps_mods=(('D-FPS'), ('FS'), ('F-FPS', 'D-FPS')),
        fps_sample_range_lists=((-1), (-1), (64, -1)),
        norm_cfg=dict(type='BN2d'),
        sa_cfg=dict(
            type='PointSAModuleMSG',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=False))

    self = build_backbone(cfg)
    self.cuda()
    assert self.SA_modules[0].mlps[0].layer0.conv.in_channels == 4
    assert self.SA_modules[0].mlps[0].layer0.conv.out_channels == 8
    assert self.SA_modules[0].mlps[1].layer1.conv.out_channels == 8
    assert self.SA_modules[2].mlps[2].layer2.conv.out_channels == 64

    xyz = np.fromfile('tests/data/sunrgbd/points/000001.bin', dtype=np.float32)
    xyz = torch.from_numpy(xyz).view(1, -1, 6).cuda()  # (B, N, 6)
    # test forward
    ret_dict = self(xyz[:, :, :4])
    sa_xyz = ret_dict['sa_xyz'][-1]
    sa_features = ret_dict['sa_features'][-1]
    sa_indices = ret_dict['sa_indices'][-1]

    assert sa_xyz.shape == torch.Size([1, 64, 3])
    assert sa_features.shape == torch.Size([1, 64, 64])
    assert sa_indices.shape == torch.Size([1, 64])

    # out_indices should smaller than the length of SA Modules.
    with pytest.raises(AssertionError):
        build_backbone(
            dict(
                type='PointNet2SAMSG',
                in_channels=4,
                num_points=(256, 64, (32, 32)),
                radii=((0.2, 0.4, 0.8), (0.4, 0.8, 1.6), (1.6, 3.2, 4.8)),
                num_samples=((8, 8, 16), (8, 8, 16), (8, 8, 8)),
                sa_channels=(((8, 8, 16), (8, 8, 16), (8, 8, 16)),
                             ((16, 16, 32), (16, 16, 32), (16, 24, 32)),
                             ((32, 32, 64), (32, 24, 64), (32, 64, 64))),
                aggregation_channels=(16, 32, 64),
                fps_mods=(('D-FPS'), ('FS'), ('F-FPS', 'D-FPS')),
                fps_sample_range_lists=((-1), (-1), (64, -1)),
                out_indices=(2, 3),
                norm_cfg=dict(type='BN2d'),
                sa_cfg=dict(
                    type='PointSAModuleMSG',
                    pool_mod='max',
                    use_xyz=True,
                    normalize_xyz=False)))

    # PN2MSG used in segmentation
    cfg = dict(
        type='PointNet2SAMSG',
        in_channels=6,  # [xyz, rgb]
        num_points=(1024, 256, 64, 16),
        radii=((0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 0.8)),
        num_samples=((16, 32), (16, 32), (16, 32), (16, 32)),
        sa_channels=(((16, 16, 32), (32, 32, 64)), ((64, 64, 128), (64, 96,
                                                                    128)),
                     ((128, 196, 256), (128, 196, 256)), ((256, 256, 512),
                                                          (256, 384, 512))),
        aggregation_channels=(None, None, None, None),
        fps_mods=(('D-FPS'), ('D-FPS'), ('D-FPS'), ('D-FPS')),
        fps_sample_range_lists=((-1), (-1), (-1), (-1)),
        dilated_group=(False, False, False, False),
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type='BN2d'),
        sa_cfg=dict(
            type='PointSAModuleMSG',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=False))

    self = build_backbone(cfg)
    self.cuda()
    ret_dict = self(xyz)
    sa_xyz = ret_dict['sa_xyz']
    sa_features = ret_dict['sa_features']
    sa_indices = ret_dict['sa_indices']

    assert len(sa_xyz) == len(sa_features) == len(sa_indices) == 5
    assert sa_xyz[0].shape == torch.Size([1, 100, 3])
    assert sa_xyz[1].shape == torch.Size([1, 1024, 3])
    assert sa_xyz[2].shape == torch.Size([1, 256, 3])
    assert sa_xyz[3].shape == torch.Size([1, 64, 3])
    assert sa_xyz[4].shape == torch.Size([1, 16, 3])
    assert sa_features[0].shape == torch.Size([1, 3, 100])
    assert sa_features[1].shape == torch.Size([1, 96, 1024])
    assert sa_features[2].shape == torch.Size([1, 256, 256])
    assert sa_features[3].shape == torch.Size([1, 512, 64])
    assert sa_features[4].shape == torch.Size([1, 1024, 16])
    assert sa_indices[0].shape == torch.Size([1, 100])
    assert sa_indices[1].shape == torch.Size([1, 1024])
    assert sa_indices[2].shape == torch.Size([1, 256])
    assert sa_indices[3].shape == torch.Size([1, 64])
    assert sa_indices[4].shape == torch.Size([1, 16])
