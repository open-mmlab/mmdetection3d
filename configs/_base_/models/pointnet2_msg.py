_base_ = './pointnet2_ssg.py'

# model settings
model = dict(
    backbone=dict(
        _delete_=True,
        type='PointNet2SAMSG',
        in_channels=6,  # [xyz, rgb], should be modified with dataset
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
        sa_cfg=dict(
            type='PointSAModuleMSG',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=False)),
    decode_head=dict(
        fp_channels=((1536, 256, 256), (512, 256, 256), (352, 256, 128),
                     (128, 128, 128, 128))))
