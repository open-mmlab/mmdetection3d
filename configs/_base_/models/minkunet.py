model = dict(
    type='MinkUNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_type='minkunet',
        voxel_layer=dict(
            max_num_points=-1,
            point_cloud_range=[-100, -100, -20, 100, 100, 20],
            voxel_size=[0.05, 0.05, 0.05],
            max_voxels=(-1, -1)),
    ),
    backbone=dict(
        type='MinkUNetBackbone',
        in_channels=4,
        base_channels=16,
        enc_channels=[16, 32, 64, 128],
        dec_channels=[128, 64, 48, 48],
        num_stages=4,
        init_cfg=None),
    decode_head=dict(
        type='MinkUNetHead', channels=48, num_classes=19, ignore_index=19),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
