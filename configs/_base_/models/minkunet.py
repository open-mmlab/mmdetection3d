model = dict(
    type='MinkUNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=64,
            point_cloud_range=[-50, -50, -5, 50, 50, 3],
            voxel_size=[0.25, 0.25, 8],
            max_voxels=(30000, 40000)),
    ),
    voxel_encoder=dict(type='HardSimpleVFE'),
    backbone=dict(
        type='MinkUNetBackbone',
        in_channels=4,
        base_channels=16,
        enc_channels=[16, 32, 64, 128],
        dec_channels=[128, 64, 48, 48],
        num_stages=4,
        init_cfg=None),
    decode_head=dict(
        type='MinkUNetHead',
        channels=48,
        num_classes=19,
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
