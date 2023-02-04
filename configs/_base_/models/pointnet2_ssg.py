# model settings
model = dict(
    type='EncoderDecoder3D',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    backbone=dict(
        type='PointNet2SASSG',
        in_channels=6,  # [xyz, rgb], should be modified with dataset
        num_points=(1024, 256, 64, 16),
        radius=(0.1, 0.2, 0.4, 0.8),
        num_samples=(32, 32, 32, 32),
        sa_channels=((32, 32, 64), (64, 64, 128), (128, 128, 256), (256, 256,
                                                                    512)),
        fp_channels=(),
        norm_cfg=dict(type='BN2d'),
        sa_cfg=dict(
            type='PointSAModule',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=False)),
    decode_head=dict(
        type='PointNet2Head',
        fp_channels=((768, 256, 256), (384, 256, 256), (320, 256, 128),
                     (128, 128, 128, 128)),
        channels=128,
        dropout_ratio=0.5,
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        act_cfg=dict(type='ReLU'),
        loss_decode=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=None,  # should be modified with dataset
            loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide'))
