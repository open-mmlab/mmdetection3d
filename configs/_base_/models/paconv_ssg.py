# model settings
model = dict(
    type='EncoderDecoder3D',
    backbone=dict(
        type='PointNet2SASSG',
        in_channels=9,  # [xyz, rgb, normalized_xyz]
        num_points=(1024, 256, 64, 16),
        radius=(None, None, None, None),  # use kNN instead of ball query
        num_samples=(32, 32, 32, 32),
        sa_channels=((32, 32, 64), (64, 64, 128), (128, 128, 256), (256, 256,
                                                                    512)),
        fp_channels=(),
        norm_cfg=dict(type='BN2d', momentum=0.1),
        sa_cfg=dict(
            type='PAConvSAModule',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=False,
            paconv_num_kernels=[16, 16, 16],
            paconv_kernel_input='neighbor',
            scorenet_input='ed7',
            scorenet_cfg=dict(
                mlp_channels=[16, 16, 16],
                score_norm='softmax',
                temp_factor=1.0,
                last_bn=False))),
    decode_head=dict(
        type='PAConvHead',
        fp_channels=((768, 256, 256), (384, 256, 256), (320, 256, 128),
                     (128 + 6, 128, 128, 128)),
        channels=128,
        dropout_ratio=0.5,
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        act_cfg=dict(type='ReLU'),
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=None,  # should be modified with dataset
            loss_weight=1.0)),
    # regularization loss for PAConv's weight kernels
    regularization_loss=dict(
        type='PAConvCorrelationLoss', reduction='sum', loss_weight=10.0),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide'))
