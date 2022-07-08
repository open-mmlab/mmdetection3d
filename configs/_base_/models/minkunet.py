# model settings
model = dict(
    type='EncoderDecoder3D',
    backbone=dict(
        type='MinkUNetBase',
        depth=18,
        in_channels=3,
        D=3,
    ),
    decode_head=dict(
        type='UNetHead',
        in_channels=96*1,
        out_channels=18,
        dropout_ratio=0.5,
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        act_cfg=dict(type='ReLU'),
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=None,  # should be modified with dataset
            loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide'))
