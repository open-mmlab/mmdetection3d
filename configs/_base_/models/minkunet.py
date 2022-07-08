# model settings
model = dict(
    type='EncoderDecoder3D',
    backbone=dict(
        type='MinkUNet',
        depth=18,
        in_channels=3,
        out_channels=18,
        D=3,
    ),
    decode_head=dict(
        type='PAConvHead',
        # PAConv model's decoder takes skip connections from beckbone
        # different from PointNet++, it also concats input features in the last
        # level of decoder, leading to `128 + 6` as the channel number
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
    # correlation loss to regularize PAConv's kernel weights
    loss_regularization=dict(
        type='PAConvRegularizationLoss', reduction='sum', loss_weight=10.0),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide'))
