# model settings
model = dict(
    type='SparseEncoderDecoder3D',
    voxel_size=1,
    backbone=dict(
        type='MinkUNetBase',
        depth=18,
        in_channels=3,
        D=3,
    ),
    decode_head=dict(
        type='MinkUNetHead',
        channels=96*1,
        num_classes=20,
        loss_decode=dict(
            type='CrossEntropyLoss',
            class_weight=None,  # should be modified with dataset
            loss_weight=1.0,
        )),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide'))
