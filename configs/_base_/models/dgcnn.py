# model settings
model = dict(
    type='EncoderDecoder3D',
    backbone=dict(
        type='DGCNNBackbone',
        in_channels=9,  # [xyz, rgb, normal_xyz], modified with dataset
        num_samples=(20, 20, 20),
        knn_modes=('D-KNN', 'F-KNN', 'F-KNN'),
        radius=(None, None, None),
        gf_channels=((64, 64), (64, 64), (64, )),
        fa_channels=(1024, ),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.2)),
    decode_head=dict(
        type='DGCNNHead',
        fp_channels=(1216, 512),
        channels=256,
        dropout_ratio=0.5,
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=None,  # modified with dataset
            loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide'))
