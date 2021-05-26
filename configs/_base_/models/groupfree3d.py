model = dict(
    type='GroupFree3DNet',
    backbone=dict(
        type='PointNet2SASSG',
        in_channels=3,
        num_points=(2048, 1024, 512, 256),
        radius=(0.2, 0.4, 0.8, 1.2),
        num_samples=(64, 32, 16, 16),
        sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                     (128, 128, 256)),
        fp_channels=((256, 256), (256, 288)),
        norm_cfg=dict(type='BN2d'),
        sa_cfg=dict(
            type='PointSAModule',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=True)),
    bbox_head=dict(
        type='GroupFree3DHead',
        in_channels=288,
        num_decoder_layers=6,
        num_proposal=256,
        transformerlayers=dict(
            type='BaseTransformerLayer',
            attn_cfgs=dict(
                type='GroupFree3DMultiheadAttention',
                embed_dims=288,
                num_heads=8,
                attn_drop=0.1,
                dropout_layer=dict(type='DropOut', drop_prob=0.1)),
            ffn_cfgs=dict(
                embed_dims=288,
                feedforward_channels=2048,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True)),
            operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn',
                             'norm')),
        pred_layer_cfg=dict(
            in_channels=288, shared_conv_channels=(288, 288), bias=True)),
    # model training and testing settings
    train_cfg=dict(sample_mod='kps'),
    test_cfg=dict(
        sample_mod='kps',
        nms_thr=0.25,
        score_thr=0.05,
        per_class_proposal=True,
        prediction_stages='last'))
