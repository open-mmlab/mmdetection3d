model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=5,
            point_cloud_range=[0, -40, -3, 70.4, 40, 1],
            voxel_size=[0.05, 0.05, 0.1],
            max_voxels=(16000, 40000))),
    voxel_encoder=dict(type='HardSimpleVFE'),
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act')),
    backbone=dict(
        type='SQUEEZE',
        in_channels=3,
        out_channels=[64, 128, 256, 512],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    neck=dict(
        type='SQUEEZEFPN',
        in_channels=[64, 128, 256, 512],
        out_channels=[256, 256, 256, 256],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        conv_cfg=dict(type='Conv2d', bias=False)),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[[0, -40, -1.8, 70.4, 40, -1.8]],
            sizes=[[1.6, 3.9, 1.56]],
            rotations=[0, 1.57]),
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
        loss_dir=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    train_cfg=dict(assigner=dict(type='MaxIoUAssigner')),
    test_cfg=dict(use_rotate_nms=True, nms_across_levels=False, nms_pre=1000, nms_thr=0.01, score_thr=0.1, min_bbox_size=0, max_num=500)
)
