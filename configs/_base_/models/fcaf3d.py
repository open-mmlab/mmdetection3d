model = dict(
    type='MinkSingleStage3DDetector',
    voxel_size=.01,
    backbone=dict(type='MinkResNet', in_channels=3, depth=34),
    head=dict(
        type='FCAF3DHead',
        in_channels=(64, 128, 256, 512),
        out_channels=128,
        voxel_size=.01,
        pts_prune_threshold=100000,
        pts_assign_threshold=27,
        pts_center_threshold=18,
        n_classes=18,
        n_reg_outs=6),
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=.5, score_thr=.01))
