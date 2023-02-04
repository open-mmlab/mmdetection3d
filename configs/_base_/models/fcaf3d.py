model = dict(
    type='MinkSingleStage3DDetector',
<<<<<<< HEAD
    voxel_size=.01,
    backbone=dict(type='MinkResNet', in_channels=3, depth=34),
    head=dict(
=======
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    backbone=dict(type='MinkResNet', in_channels=3, depth=34),
    bbox_head=dict(
>>>>>>> bf9488d7e9839e3b785703788a42532d19c19973
        type='FCAF3DHead',
        in_channels=(64, 128, 256, 512),
        out_channels=128,
        voxel_size=.01,
        pts_prune_threshold=100000,
        pts_assign_threshold=27,
        pts_center_threshold=18,
<<<<<<< HEAD
        n_classes=18,
        n_reg_outs=6),
=======
        num_classes=18,
        num_reg_outs=6,
        center_loss=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True),
        bbox_loss=dict(type='AxisAlignedIoULoss'),
        cls_loss=dict(type='mmdet.FocalLoss'),
    ),
>>>>>>> bf9488d7e9839e3b785703788a42532d19c19973
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=.5, score_thr=.01))
