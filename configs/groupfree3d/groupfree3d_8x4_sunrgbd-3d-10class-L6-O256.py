_base_ = [
    '../_base_/datasets/sunrgbd-3d-10class.py',
    '../_base_/models/groupfree3d.py', '../_base_/schedules/schedule_3x.py',
    '../_base_/default_runtime.py'
]
# model settings
model = dict(
    bbox_head=dict(
        num_classes=10,
        size_cls_agnostic=True,
        bbox_coder=dict(
            type='GroupFree3DBBoxCoder',
            num_sizes=10,
            num_dir_bins=12,
            with_rot=True,
            size_cls_agnostic=True,
            axis_aligned_lw=True,
            mean_sizes=[[2.114256, 1.620300, 0.927272],
                        [0.791118, 1.279516, 0.718182],
                        [0.923508, 1.867419, 0.845495],
                        [0.591958, 0.552978, 0.827272],
                        [0.699104, 0.454178, 0.75625],
                        [0.69519, 1.346299, 0.736364],
                        [0.528526, 1.002642, 1.172878],
                        [0.500618, 0.632163, 0.683424],
                        [0.404671, 1.071108, 1.688889],
                        [0.76584, 1.398258, 0.472728]]),
        sampling_objectness_loss=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        objectness_loss=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=4.0),
        center_loss=dict(
            type='SmoothL1Loss',
            beta=1.0 / 9.0,
            reduction='sum',
            loss_weight=10.0),
        dir_class_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        dir_res_loss=dict(
            type='SmoothL1Loss',
            beta=0.04,
            reduction='sum',
            loss_weight=10.0 * 0.04),
        size_reg_loss=dict(
            type='SmoothL1Loss',
            beta=0.0625,
            reduction='sum',
            loss_weight=10.0 * 0.0625),
        semantic_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0)),
    test_cfg=dict(
        sample_mod='kps',
        nms_thr=0.25,
        score_thr=0.0,
        per_class_proposal=True,
        prediction_stages='all'))

dataset_type = 'SUNRGBDDataset'
data_root = 'data/sunrgbd/'
class_names = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
               'night_stand', 'bookshelf', 'bathtub')
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        load_dim=6,
        use_dim=[0, 1, 2]),
    dict(type='LoadAnnotations3D'),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
    ),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.523599, 0.523599],
        scale_ratio_range=[0.85, 1.15]),
    dict(type='PointSample', num_points=20000),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        load_dim=6,
        use_dim=[0, 1, 2]),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.5,
            ),
            dict(type='PointSample', num_points=20000),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'sunrgbd_infos_train.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            filter_empty_gt=False,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='Depth')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'sunrgbd_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'sunrgbd_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'))

# optimizer
lr = 0.004
optimizer = dict(
    lr=lr,
    weight_decay=0.00000001,
    paramwise_cfg=dict(
        custom_keys={
            'bbox_head.decoder_layers': dict(lr_mult=0.05, decay_mult=1.0),
            'bbox_head.decoder_self_posembeds': dict(
                lr_mult=0.05, decay_mult=1.0),
            'bbox_head.decoder_cross_posembeds': dict(
                lr_mult=0.05, decay_mult=1.0),
            'bbox_head.decoder_query_proj': dict(lr_mult=0.05, decay_mult=1.0),
            'bbox_head.decoder_key_proj': dict(lr_mult=0.05, decay_mult=1.0)
        }))

optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[84, 96, 108])

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=120)

evaluation = dict(axis_aligned_lw=True)
checkpoint_config = dict(interval=1, max_keep_ckpts=10)
