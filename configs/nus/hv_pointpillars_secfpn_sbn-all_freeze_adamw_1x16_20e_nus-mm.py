# model settings
voxel_size = [0.25, 0.25, 8]
point_cloud_range = [-50, -50, -5, 50, 50, 3]
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
model = dict(
    type='MVXFasterRCNNV2',
    pretrained=('./pretrain_detectron/'
                'ImageNetPretrained/MSRA/resnet50_msra.pth'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=4,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe'),
    pts_voxel_layer=dict(
        max_num_points=64,  # max_points_per_voxel
        point_cloud_range=point_cloud_range,  # velodyne coordinates, x, y, z
        voxel_size=voxel_size,
        max_voxels=(30000, 40000),  # (training, testing) max_coxels
    ),
    pts_voxel_encoder=dict(
        type='HardVFE',
        num_input_features=4,
        num_filters=[64, 64],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
        fusion_layer=dict(
            type='MultiViewPointFusion',
            img_channels=2048,
            pts_channels=64,
            mid_channels=128,
            out_channels=128,
            norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
            img_levels=[3],
            align_corners=False,
            activate_out=True,
            fuse_out=False),
    ),
    pts_middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=128,
        output_shape=[400, 400],  # checked from PointCloud3D
    ),
    pts_backbone=dict(
        type='SECOND',
        in_channels=128,
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        num_filters=[64, 128, 256],
    ),
    pts_neck=dict(
        type='SECONDFPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        num_upsample_filters=[128, 128, 128],
    ),
    pts_bbox_head=dict(
        type='Anchor3DVeloHead',
        class_names=class_names,
        num_classes=10,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        encode_bg_as_zeros=True,
        anchor_generator=dict(type='AlignedAnchorGeneratorRange', ),
        anchor_range=[
            [-50, -50, -1.80032795, 50, 50, -1.80032795],  # car
            [-50, -50, -1.74440365, 50, 50, -1.74440365],  # truck
            [-50, -50, -1.68526504, 50, 50, -1.68526504],  # trailer
            [-50, -50, -1.67339111, 50, 50, -1.67339111],  # bicycle
            [-50, -50, -1.61785072, 50, 50, -1.61785072],  # pedestrian
            [-50, -50, -1.80984986, 50, 50, -1.80984986],  # traffic_cone
            [-50, -50, -1.763965, 50, 50, -1.763965],  # barrier
        ],
        anchor_strides=[2],
        anchor_sizes=[
            [1.95017717, 4.60718145, 1.72270761],  # car
            [2.4560939, 6.73778078, 2.73004906],  # truck
            [2.87427237, 12.01320693, 3.81509561],  # trailer
            [0.60058911, 1.68452161, 1.27192197],  # bicycle
            [0.66344886, 0.7256437, 1.75748069],  # pedestrian
            [0.39694519, 0.40359262, 1.06232151],  # traffic_cone
            [2.49008838, 0.48578221, 0.98297065],  # barrier
        ],
        anchor_custom_values=[0, 0],
        anchor_rotations=[0, 1.57],
        assigner_per_size=False,
        assign_per_class=False,
        diff_rad_by_sin=True,
        dir_offset=0.7854,  # pi/4
        dir_limit_offset=0,
        bbox_coder=dict(type='ResidualCoder', ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2),
    ),
)
# model training and testing settings
train_cfg = dict(
    pts=dict(
        assigner=dict(  # for Car
            type='MaxIoUAssigner',
            iou_type='nearest_3d',
            pos_iou_thr=0.6,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        allowed_border=0,
        code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    pts=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_pre=1000,
        nms_thr=0.2,
        score_thr=0.05,
        min_bbox_size=0,
        max_per_img=500,
        post_center_limit_range=point_cloud_range,
        # TODO: check whether need to change this
        # post_center_limit_range=[-59.6, -59.6, -6, 59.6, 59.6, 4],
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
    ))

# dataset settings
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
input_modality = dict(
    use_lidar=True,
    use_radar=False,
    use_map=False,
    use_external=False,
    use_camera=True,
)
db_sampler = dict(
    root_path=data_root,
    info_path=data_root + 'nuscenes_dbinfos_train.pkl',
    rate=1.0,
    use_road_plane=False,
    object_rot_range=[0.0, 0.0],
    prepare=dict(),
    sample_groups=dict(
        bus=4,
        trailer=4,
        truck=4,
    ),
)

train_pipeline = [
    dict(
        type='Resize',
        img_scale=(1280, 720),
        ratio_range=(0.8, 1.2),
        keep_ratio=True),
    dict(
        type='GlobalRotScale',
        rot_uniform_noise=[-0.3925, 0.3925],
        scaling_uniform_noise=[0.95, 1.05],
        trans_normal_noise=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d']),
]
test_pipeline = [
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='Resize',
        img_scale=[
            (1280, 720),
        ],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points', 'img']),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        modality=input_modality,
        class_names=class_names,
        with_label=True),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        class_names=class_names,
        with_label=True),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        ann_file=data_root + 'nuscenes_infos_test.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        class_names=class_names,
        with_label=False))
# optimizer
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.01)
# max_norm=10 is better for SECOND
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[16, 19])
momentum_config = None
checkpoint_config = dict(interval=1)
# yapf:disable
evaluation = dict(interval=20)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 20
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/pp_secfpn_80e'
load_from = './pretrain_mmdet/mvx_faster_rcnn_r50_fpn_detectron2-caffe_freezeBN_l1-loss_roialign-v2_nus_1x_coco-3x-pre_ap-28.8-4e72d8c7.pth'  # noqa
resume_from = None
workflow = [('train', 1)]
