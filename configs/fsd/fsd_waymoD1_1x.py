_base_ = [
    '../_base_/datasets/waymoD5-3d-3class.py',
    # '../_base_/models/fsd.py',
    '../_base_/schedules/cosine_2x.py',
    '../_base_/default_runtime.py',
]

class_names = ['Car', 'Pedestrian', 'Cyclist']
num_classes = len(class_names)

point_cloud_range = [-80, -80, -2, 80, 80, 4]

dataset_type = 'WaymoDataset'
data_root = 'data/waymo/kitti_format/'
file_client_args = dict(backend='disk')


# ==================== model ../_base_/models/fsd.py

seg_voxel_size = (0.25, 0.25, 0.2)
seg_score_thresh = (0.3, 0.25, 0.25)

segmentor = dict(
    type='VoteSegmentor',
    voxel_layer=dict(  # need to refactor to Det3DDataPreprocessor
        voxel_size=seg_voxel_size,
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        max_voxels=(-1, -1)
    ),

    voxel_encoder=dict(  # need to refactor to pts_voxel_encoder
        type='DynamicScatterVFE',
        in_channels=5,
        feat_channels=[64, 64],
        voxel_size=seg_voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
        unique_once=True,
    ),

    middle_encoder=dict(
        type='PseudoMiddleEncoderForSpconvFSD',
    ),

    backbone=dict(
        type='SimpleSparseUNet',
        in_channels=64,
        sparse_shape=[32, 640, 640],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
        base_channels=64,
        output_channels=128,
        encoder_channels=((64, ), (64, 64, 64), (64, 64, 64), (128, 128, 128), (256, 256, 256)),
        encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1), (1, 1, 1)),
        decoder_channels=((256, 256, 128), (128, 128, 64), (64, 64, 64), (64, 64, 64), (64, 64, 64)),
        decoder_paddings=((1, 1), (1, 0), (1, 0), (0, 0), (0, 1)), # decoder paddings seem useless in SubMConv
    ),

    decode_neck=dict(
        type='Voxel2PointScatterNeck',
        voxel_size=seg_voxel_size,
        point_cloud_range=point_cloud_range,
    ),

    segmentation_head=dict(
        type='VoteSegHead',
        in_channel=67,
        hidden_dims=[128, 128],
        num_classes=num_classes,
        dropout_ratio=0.0,
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='naiveSyncBN1d'),
        act_cfg=dict(type='ReLU'),
        loss_decode=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=3.0,
            alpha=0.8,
            loss_weight=1.0),
        loss_vote=dict(
            type='mmdet.L1Loss',
            loss_weight=1.0),
    ),
    train_cfg=dict(
        point_loss=True,
        score_thresh=seg_score_thresh, # for training log
        class_names=('Car', 'Ped', 'Cyc'), # for training log
        centroid_offset=False,
    ),
)

model = dict(
    type='FSD',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),  # hin added
    segmentor=segmentor,
    backbone=dict(
        type='SIR',
        num_blocks=3,
        in_channels=[84,] + [133, ] * 2,
        feat_channels=[[128, 128], ] * 3,
        rel_mlp_hidden_dims=[[16, 32],] * 3,
        norm_cfg=dict(type='LN', eps=1e-3),
        mode='max',
        xyz_normalizer=[20, 20, 4],
        act='gelu',
        unique_once=True,
    ),

    bbox_head=dict(
        type='SparseClusterHeadV2',
        num_classes=num_classes,
        bbox_coder=dict(type='BasePointBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_center=dict(type='mmdet.L1Loss', loss_weight=0.5),
        loss_size=dict(type='mmdet.L1Loss', loss_weight=0.5),
        loss_rot=dict(type='mmdet.L1Loss', loss_weight=0.2),
        in_channel=128 * 3 * 2,
        shared_mlp_dims=[1024, 1024],
        train_cfg=None,
        test_cfg=None,
        norm_cfg=dict(type='LN'),
        tasks=[
            dict(class_names=['Car',]),
            dict(class_names=['Pedestrian',]),
            dict(class_names=['Cyclist',]),
        ],
        class_names=class_names,
        common_attrs=dict(
            center=(3, 2, 128), dim=(3, 2, 128), rot=(2, 2, 128),  # (out_dim, num_layers, hidden_dim)
        ),
        num_cls_layer=2,
        cls_hidden_dim=128,
        separate_head=dict(
            type='FSDSeparateHead',
            norm_cfg=dict(type='LN'),
            act='relu',
        ),
        as_rpn=True,
    ),
    roi_head=dict(
        type='GroupCorrectionHead',
        num_classes=num_classes,
        roi_extractor=dict(
             type='DynamicPointROIExtractor',
             extra_wlh=[0.5, 0.5, 0.5],
             max_inbox_point=256,
             debug=False,
        ),
        bbox_head=dict(
            type='FullySparseBboxHead',
            num_classes=num_classes,
            num_blocks=6,
            in_channels=[213, 146, 146, 146, 146, 146],
            feat_channels=[[128, 128], ] * 6,
            rel_mlp_hidden_dims=[[16, 32],] * 6,
            rel_mlp_in_channels=[13, ] * 6,
            reg_mlp=[512, 512],
            cls_mlp=[512, 512],
            mode='max',
            xyz_normalizer=[20, 20, 4],
            act='gelu',
            geo_input=True,
            with_corner_loss=True,
            corner_loss_weight=1.0,
            bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
            norm_cfg=dict(type='LN', eps=1e-3),
            unique_once=True,

            loss_bbox=dict(
                type='mmdet.L1Loss',
                reduction='mean',
                loss_weight=2.0),

            loss_cls=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                reduction='mean',
                loss_weight=1.0),
            cls_dropout=0.1,
            reg_dropout=0.1,
        ),
        train_cfg=None,
        test_cfg=None,
        init_cfg=None
    ),

    train_cfg=dict(
        score_thresh=seg_score_thresh,
        sync_reg_avg_factor=True,
        pre_voxelization_size=(0.1, 0.1, 0.1),
        disable_pretrain=True,
        disable_pretrain_topks=[600, 200, 200],
        rpn=dict(
            use_rotate_nms=True,
            nms_pre=-1,
            nms_thr=None,
            score_thr=0.1,
            min_bbox_size=0,
            max_num=500,
        ),
        rcnn=dict(
            assigner=[
                dict( # Car
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.45,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    ignore_iof_thr=-1
                ),
                dict( # Ped
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.35,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1
                ),
                dict( # Cyc
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.35,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1
                ),
            ],

            sampler=dict(
                type='IoUNegPiecewiseSampler',
                num=256,
                pos_fraction=0.55,
                neg_piece_fractions=[0.8, 0.2],
                neg_iou_piece_thrs=[0.55, 0.1],
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
                return_iou=True
            ),
            cls_pos_thr=(0.8, 0.65, 0.65),
            cls_neg_thr=(0.2, 0.15, 0.15),
            sync_reg_avg_factor=True,
            sync_cls_avg_factor=True,
            corner_loss_only_car=True,
            class_names=class_names,
        )
    ),
    test_cfg=dict(
        score_thresh=seg_score_thresh,
        pre_voxelization_size=(0.1, 0.1, 0.1),
        skip_rcnn=False,
        rpn=dict(
            use_rotate_nms=True,
            nms_pre=-1,
            nms_thr=0.25,
            score_thr=0.1,
            min_bbox_size=0,
            max_num=500,
        ),
        rcnn=dict(
            use_rotate_nms=True,
            nms_pre=-1,
            nms_thr=0.25,
            score_thr=0.1,
            min_bbox_size=0,
            max_num=500,
        ),
    ),
    cluster_assigner=dict(
        cluster_voxel_size=dict(
            Car=(0.3, 0.3, 6),
            Cyclist=(0.2, 0.2, 6),
            Pedestrian=(0.05, 0.05, 6),
        ),
        min_points=2,
        point_cloud_range=point_cloud_range,
        connected_dist=dict(
            Car=0.6,
            Cyclist=0.4,
            Pedestrian=0.1,
        ),  # xy-plane distance
        class_names=class_names,
    ),
)

# ==================== setting

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'waymo_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5)),
    classes=class_names,
    sample_groups=dict(Car=5, Pedestrian=5, Cyclist=3),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6, # 5
        use_dim=5)) # [0,1,2,3,4]

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=5),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0.2]),
    dict(type='PointsRangeFilter', point_cloud_range=_base_.point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=_base_.point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='Pack3DDetInputs', keys=['points'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=5,
        file_client_args=file_client_args),
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
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=_base_.point_cloud_range),
            dict(type='Pack3DDetInputs', keys=['points'])
        ])
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='waymo_infos_train.pkl',
            data_prefix=dict(
                pts='training/velodyne', sweeps='training/velodyne'),
            pipeline=train_pipeline,
            load_interval=1)))


val_dataloader = dict(
    dataset=dict(pipeline=test_pipeline, metainfo=dict(classes=class_names)))
test_dataloader = dict(
    dataset=dict(pipeline=test_pipeline, metainfo=dict(classes=class_names)))

custom_hooks = [
    dict(type='DisableAugmentationHook', num_last_epochs=1, skip_type_keys=('ObjectSample', 'RandomFlip3D', 'GlobalRotScaleTrans')),
    dict(type='EnableFSDDetectionHookIter', enable_after_iter=4000, threshold_buffer=0.3, buffer_iter=8000) 
]

optim_wrapper = dict(optimizer=dict(lr=3e-5))

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=12)

evaluation = dict(interval=12, pipeline=eval_pipeline)
