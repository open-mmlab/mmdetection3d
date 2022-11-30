_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']
backbone_norm_cfg = dict(type='LN', requires_grad=True)

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
model = dict(
    type='Petr3D',
    use_grid_mask=True,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(
            2,
            3,
        ),
        frozen_stages=-1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        with_cp=True,
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
        pretrained='ckpts/resnet50_msra-5891d200.pth',
    ),
    img_neck=dict(
        type='CPFPN', in_channels=[1024, 2048], out_channels=256, num_outs=2),
    pts_bbox_head=dict(
        type='PETRHead',
        num_classes=10,
        in_channels=256,
        num_query=900,
        LID=True,
        with_position=True,
        with_multiview=True,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        normedlinear=False,
        transformer=dict(
            type='PETRTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=True,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        bbox_coder=dict(
            type='NMSFreeCoder',
            # type='NMSFreeClsCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10),
        positional_encoding=dict(
            type='SinePositionalEncoding3D', num_feats=128, normalize=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(
                    type='IoUCost', weight=0.0
                ),  # Fake cost. Just to be compatible with DETR head.
                pc_range=point_cloud_range))))

dataset_type = 'CustomNuScenesDataset'
data_root = '/mnt/d/nus/v1.0-mini/'

file_client_args = dict(backend='disk')
db_sampler = dict()
ida_aug_conf = {
    'resize_lim': (0.8, 1.0),
    'final_dim': (512, 1408),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 900,
    'W': 1600,
    'rand_flip': True,
}
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(
        type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=True),
    dict(
        type='GlobalRotScaleTransImage',
        rot_range=[-0.3925, 0.3925],
        translation_std=[0, 0, 0],
        scale_ratio_range=[0.95, 1.05],
        reverse_angle=True,
        training=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf,
        training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality))

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.1),
    }),
    weight_decay=0.01)

optimizer_config = dict(
    type='Fp16OptimizerHook',
    loss_scale=512.,
    grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    # by_epoch=False
)
total_epochs = 24
evaluation = dict(interval=24, pipeline=test_pipeline)
find_unused_parameters = False

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from = None
resume_from = None

# mAP: 0.3174
# mATE: 0.8397
# mASE: 0.2796
# mAOE: 0.6158
# mAVE: 0.9543
# mAAE: 0.2326
# NDS: 0.3665
# Eval time: 199.1s

# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.503   0.607   0.155   0.120   1.107   0.241
# truck   0.259   0.874   0.232   0.217   0.968   0.261
# bus     0.329   0.864   0.219   0.188   2.289   0.411
# trailer 0.105   1.143   0.253   0.548   0.400   0.104
# construction_vehicle    0.071   1.233   0.503   1.216   0.122   0.349
# pedestrian      0.407   0.735   0.294   1.016   0.770   0.313
# motorcycle      0.294   0.810   0.277   0.901   1.471   0.146
# bicycle 0.290   0.698   0.260   1.176   0.509   0.036
# traffic_cone    0.497   0.608   0.322   nan     nan     nan
# barrier 0.419   0.824   0.281   0.160   nan     nan
