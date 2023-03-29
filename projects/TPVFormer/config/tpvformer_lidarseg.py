_base_ = ['mmdet3d::_base_/default_runtime.py']

custom_imports = dict(
    imports=['projects.TPVFormer.tpvformer'], allow_failed_imports=False)

dataset_type = 'NuScenesSegDataset'
data_root = 'data/nuscenes/'
data_prefix = dict(
    pts='samples/LIDAR_TOP',
    pts_semantic_mask='lidarseg/v1.0-trainval',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT')
# class_names = [
#     'noise', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
#     'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
#     'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
#     'vegetation'
# ]
# metainfo = dict(classes=class_names)

backend_args = None

# train_pipeline = [
#     dict(type='LoadImageFromFileMono3D', backend_args=backend_args),
#     dict(
#         type='LoadPointsFromFile',
#         coord_type='LIDAR',
#         load_dim=5,
#         use_dim=3,
#         backend_args=backend_args),
#     dict(
#         type='LoadAnnotations3D',
#         with_bbox=True,
#         with_label=True,
#         with_attr_label=True,
#         with_bbox_3d=True,
#         with_label_3d=True,
#         with_bbox_depth=True),
#     dict(type='Resize', scale=(1600, 900), keep_ratio=True),
#     dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
#     dict(
#         type='Pack3DDetInputs',
#         keys=[
#             'img', 'gt_bboxes', 'gt_bboxes_labels', 'attr_labels',
#             'gt_bboxes_3d', 'gt_labels_3d', 'centers_2d', 'depths'
#         ]),
# ]

val_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=False,
        color_type='unchanged',
        num_views=6,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=3,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        with_attr_label=False,
        seg_3d_dtype='np.uint8'),
    dict(type='SegLabelMapping', ),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'pts_semantic_mask'],
        meta_keys=['lidar2img'])
]

test_pipeline = val_pipeline

# train_dataloader = dict(
#     batch_size=2,
#     num_workers=2,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_prefix=dict(
#             pts='',
#             CAM_FRONT='samples/CAM_FRONT',
#             CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
#             CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
#             CAM_BACK='samples/CAM_BACK',
#             CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
#             CAM_BACK_LEFT='samples/CAM_BACK_LEFT'),
#         ann_file='nuscenes_infos_train.pkl',
#         load_type='mv_image_based',
#         pipeline=train_pipeline,
#         metainfo=metainfo,
#         modality=input_modality,
#         test_mode=False,
#         # we use box_type_3d='Camera' in monocular 3d
#         # detection task
#         box_type_3d='Camera',
#         use_valid_flag=True,
#         backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=data_prefix,
        ann_file='nuscenes_infos_val.pkl',
        pipeline=val_pipeline,
        test_mode=True,
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(type='SegMetric')

test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# optimizer = dict(
#     type='AdamW',
#     lr=2e-4,
#     paramwise_cfg=dict(custom_keys={
#         'img_backbone': dict(lr_mult=0.1),
#     }),
#     weight_decay=0.01)

# grad_max_norm = 35

# print_freq = 50
# max_epochs = 24

load_from = 'checkpoints/tpvformer.pth'
grid_shape = [200, 200, 16]

occupancy = False
lovasz_input = 'points'
ce_input = 'voxel'

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

_dim_ = 128
num_heads = 8
_pos_dim_ = [48, 48, 32]
_ffn_dim_ = _dim_ * 2
_num_levels_ = 4
_num_cams_ = 6

tpv_h_ = 200
tpv_w_ = 200
tpv_z_ = 16
scale_h = 1
scale_w = 1
scale_z = 1
tpv_encoder_layers = 5
num_points_in_pillar = [4, 32, 32]
num_points = [8, 64, 64]
hybrid_attn_anchors = 16
hybrid_attn_points = 32
hybrid_attn_init = 0

grid_size = [tpv_h_ * scale_h, tpv_w_ * scale_w, tpv_z_ * scale_z]
nbr_class = 17

self_cross_layer = dict(
    type='TPVFormerLayer',
    attn_cfgs=[
        dict(
            type='TPVCrossViewHybridAttention',
            tpv_h=tpv_h_,
            tpv_w=tpv_w_,
            tpv_z=tpv_z_,
            num_anchors=hybrid_attn_anchors,
            embed_dims=_dim_,
            num_heads=num_heads,
            num_points=hybrid_attn_points,
            init_mode=hybrid_attn_init,
        ),
        dict(
            type='TPVImageCrossAttention',
            pc_range=point_cloud_range,
            num_cams=_num_cams_,
            deformable_attention=dict(
                type='TPVMSDeformableAttention3D',
                embed_dims=_dim_,
                num_heads=num_heads,
                num_points=num_points,
                num_z_anchors=num_points_in_pillar,
                num_levels=_num_levels_,
                floor_sampling_offset=False,
                tpv_h=tpv_h_,
                tpv_w=tpv_w_,
                tpv_z=tpv_z_,
            ),
            embed_dims=_dim_,
            tpv_h=tpv_h_,
            tpv_w=tpv_w_,
            tpv_z=tpv_z_,
        )
    ],
    feedforward_channels=_ffn_dim_,
    ffn_dropout=0.1,
    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'))

self_layer = dict(
    type='TPVFormerLayer',
    attn_cfgs=[
        dict(
            type='TPVCrossViewHybridAttention',
            tpv_h=tpv_h_,
            tpv_w=tpv_w_,
            tpv_z=tpv_z_,
            num_anchors=hybrid_attn_anchors,
            embed_dims=_dim_,
            num_heads=num_heads,
            num_points=hybrid_attn_points,
            init_mode=hybrid_attn_init,
        )
    ],
    feedforward_channels=_ffn_dim_,
    ffn_dropout=0.1,
    operation_order=('self_attn', 'norm', 'ffn', 'norm'))

model = dict(
    type='TPVFormer',
    data_preprocessor=dict(
        type='TPVFormerDataPreprocessor',
        pad_size_divisor=32,
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        voxel=True,
        voxel_type='cylindrical',
        voxel_layer=dict(
            grid_shape=grid_shape,
            point_cloud_range=point_cloud_range,
            max_num_points=-1,
            max_voxels=-1,
        ),
    ),
    use_grid_mask=True,
    backbone=dict(
        type='mmdet.ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(
            type='DCNv2', deform_groups=1, fallback_on_stride=False
        ),  # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True)),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    encoder=dict(
        type='TPVFormerEncoder',
        tpv_h=tpv_h_,
        tpv_w=tpv_w_,
        tpv_z=tpv_z_,
        num_layers=tpv_encoder_layers,
        pc_range=point_cloud_range,
        num_points_in_pillar=num_points_in_pillar,
        num_points_in_pillar_cross_view=[16, 16, 16],
        return_intermediate=False,
        transformerlayers=[
            self_cross_layer,
            self_cross_layer,
            self_cross_layer,
            self_layer,
            self_layer,
        ],
        embed_dims=_dim_,
        positional_encoding=dict(
            type='TPVFormerPositionalEncoding',
            num_feats=_pos_dim_,
            h=tpv_h_,
            w=tpv_w_,
            z=tpv_z_)),
    decode_head=dict(
        type='TPVFormerDecoder',
        tpv_h=tpv_h_,
        tpv_w=tpv_w_,
        tpv_z=tpv_z_,
        nbr_classes=nbr_class,
        in_dims=_dim_,
        hidden_dims=2 * _dim_,
        out_dims=_dim_,
        scale_h=scale_h,
        scale_w=scale_w,
        scale_z=scale_z),
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
