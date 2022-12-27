_base_ = [
    # '.../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    './detr3d_res101_gridmask.py',
    'mmdet3d::configs/_base_/default_runtime.py'
]

custom_imports = dict(imports=['projects.detr3d'])
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(mean=[103.530, 116.280, 123.675],
                    std=[57.375, 57.120, 58.395],
                    bgr_to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

input_modality = dict(use_lidar=False,
                      use_camera=True,
                      use_radar=False,
                      use_map=False,
                      use_external=False)
# this means type='Detr3D' will be processed as 'mmdet3d.Detr3D'
default_scope = 'mmdet3d'
model = dict(type='Detr3D',
             use_grid_mask=True,
             data_preprocessor=dict(type='Det3DDataPreprocessor',
                                    **img_norm_cfg,
                                    pad_size_divisor=32),
             img_backbone=dict(
                 type='VoVNet',
                 spec_name='V-99-eSE',
                 norm_eval=True,
                 frozen_stages=1,
                 input_ch=3,
                 out_features=['stage2', 'stage3', 'stage4', 'stage5']),
             img_neck=dict(type='mmdet.FPN',
                           in_channels=[256, 512, 768, 1024],
                           out_channels=256,
                           start_level=0,
                           add_extra_convs='on_output',
                           num_outs=4,
                           relu_before_extra_convs=True))

dataset_type = 'NuScenesDataset'
data_root = 'data/nus_v2/'

test_transforms = [
    dict(type='RandomResize3D',
         scale=(1600, 900),
         ratio_range=(1., 1.),
         keep_ratio=True)
]
train_transforms = test_transforms

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, num_views=6),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D',
         with_bbox_3d=True,
         with_label_3d=True,
         with_attr_label=False),
    dict(type='MultiViewWrapper', transforms=train_transforms),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='Pack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

metainfo = dict(classes=class_names)
data_prefix = dict(pts='',
                   CAM_FRONT='samples/CAM_FRONT',
                   CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
                   CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
                   CAM_BACK='samples/CAM_BACK',
                   CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
                   CAM_BACK_LEFT='samples/CAM_BACK_LEFT')

train_dataloader = dict(dataset=dict(
    type='CBGSDataset',
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_trainval.pkl',
        pipeline=train_pipeline,
        load_type='frame_based',
        metainfo=metainfo,
        modality=input_modality,
        test_mode=False,
        data_prefix=data_prefix,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR')))
