# PandaSet dataset settings for 3D object detection
# PandaSet features dual LiDAR (Pandar64 + PandarGT) in world coordinates.
# Points are transformed to ego-centric frame on-the-fly
# by LoadPointsFromPandaSet.
point_cloud_range = [-50, -50, -5, 50, 50, 3]

# Use the same 10 classes as nuScenes for cross-dataset evaluation
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
metainfo = dict(classes=class_names)
dataset_type = 'PandaSetDataset'
data_root = 'data/pandaset/'
input_modality = dict(use_lidar=True, use_camera=False)

test_pipeline = [
    dict(type='LoadPointsFromPandaSet', coord_type='LIDAR', use_dim=4),
    dict(type='Pack3DDetInputs', keys=['points']),
]

test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='pandaset_pandar64_infos_test.pkl',
        data_prefix=dict(pts=''),
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR'))

test_evaluator = dict(
    type='PandaSetMetric',
    ann_file=data_root + 'pandaset_pandar64_infos_test.pkl')
test_cfg = dict()

val_dataloader = test_dataloader
val_cfg = dict()
val_evaluator = test_evaluator
