"""PointPillars nuScenes-trained model evaluated on PandaSet (cross-dataset).

Uses LoadPointsFromPandaSet to read original PandaSet .pkl files on-the-fly,
avoiding the need to pre-convert ~22 GB of point cloud data to .bin format.

Config for Pandar64 (360 degree lidar). For PandarGT, use the pandargt config.
"""

_base_ = [
    '../_base_/models/pointpillars_hv_fpn_nus.py',
    '../_base_/default_runtime.py',
]

# ── Dataset settings ─────────────────────────────────────────────────────────
data_root = 'data/pandaset/'
ann_file = 'pandaset_pandar64_infos_test.pkl'

class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
point_cloud_range = [-50, -50, -5, 50, 50, 3]
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=class_names)

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
        type='PandaSetDataset',
        data_root=data_root,
        ann_file=ann_file,
        data_prefix=dict(pts=''),
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR'))

test_evaluator = dict(type='PandaSetMetric', ann_file=data_root + ann_file)
test_cfg = dict()

val_dataloader = test_dataloader
val_cfg = dict()
val_evaluator = test_evaluator
