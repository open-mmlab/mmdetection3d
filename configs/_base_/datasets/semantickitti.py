# dataset settings
dataset_type = 'SemanticKITTIDataset'
data_root = 'data/semantickitti/'
class_names = [
    'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person',
    'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground',
    'building', 'fence', 'vegetation', 'trunck', 'terrian', 'pole',
    'traffic-sign'
]
palette = [
    [174, 199, 232],
    [152, 223, 138],
    [31, 119, 180],
    [255, 187, 120],
    [188, 189, 34],
    [140, 86, 75],
    [255, 152, 150],
    [214, 39, 40],
    [197, 176, 213],
    [148, 103, 189],
    [196, 156, 148],
    [23, 190, 207],
    [247, 182, 210],
    [219, 219, 141],
    [255, 127, 14],
    [158, 218, 229],
    [44, 160, 44],
    [112, 128, 144],
    [227, 119, 194],
    [82, 84, 163],
]

labels_map = {
    0: 0,  # "unlabeled"
    1: 0,  # "outlier" mapped to "unlabeled" --------------mapped
    10: 1,  # "car"
    11: 2,  # "bicycle"
    13: 5,  # "bus" mapped to "other-vehicle" --------------mapped
    15: 3,  # "motorcycle"
    16: 5,  # "on-rails" mapped to "other-vehicle" ---------mapped
    18: 4,  # "truck"
    20: 5,  # "other-vehicle"
    30: 6,  # "person"
    31: 7,  # "bicyclist"
    32: 8,  # "motorcyclist"
    40: 9,  # "road"
    44: 10,  # "parking"
    48: 11,  # "sidewalk"
    49: 12,  # "other-ground"
    50: 13,  # "building"
    51: 14,  # "fence"
    52: 0,  # "other-structure" mapped to "unlabeled" ------mapped
    60: 9,  # "lane-marking" to "road" ---------------------mapped
    70: 15,  # "vegetation"
    71: 16,  # "trunk"
    72: 17,  # "terrain"
    80: 18,  # "pole"
    81: 19,  # "traffic-sign"
    99: 0,  # "other-object" to "unlabeled" ----------------mapped
    252: 1,  # "moving-car" to "car" ------------------------mapped
    253: 7,  # "moving-bicyclist" to "bicyclist" ------------mapped
    254: 6,  # "moving-person" to "person" ------------------mapped
    255: 8,  # "moving-motorcyclist" to "motorcyclist" ------mapped
    256: 5,  # "moving-on-rails" mapped to "other-vehic------mapped
    257: 5,  # "moving-bus" mapped to "other-vehicle" -------mapped
    258: 4,  # "moving-truck" to "truck" --------------------mapped
    259: 5  # "moving-other"-vehicle to "other-vehicle"-----mapped
}

metainfo = dict(
    classes=class_names,
    palette=palette,
    seg_label_mapping=labels_map,
    max_label=259)

input_modality = dict(use_lidar=True, use_camera=False)

file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/semantickitti/':
#         's3://semantickitti/',
#     }))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='LoadAnnotations3D',
        with_seg_3d=True,
        seg_offset=2**16,
        dataset_type='semantickitti'),
    dict(type='PointSegClassMapping', ),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1],
    ),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='LoadAnnotations3D',
        with_seg_3d=True,
        seg_offset=2**16,
        dataset_type='semantickitti'),
    dict(type='PointSegClassMapping', ),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='LoadAnnotations3D',
        with_seg_3d=True,
        seg_offset=2**16,
        dataset_type='semantickitti'),
    dict(type='PointSegClassMapping', ),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='train_infos.pkl',
            pipeline=train_pipeline,
            metainfo=metainfo,
            modality=input_modality)),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='valid_infos.pkl',
            pipeline=test_pipeline,
            metainfo=metainfo,
            modality=input_modality,
            test_mode=True,
        )),
)

val_dataloader = test_dataloader

val_evaluator = dict(type='SegMetric')
test_evaluator = val_evaluator
