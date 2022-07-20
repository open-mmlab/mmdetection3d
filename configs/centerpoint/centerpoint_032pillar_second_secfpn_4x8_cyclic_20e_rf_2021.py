_base_ = [
    '../_base_/models/centerpoint_032pillar_second_secfpn_rf_2021.py',
    '../_base_/datasets/rf2021-3d-2class.py',
    '../_base_/schedules/cyclic_40e.py',
    '../_base_/default_runtime.py',
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range=[-60, -103.84, -3, 62.88, 60, 1]
# For nuScenes we usually do 10-class detection
class_names = ['Car', 'Pedestrian']

dataset_type = 'Custom3DDataset'
data_root = 'data/rf2021/'
file_client_args = dict(backend='disk')

model = dict(
    pts_voxel_layer=dict(point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(point_cloud_range=point_cloud_range),
    pts_bbox_head=dict(bbox_coder=dict(pc_range=point_cloud_range[:2])),
    # model training and testing settings
    train_cfg=dict(pts=dict(point_cloud_range=point_cloud_range)),
    test_cfg=dict(pts=dict(pc_range=point_cloud_range[:2])))

# db_sampler = dict(
#     data_root=data_root,
#     info_path=data_root + 'rf2021_dbinfos_train.pkl',
#     rate=1.0,
#     prepare=dict(
#         filter_by_difficulty=[-1],
#         filter_by_min_points=dict(Car=15, Pedestrian=10)),
#     classes=class_names,
#     sample_groups=dict(Car=10, Pedestrian=10))


# train_pipeline = [
#     dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
#     dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
#     dict(type='ObjectSample', db_sampler=db_sampler, use_ground_plane=True),
#     dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
#     dict(
#         type='GlobalRotScaleTrans',
#         rot_range=[-0.78539816, 0.78539816],
#         scale_ratio_range=[0.95, 1.05]),
#     dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
#     dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
#     dict(type='PointShuffle'),
#     dict(type='DefaultFormatBundle3D', class_names=class_names),
#     dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
# ]

# test_pipeline = [
#     dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
#     dict(
#         type='MultiScaleFlipAug3D',
#         img_scale=(1333, 800),
#         pts_scale_ratio=1,
#         flip=False,
#         transforms=[
#             dict(
#                 type='GlobalRotScaleTrans',
#                 rot_range=[0, 0],
#                 scale_ratio_range=[1., 1.],
#                 translation_std=[0, 0, 0]),
#             dict(type='RandomFlip3D'),
#             dict(
#                 type='PointsRangeFilter', point_cloud_range=point_cloud_range),
#             dict(
#                 type='DefaultFormatBundle3D',
#                 class_names=class_names,
#                 with_label=False),
#             dict(type='Collect3D', keys=['points'])
#         ])
# ]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)

data = dict(
    train=dict(dataset=dict(pipeline=train_pipeline, classes=class_names)),
    val=dict(pipeline=test_pipeline, classes=class_names),
    test=dict(pipeline=test_pipeline, classes=class_names))