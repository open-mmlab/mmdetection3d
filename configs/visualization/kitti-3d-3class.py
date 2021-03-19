# dataset settings
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
input_modality = dict(use_lidar=True, use_camera=False)

file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel', path_mapping=dict(data='s3://kitti_data/'))

pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        file_client_args=file_client_args),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

data = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=data_root + 'kitti_infos_trainval.pkl',
    split='training',
    pts_prefix='velodyne_reduced',
    pipeline=pipeline,
    modality=input_modality,
    classes=class_names,
    test_mode=False,
    # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
    # and box_type_3d='Depth' in sunrgbd and scannet dataset.
    box_type_3d='LiDAR',
    filter_empty_gt=False)
