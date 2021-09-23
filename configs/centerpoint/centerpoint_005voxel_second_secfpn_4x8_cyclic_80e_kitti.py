_base_ = [
    '../_base_/datasets/kitti-3d-3class.py',
    '../_base_/models/centerpoint_005voxel_second_secfpn_kitti.py',
    '../_base_/schedules/cyclic_80e.py', '../_base_/default_runtime.py'
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [0, -40, -3, 70.4, 40, 1]

# Add 'point_cloud_range' into model config according to dataset
model = dict(
    pts_voxel_layer=dict(point_cloud_range=point_cloud_range),
    pts_bbox_head=dict(bbox_coder=dict(pc_range=point_cloud_range[:2])),
    # model training and testing settings
    train_cfg=dict(pts=dict(point_cloud_range=point_cloud_range)),
    test_cfg=dict(pts=dict(pc_range=point_cloud_range[:2])))
