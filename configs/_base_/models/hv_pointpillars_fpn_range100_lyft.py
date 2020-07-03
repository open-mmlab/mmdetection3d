_base_ = './hv_pointpillars_fpn_nus.py'

# model settings (based on nuScenes model settings)
# Voxel size for voxel encoder
# Usually voxel size is changed consistently with the point cloud range
# If point cloud range is modified, do remember to change all related
# keys in the config.
model = dict(
    pts_voxel_layer=dict(
        max_num_points=20,
        point_cloud_range=[-100, -100, -5, 100, 100, 3],
        max_voxels=(60000, 60000)),
    pts_voxel_encoder=dict(
        feat_channels=[64], point_cloud_range=[-100, -100, -5, 100, 100, 3]),
    pts_middle_encoder=dict(output_shape=[800, 800]),
    pts_bbox_head=dict(
        num_classes=9,
        anchor_generator=dict(
            ranges=[[-100, -100, -1.8, 100, 100, -1.8]], custom_values=[]),
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=7)))
# model training settings (based on nuScenes model settings)
train_cfg = dict(pts=dict(code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
