_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-3class.py', '../_base_/schedules/cosine.py',
    '../_base_/default_runtime.py'
]

point_cloud_range = [0, -40, -3, 70.4, 40, 1]
voxel_size = [0.05, 0.05, 0.1]

model = dict(
    type='DynamicVoxelNet',
    data_preprocessor=dict(
        voxel_type='dynamic',
        voxel_layer=dict(
            _delete_=True,
            max_num_points=-1,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(-1, -1))),
    voxel_encoder=dict(
        _delete_=True,
        type='DynamicSimpleVFE',
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range))
