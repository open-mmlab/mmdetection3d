_base_ = '../second/hv_second_secfpn_6x8_80e_kitti-3d-car.py'

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
