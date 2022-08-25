_base_ = '../pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'

voxel_size = [0.16, 0.16, 4]
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]

model = dict(
    type='DynamicVoxelNet',
    data_preprocessor=dict(
        voxel_type='dynamic',
        voxel_layer=dict(
            max_num_points=-1,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(-1, -1))),
    voxel_encoder=dict(
        type='DynamicPillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range))
