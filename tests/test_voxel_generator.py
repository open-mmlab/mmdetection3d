import numpy as np

from mmdet3d.core.voxel.voxel_generator import VoxelGenerator


def test_voxel_generator():
    voxel_size = [0.5, 0.5, 0.5]
    point_cloud_range = [0, -40, -3, 70.4, 40, 1]
    max_num_points = 1000
    self = VoxelGenerator(voxel_size, point_cloud_range, max_num_points)
    points = np.random.rand(1000, 4)
    voxels = self.generate(points)
    coors, voxels, num_points_per_voxel = voxels
    assert coors.shape == (8, 1000, 4)
    assert voxels.shape == (8, 3)
    assert num_points_per_voxel.shape == (8, )
