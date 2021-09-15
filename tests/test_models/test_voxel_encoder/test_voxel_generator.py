# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmdet3d.core.voxel.voxel_generator import VoxelGenerator


def test_voxel_generator():
    np.random.seed(0)
    voxel_size = [0.5, 0.5, 0.5]
    point_cloud_range = [0, -40, -3, 70.4, 40, 1]
    max_num_points = 1000
    self = VoxelGenerator(voxel_size, point_cloud_range, max_num_points)
    points = np.random.rand(1000, 4)
    voxels = self.generate(points)
    voxels, coors, num_points_per_voxel = voxels
    expected_coors = np.array([[7, 81, 1], [6, 81, 0], [7, 80, 1], [6, 81, 1],
                               [7, 81, 0], [6, 80, 1], [7, 80, 0], [6, 80, 0]])
    expected_num_points_per_voxel = np.array(
        [120, 121, 127, 134, 115, 127, 125, 131])
    assert voxels.shape == (8, 1000, 4)
    assert np.all(coors == expected_coors)
    assert np.all(num_points_per_voxel == expected_num_points_per_voxel)
