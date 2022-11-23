# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmdet3d.models.task_modules.voxel import VoxelGenerator


def test_voxel_generator():
    np.random.seed(0)
    voxel_size = [5, 5, 1]
    point_cloud_range = [0, 0, 0, 20, 40, 4]
    max_num_points = 5
    self = VoxelGenerator(voxel_size, point_cloud_range, max_num_points)
    points = np.random.uniform(0, 4, (20, 3))
    voxels = self.generate(points)
    voxels, coors, num_points_per_voxel = voxels
    expected_coors = np.array([[2, 0, 0], [3, 0, 0], [0, 0, 0], [1, 0, 0]])
    expected_num_points_per_voxel = np.array([5, 5, 5, 3])
    assert voxels.shape == (4, 5, 3)
    assert np.all(coors == expected_coors)
    assert np.all(num_points_per_voxel == expected_num_points_per_voxel)
