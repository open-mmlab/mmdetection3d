import numpy as np
import torch

from mmdet3d.ops.voxel.voxelize import Voxelization


def _get_same_voxel_num(coor_array, voxel):
    sum = 0
    for i in range(coor_array.shape[0]):
        if (coor_array[i] == voxel).all():
            sum += 1
    return sum


def _get_invalid_voxel_num(coor_array):
    sum = 0
    for i in range(coor_array.shape[0]):
        if (coor_array[i] == [-1, -1, -1]).any():
            sum += 1
    return sum


def test_voxelization():
    np.random.seed(0)
    voxel_size = [0.5, 0.5, 0.5]
    point_cloud_range = [0, -40, -3, 70.4, 40, 1]
    points = np.random.rand(1000, 4)
    outside_points = np.array([[100, 100, 0, 0], [-100, 100, 0, 0],
                               [100, -100, 0, 0], [-100, -100, 10, 0],
                               [100, 100, 10, 0], [100, 100, -10, 0]])
    points = np.vstack((points, outside_points)).astype('float32')
    points = torch.tensor(points)
    max_num_points = -1
    dynamic_voxelization = Voxelization(voxel_size, point_cloud_range,
                                        max_num_points)
    max_num_points = 1000
    hard_voxelization = Voxelization(voxel_size, point_cloud_range,
                                     max_num_points)
    # test hard_voxelization on cpu
    coors, voxels, num_points_per_voxel = hard_voxelization.forward(points)
    coors = coors.detach().numpy()
    voxels = voxels.detach().numpy()
    num_points_per_voxel = num_points_per_voxel.detach().numpy()
    expected_voxels = np.array([[7, 81, 1], [6, 81, 0], [7, 80, 1], [6, 81, 1],
                                [7, 81, 0], [6, 80, 1], [7, 80, 0], [6, 80,
                                                                     0]])
    expected_num_points_per_voxel = np.array(
        [120, 121, 127, 134, 115, 127, 125, 131])
    assert np.all(voxels == expected_voxels)
    assert coors.shape == (8, 1000, 4)
    assert np.all(num_points_per_voxel == expected_num_points_per_voxel)

    # test dynamic_voxelization on cpu
    max_num_points = -1
    coors = dynamic_voxelization.forward(points)
    coors = coors.detach().numpy()
    for i in range(expected_voxels.shape[0]):
        voxel_num = _get_same_voxel_num(coors, expected_voxels[i])
        assert voxel_num == expected_num_points_per_voxel[i]
    assert _get_invalid_voxel_num(coors) == 6
    assert coors.shape == (1006, 3)

    # test hard_voxelization on gpu
    points = points.to(device='cuda:0')
    coors, voxels, num_points_per_voxel = hard_voxelization.forward(points)
    coors = coors.cpu().detach().numpy()
    voxels = voxels.cpu().detach().numpy()
    num_points_per_voxel = num_points_per_voxel.cpu().detach().numpy()
    assert np.all(voxels == expected_voxels)
    assert coors.shape == (8, 1000, 4)
    assert np.all(num_points_per_voxel == expected_num_points_per_voxel)

    # test dynamic_voxelization on gpu
    coors = dynamic_voxelization.forward(points)
    coors = coors.cpu().detach().numpy()
    for i in range(expected_voxels.shape[0]):
        voxel_num = _get_same_voxel_num(coors, expected_voxels[i])
        assert voxel_num == expected_num_points_per_voxel[i]
    assert _get_invalid_voxel_num(coors) == 6
    assert coors.shape == (1006, 3)
