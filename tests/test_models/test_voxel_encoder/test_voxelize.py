import numpy as np
import pytest
import torch

from mmdet3d.core.voxel.voxel_generator import VoxelGenerator
from mmdet3d.datasets.pipelines import LoadPointsFromFile
from mmdet3d.ops.voxel.voxelize import Voxelization


def _get_voxel_points_indices(points, coors, voxel):
    result_form = np.equal(coors, voxel)
    return result_form[:, 0] & result_form[:, 1] & result_form[:, 2]


def test_voxelization():
    voxel_size = [0.5, 0.5, 0.5]
    point_cloud_range = [0, -40, -3, 70.4, 40, 1]
    max_num_points = 1000
    self = VoxelGenerator(voxel_size, point_cloud_range, max_num_points)
    data_path = './tests/data/kitti/training/velodyne_reduced/000000.bin'
    load_points_from_file = LoadPointsFromFile(
        coord_type='LIDAR', load_dim=4, use_dim=4)
    results = dict()
    results['pts_filename'] = data_path
    results = load_points_from_file(results)
    points = results['points'].tensor.numpy()
    voxels_generator = self.generate(points)
    coors, voxels, num_points_per_voxel = voxels_generator
    expected_coors = coors
    expected_voxels = voxels
    expected_num_points_per_voxel = num_points_per_voxel

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
    assert np.all(coors == expected_coors)
    assert np.all(voxels == expected_voxels)
    assert np.all(num_points_per_voxel == expected_num_points_per_voxel)

    # test dynamic_voxelization on cpu
    coors = dynamic_voxelization.forward(points)
    coors = coors.detach().numpy()
    points = points.detach().numpy()
    for i in range(expected_voxels.shape[0]):
        indices = _get_voxel_points_indices(points, coors, expected_voxels[i])
        num_points_current_voxel = points[indices].shape[0]
        assert num_points_current_voxel > 0
        assert np.all(
            points[indices] == expected_coors[i][:num_points_current_voxel])
        assert num_points_current_voxel == expected_num_points_per_voxel[i]

    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    # test hard_voxelization on gpu
    points = torch.tensor(points).contiguous().to(device='cuda:0')
    coors, voxels, num_points_per_voxel = hard_voxelization.forward(points)
    coors = coors.cpu().detach().numpy()
    voxels = voxels.cpu().detach().numpy()
    num_points_per_voxel = num_points_per_voxel.cpu().detach().numpy()
    assert np.all(coors == expected_coors)
    assert np.all(voxels == expected_voxels)
    assert np.all(num_points_per_voxel == expected_num_points_per_voxel)

    # test dynamic_voxelization on gpu
    coors = dynamic_voxelization.forward(points)
    coors = coors.cpu().detach().numpy()
    points = points.cpu().detach().numpy()
    for i in range(expected_voxels.shape[0]):
        indices = _get_voxel_points_indices(points, coors, expected_voxels[i])
        num_points_current_voxel = points[indices].shape[0]
        assert num_points_current_voxel > 0
        assert np.all(
            points[indices] == expected_coors[i][:num_points_current_voxel])
        assert num_points_current_voxel == expected_num_points_per_voxel[i]
