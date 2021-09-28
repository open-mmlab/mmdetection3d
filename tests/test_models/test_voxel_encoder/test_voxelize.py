# Copyright (c) OpenMMLab. All rights reserved.
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


def test_voxelization_nondeterministic():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')

    voxel_size = [0.5, 0.5, 0.5]
    point_cloud_range = [0, -40, -3, 70.4, 40, 1]
    data_path = './tests/data/kitti/training/velodyne_reduced/000000.bin'
    load_points_from_file = LoadPointsFromFile(
        coord_type='LIDAR', load_dim=4, use_dim=4)
    results = dict()
    results['pts_filename'] = data_path
    results = load_points_from_file(results)
    points = results['points'].tensor.numpy()

    points = torch.tensor(points)
    max_num_points = -1
    dynamic_voxelization = Voxelization(voxel_size, point_cloud_range,
                                        max_num_points)

    max_num_points = 10
    max_voxels = 50
    hard_voxelization = Voxelization(
        voxel_size,
        point_cloud_range,
        max_num_points,
        max_voxels,
        deterministic=False)

    # test hard_voxelization (non-deterministic version) on gpu
    points = torch.tensor(points).contiguous().to(device='cuda:0')
    voxels, coors, num_points_per_voxel = hard_voxelization.forward(points)
    coors = coors.cpu().detach().numpy().tolist()
    voxels = voxels.cpu().detach().numpy().tolist()
    num_points_per_voxel = num_points_per_voxel.cpu().detach().numpy().tolist()

    coors_all = dynamic_voxelization.forward(points)
    coors_all = coors_all.cpu().detach().numpy().tolist()

    coors_set = set([tuple(c) for c in coors])
    coors_all_set = set([tuple(c) for c in coors_all])

    assert len(coors_set) == len(coors)
    assert len(coors_set - coors_all_set) == 0

    points = points.cpu().detach().numpy().tolist()

    coors_points_dict = {}
    for c, ps in zip(coors_all, points):
        if tuple(c) not in coors_points_dict:
            coors_points_dict[tuple(c)] = set()
        coors_points_dict[tuple(c)].add(tuple(ps))

    for c, ps, n in zip(coors, voxels, num_points_per_voxel):
        ideal_voxel_points_set = coors_points_dict[tuple(c)]
        voxel_points_set = set([tuple(p) for p in ps[:n]])
        assert len(voxel_points_set) == n
        if n < max_num_points:
            assert voxel_points_set == ideal_voxel_points_set
            for p in ps[n:]:
                assert max(p) == min(p) == 0
        else:
            assert len(voxel_points_set - ideal_voxel_points_set) == 0

    # test hard_voxelization (non-deterministic version) on gpu
    # with all input point in range
    points = torch.tensor(points).contiguous().to(device='cuda:0')[:max_voxels]
    coors_all = dynamic_voxelization.forward(points)
    valid_mask = coors_all.ge(0).all(-1)
    points = points[valid_mask]
    coors_all = coors_all[valid_mask]
    coors_all = coors_all.cpu().detach().numpy().tolist()

    voxels, coors, num_points_per_voxel = hard_voxelization.forward(points)
    coors = coors.cpu().detach().numpy().tolist()

    coors_set = set([tuple(c) for c in coors])
    coors_all_set = set([tuple(c) for c in coors_all])

    assert len(coors_set) == len(coors) == len(coors_all_set)
