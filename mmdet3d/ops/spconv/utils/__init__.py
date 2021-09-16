# Copyright 2019 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
from spconv.spconv_utils import (points_to_voxel_3d_np,
                                 points_to_voxel_3d_np_mean,
                                 points_to_voxel_3d_with_filtering)


def points_to_voxel(points,
                    voxel_size,
                    coors_range,
                    coor_to_voxelidx,
                    max_points=35,
                    max_voxels=20000,
                    full_mean=False,
                    block_filtering=True,
                    block_factor=1,
                    block_size=8,
                    height_threshold=0.2,
                    height_high_threshold=3.0,
                    pad_output=False):
    """convert 3d points(N, >=3) to voxels. This version calculate everything
    in one loop. now it takes only 0.8ms(~6k voxels) with c++ and 3.2ghz cpu.

    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coors_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        coor_to_voxelidx: int array. used as a dense map.
        max_points: int. indicate maximum points contained in a voxel.
        max_voxels: int. indicate maximum voxels this function create.
            for voxelnet, 20000 is a good choice. you should shuffle points
            before call this function because max_voxels may drop some points.
        full_mean: bool. if true, all empty points in voxel will be filled with
         mean of exist points.
        block_filtering: filter voxels by height. used for lidar point cloud.
            use some visualization tool to see filtered result.
    Returns:
        voxels: [M, max_points, ndim] float tensor. only contain points.
        coordinates: [M, 3] int32 tensor. zyx format.
        num_points_per_voxel: [M] int32 tensor.
    """
    if full_mean:
        assert block_filtering is False
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    voxelmap_shape = voxelmap_shape[::-1]
    num_points_per_voxel = np.zeros(shape=(max_voxels, ), dtype=np.int32)
    voxels = np.zeros(
        shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype)
    voxel_point_mask = np.zeros(
        shape=(max_voxels, max_points), dtype=points.dtype)
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
    res = {
        'voxels': voxels,
        'coordinates': coors,
        'num_points_per_voxel': num_points_per_voxel,
        'voxel_point_mask': voxel_point_mask,
    }
    if full_mean:
        means = np.zeros(
            shape=(max_voxels, points.shape[-1]), dtype=points.dtype)
        voxel_num = points_to_voxel_3d_np_mean(points, voxels,
                                               voxel_point_mask, means, coors,
                                               num_points_per_voxel,
                                               coor_to_voxelidx,
                                               voxel_size.tolist(),
                                               coors_range.tolist(),
                                               max_points, max_voxels)
    else:
        if block_filtering:
            block_shape = [*voxelmap_shape[1:]]
            block_shape = [b // block_factor for b in block_shape]
            mins = np.full(block_shape, 99999999, dtype=points.dtype)
            maxs = np.full(block_shape, -99999999, dtype=points.dtype)
            voxel_mask = np.zeros((max_voxels, ), dtype=np.int32)
            voxel_num = points_to_voxel_3d_with_filtering(
                points, voxels, voxel_point_mask, voxel_mask, mins, maxs,
                coors, num_points_per_voxel, coor_to_voxelidx,
                voxel_size.tolist(), coors_range.tolist(), max_points,
                max_voxels, block_factor, block_size, height_threshold,
                height_high_threshold)
            voxel_mask = voxel_mask.astype(np.bool_)
            coors_ = coors[voxel_mask]
            if pad_output:
                res['coordinates'][:voxel_num] = coors_
                res['voxels'][:voxel_num] = voxels[voxel_mask]
                res['voxel_point_mask'][:voxel_num] = voxel_point_mask[
                    voxel_mask]

                res['num_points_per_voxel'][:voxel_num] = num_points_per_voxel[
                    voxel_mask]
                res['coordinates'][voxel_num:] = 0
                res['voxels'][voxel_num:] = 0
                res['num_points_per_voxel'][voxel_num:] = 0
                res['voxel_point_mask'][voxel_num:] = 0
            else:
                res['coordinates'] = coors_
                res['voxels'] = voxels[voxel_mask]
                res['num_points_per_voxel'] = num_points_per_voxel[voxel_mask]
                res['voxel_point_mask'] = voxel_point_mask[voxel_mask]
            voxel_num = coors_.shape[0]
        else:
            voxel_num = points_to_voxel_3d_np(points, voxels, voxel_point_mask,
                                              coors, num_points_per_voxel,
                                              coor_to_voxelidx,
                                              voxel_size.tolist(),
                                              coors_range.tolist(), max_points,
                                              max_voxels)
    res['voxel_num'] = voxel_num
    res['voxel_point_mask'] = res['voxel_point_mask'].reshape(
        -1, max_points, 1)
    return res


class VoxelGenerator:

    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 max_num_points,
                 max_voxels=20000,
                 full_mean=True):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = (point_cloud_range[3:] -
                     point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)
        voxelmap_shape = tuple(np.round(grid_size).astype(np.int32).tolist())
        voxelmap_shape = voxelmap_shape[::-1]

        self._coor_to_voxelidx = np.full(voxelmap_shape, -1, dtype=np.int32)
        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._max_num_points = max_num_points
        self._max_voxels = max_voxels
        self._grid_size = grid_size
        self._full_mean = full_mean

    def generate(self, points, max_voxels=None):
        res = points_to_voxel(points, self._voxel_size,
                              self._point_cloud_range, self._coor_to_voxelidx,
                              self._max_num_points, max_voxels
                              or self._max_voxels, self._full_mean)
        voxels = res['voxels']
        coors = res['coordinates']
        num_points_per_voxel = res['num_points_per_voxel']
        voxel_num = res['voxel_num']
        coors = coors[:voxel_num]
        voxels = voxels[:voxel_num]
        num_points_per_voxel = num_points_per_voxel[:voxel_num]

        return (voxels, coors, num_points_per_voxel)

    def generate_multi_gpu(self, points, max_voxels=None):
        res = points_to_voxel(points, self._voxel_size,
                              self._point_cloud_range, self._coor_to_voxelidx,
                              self._max_num_points, max_voxels
                              or self._max_voxels, self._full_mean)
        voxels = res['voxels']
        coors = res['coordinates']
        num_points_per_voxel = res['num_points_per_voxel']
        return (voxels, coors, num_points_per_voxel)

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self):
        return self._max_num_points

    @property
    def point_cloud_range(self):
        return self._point_cloud_range

    @property
    def grid_size(self):
        return self._grid_size


class VoxelGeneratorV2:

    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 max_num_points,
                 max_voxels=20000,
                 full_mean=False,
                 block_filtering=False,
                 block_factor=8,
                 block_size=3,
                 height_threshold=0.1,
                 height_high_threshold=2.0):
        assert full_mean is False, "don't use this."
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = (point_cloud_range[3:] -
                     point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)
        if block_filtering:
            assert block_size > 0
            assert grid_size[0] % block_factor == 0
            assert grid_size[1] % block_factor == 0

        voxelmap_shape = tuple(np.round(grid_size).astype(np.int32).tolist())
        voxelmap_shape = voxelmap_shape[::-1]
        self._coor_to_voxelidx = np.full(voxelmap_shape, -1, dtype=np.int32)
        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._max_num_points = max_num_points
        self._max_voxels = max_voxels
        self._grid_size = grid_size
        self._full_mean = full_mean
        self._block_filtering = block_filtering
        self._block_factor = block_factor
        self._height_threshold = height_threshold
        self._block_size = block_size
        self._height_high_threshold = height_high_threshold

    def generate(self, points, max_voxels=None):
        res = points_to_voxel(points, self._voxel_size,
                              self._point_cloud_range, self._coor_to_voxelidx,
                              self._max_num_points, max_voxels
                              or self._max_voxels, self._full_mean,
                              self._block_filtering, self._block_factor,
                              self._block_size, self._height_threshold,
                              self._height_high_threshold)
        for k, v in res.items():
            if k != 'voxel_num':
                res[k] = v[:res['voxel_num']]
        return res

    def generate_multi_gpu(self, points, max_voxels=None):
        res = points_to_voxel(
            points,
            self._voxel_size,
            self._point_cloud_range,
            self._coor_to_voxelidx,
            self._max_num_points,
            max_voxels or self._max_voxels,
            self._full_mean,
            self._block_filtering,
            self._block_factor,
            self._block_size,
            self._height_threshold,
            self._height_high_threshold,
            pad_output=True)
        return res

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self):
        return self._max_num_points

    @property
    def point_cloud_range(self):
        return self._point_cloud_range

    @property
    def grid_size(self):
        return self._grid_size
