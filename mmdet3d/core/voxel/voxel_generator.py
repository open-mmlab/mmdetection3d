# Copyright (c) OpenMMLab. All rights reserved.
import numba
import numpy as np


class VoxelGenerator(object):
    """Voxel generator in numpy implementation.

    Args:
        voxel_size (list[float]): Size of a single voxel
        point_cloud_range (list[float]): Range of points
        max_num_points (int): Maximum number of points in a single voxel
        max_voxels (int, optional): Maximum number of voxels.
            Defaults to 20000.
    """

    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 max_num_points,
                 max_voxels=20000):

        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = (point_cloud_range[3:] -
                     point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)

        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._max_num_points = max_num_points
        self._max_voxels = max_voxels
        self._grid_size = grid_size

    def generate(self, points):
        """Generate voxels given points."""
        return points_to_voxel(points, self._voxel_size,
                               self._point_cloud_range, self._max_num_points,
                               True, self._max_voxels)

    @property
    def voxel_size(self):
        """list[float]: Size of a single voxel."""
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self):
        """int: Maximum number of points per voxel."""
        return self._max_num_points

    @property
    def point_cloud_range(self):
        """list[float]: Range of point cloud."""
        return self._point_cloud_range

    @property
    def grid_size(self):
        """np.ndarray: The size of grids."""
        return self._grid_size

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        indent = ' ' * (len(repr_str) + 1)
        repr_str += f'(voxel_size={self._voxel_size},\n'
        repr_str += indent + 'point_cloud_range='
        repr_str += f'{self._point_cloud_range.tolist()},\n'
        repr_str += indent + f'max_num_points={self._max_num_points},\n'
        repr_str += indent + f'max_voxels={self._max_voxels},\n'
        repr_str += indent + f'grid_size={self._grid_size.tolist()}'
        repr_str += ')'
        return repr_str


def points_to_voxel(points,
                    voxel_size,
                    coors_range,
                    max_points=35,
                    reverse_index=True,
                    max_voxels=20000):
    """convert kitti points(N, >=3) to voxels.

    Args:
        points (np.ndarray): [N, ndim]. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size (list, tuple, np.ndarray): [3] xyz, indicate voxel size
        coors_range (list[float | tuple[float] | ndarray]): Voxel range.
            format: xyzxyz, minmax
        max_points (int): Indicate maximum points contained in a voxel.
        reverse_index (bool): Whether return reversed coordinates.
            if points has xyz format and reverse_index is True, output
            coordinates will be zyx format, but points in features always
            xyz format.
        max_voxels (int): Maximum number of voxels this function creates.
            For second, 20000 is a good choice. Points should be shuffled for
            randomness before this function because max_voxels drops points.

    Returns:
        tuple[np.ndarray]:
            voxels: [M, max_points, ndim] float tensor. only contain points.
            coordinates: [M, 3] int32 tensor.
            num_points_per_voxel: [M] int32 tensor.
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    if reverse_index:
        voxelmap_shape = voxelmap_shape[::-1]
    # don't create large array in jit(nopython=True) code.
    num_points_per_voxel = np.zeros(shape=(max_voxels, ), dtype=np.int32)
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    voxels = np.zeros(
        shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype)
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
    if reverse_index:
        voxel_num = _points_to_voxel_reverse_kernel(
            points, voxel_size, coors_range, num_points_per_voxel,
            coor_to_voxelidx, voxels, coors, max_points, max_voxels)

    else:
        voxel_num = _points_to_voxel_kernel(points, voxel_size, coors_range,
                                            num_points_per_voxel,
                                            coor_to_voxelidx, voxels, coors,
                                            max_points, max_voxels)

    coors = coors[:voxel_num]
    voxels = voxels[:voxel_num]
    num_points_per_voxel = num_points_per_voxel[:voxel_num]

    return voxels, coors, num_points_per_voxel


@numba.jit(nopython=True)
def _points_to_voxel_reverse_kernel(points,
                                    voxel_size,
                                    coors_range,
                                    num_points_per_voxel,
                                    coor_to_voxelidx,
                                    voxels,
                                    coors,
                                    max_points=35,
                                    max_voxels=20000):
    """convert kitti points(N, >=3) to voxels.

    Args:
        points (np.ndarray): [N, ndim]. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size (list, tuple, np.ndarray): [3] xyz, indicate voxel size
        coors_range (list[float | tuple[float] | ndarray]): Range of voxels.
            format: xyzxyz, minmax
        num_points_per_voxel (int): Number of points per voxel.
        coor_to_voxel_idx (np.ndarray): A voxel grid of shape (D, H, W),
            which has the same shape as the complete voxel map. It indicates
            the index of each corresponding voxel.
        voxels (np.ndarray): Created empty voxels.
        coors (np.ndarray): Created coordinates of each voxel.
        max_points (int): Indicate maximum points contained in a voxel.
        max_voxels (int): Maximum number of voxels this function create.
            for second, 20000 is a good choice. Points should be shuffled for
            randomness before this function because max_voxels drops points.

    Returns:
        tuple[np.ndarray]:
            voxels: Shape [M, max_points, ndim], only contain points.
            coordinates: Shape [M, 3].
            num_points_per_voxel: Shape [M].
    """
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # reduce performance
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # np.round(grid_size)
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                continue
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num


@numba.jit(nopython=True)
def _points_to_voxel_kernel(points,
                            voxel_size,
                            coors_range,
                            num_points_per_voxel,
                            coor_to_voxelidx,
                            voxels,
                            coors,
                            max_points=35,
                            max_voxels=20000):
    """convert kitti points(N, >=3) to voxels.

    Args:
        points (np.ndarray): [N, ndim]. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size (list, tuple, np.ndarray): [3] xyz, indicate voxel size.
        coors_range (list[float | tuple[float] | ndarray]): Range of voxels.
            format: xyzxyz, minmax
        num_points_per_voxel (int): Number of points per voxel.
        coor_to_voxel_idx (np.ndarray): A voxel grid of shape (D, H, W),
            which has the same shape as the complete voxel map. It indicates
            the index of each corresponding voxel.
        voxels (np.ndarray): Created empty voxels.
        coors (np.ndarray): Created coordinates of each voxel.
        max_points (int): Indicate maximum points contained in a voxel.
        max_voxels (int): Maximum number of voxels this function create.
            for second, 20000 is a good choice. Points should be shuffled for
            randomness before this function because max_voxels drops points.

    Returns:
        tuple[np.ndarray]:
            voxels: Shape [M, max_points, ndim], only contain points.
            coordinates: Shape [M, 3].
            num_points_per_voxel: Shape [M].
    """
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)

    # lower_bound = coors_range[:3]
    # upper_bound = coors_range[3:]
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                continue
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num
