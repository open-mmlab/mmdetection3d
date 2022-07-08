import numba
import numpy as np
import copy
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
                break
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
    # need mutex if write in cuda, but numba.cuda don't support mutex.
    # in addition, pytorch don't support cuda in dataloader(tensorflow support this).
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # decrease performance
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)

    lower_bound = coors_range[:3]
    upper_bound = coors_range[3:]
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
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num


def points_to_voxel(points,
                     voxel_size,
                     coors_range,
                     max_points=35,
                     reverse_index=True,
                     max_voxels=20000):
    """convert kitti points(N, >=3) to voxels. This version calculate
    everything in one loop. now it takes only 4.2ms(complete point cloud)
    with jit and 3.2ghz cpu.(don't calculate other features)
    Note: this function in ubuntu seems faster than windows 10.

    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coors_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        max_points: int. indicate maximum points contained in a voxel.
        reverse_index: boolean. indicate whether return reversed coordinates.
            if points has xyz format and reverse_index is True, output
            coordinates will be zyx format, but points in features always
            xyz format.
        max_voxels: int. indicate maximum voxels this function create.
            for second, 20000 is a good choice. you should shuffle points
            before call this function because max_voxels may drop some points.

    Returns:
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
        voxel_num = _points_to_voxel_kernel(
            points, voxel_size, coors_range, num_points_per_voxel,
            coor_to_voxelidx, voxels, coors, max_points, max_voxels)

    coors = coors[:voxel_num]
    voxels = voxels[:voxel_num]
    num_points_per_voxel = num_points_per_voxel[:voxel_num]
    # voxels[:, :, -3:] = voxels[:, :, :3] - \
    #     voxels[:, :, :3].sum(axis=1, keepdims=True)/num_points_per_voxel.reshape(-1, 1, 1)
    return voxels, coors, num_points_per_voxel


@numba.jit(nopython=True)
def bound_points_jit(points, upper_bound, lower_bound):
    # to use nopython=True, np.bool is not supported. so you need
    # convert result to np.bool after this function.
    N = points.shape[0]
    ndim = points.shape[1]
    keep_indices = np.zeros((N, ), dtype=np.int32)
    success = 0
    for i in range(N):
        success = 1
        for j in range(ndim-1):
            if points[i, j] < lower_bound[j] or points[i, j] >= upper_bound[j]:
                success = 0
                break
        keep_indices[i] = success
    return keep_indices

