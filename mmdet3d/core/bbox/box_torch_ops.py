import numpy as np
import torch


def limit_period(val, offset=0.5, period=np.pi):
    return val - torch.floor(val / period + offset) * period


def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point.

    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.

    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    corners_norm = torch.from_numpy(
        np.stack(np.unravel_index(np.arange(2**ndim), [2] * ndim), axis=1)).to(
            device=dims.device, dtype=dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - dims.new_tensor(origin)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape(
        [1, 2**ndim, ndim])
    return corners


def rotation_3d_in_axis(points, angles, axis=0):
    # points: [N, point_size, 3]
    # angles: [N]
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = torch.stack([
            torch.stack([rot_cos, zeros, -rot_sin]),
            torch.stack([zeros, ones, zeros]),
            torch.stack([rot_sin, zeros, rot_cos])
        ])
    elif axis == 2 or axis == -1:
        rot_mat_T = torch.stack([
            torch.stack([rot_cos, -rot_sin, zeros]),
            torch.stack([rot_sin, rot_cos, zeros]),
            torch.stack([zeros, zeros, ones])
        ])
    elif axis == 0:
        rot_mat_T = torch.stack([
            torch.stack([zeros, rot_cos, -rot_sin]),
            torch.stack([zeros, rot_sin, rot_cos]),
            torch.stack([ones, zeros, zeros])
        ])
    else:
        raise ValueError('axis should in range')

    return torch.einsum('aij,jka->aik', (points, rot_mat_T))


def center_to_corner_box3d(centers,
                           dims,
                           angles,
                           origin=[0.5, 1.0, 0.5],
                           axis=1):
    """convert kitti locations, dimensions and angles to corners

    Args:
        centers (float array, shape=[N, 3]): locations in kitti label file.
        dims (float array, shape=[N, 3]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
        origin (list or array or float): origin point relate to smallest point.
            use [0.5, 1.0, 0.5] in camera and [0.5, 0.5, 0] in lidar.
        axis (int): rotation axis. 1 for camera and 2 for lidar.
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.view(-1, 1, 3)
    return corners


def lidar_to_camera(points, r_rect, velo2cam):
    num_points = points.shape[0]
    points = torch.cat(
        [points, torch.ones(num_points, 1).type_as(points)], dim=-1)
    camera_points = points @ (r_rect @ velo2cam).t()
    return camera_points[..., :3]


def box_lidar_to_camera(data, r_rect, velo2cam):
    xyz_lidar = data[..., 0:3]
    w, l, h = data[..., 3:4], data[..., 4:5], data[..., 5:6]
    r = data[..., 6:7]
    xyz = lidar_to_camera(xyz_lidar, r_rect, velo2cam)
    return torch.cat([xyz, l, h, w, r], dim=-1)


def project_to_image(points_3d, proj_mat):
    points_num = list(points_3d.shape)[:-1]
    points_shape = np.concatenate([points_num, [1]], axis=0).tolist()
    # previous implementation use new_zeros, new_one yeilds better results
    points_4 = torch.cat(
        [points_3d, points_3d.new_ones(*points_shape)], dim=-1)
    # point_2d = points_4 @ tf.transpose(proj_mat, [1, 0])
    point_2d = torch.matmul(points_4, proj_mat.t())
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
    return point_2d_res


def rbbox2d_to_near_bbox(rbboxes):
    """convert rotated bbox to nearest 'standing' or 'lying' bbox.

    Args:
        rbboxes: [N, 5(x, y, xdim, ydim, rad)] rotated bboxes
    Returns:
        bboxes: [N, 4(xmin, ymin, xmax, ymax)] bboxes
    """
    rots = rbboxes[..., -1]
    rots_0_pi_div_2 = torch.abs(limit_period(rots, 0.5, np.pi))
    cond = (rots_0_pi_div_2 > np.pi / 4)[..., None]
    bboxes_center = torch.where(cond, rbboxes[:, [0, 1, 3, 2]], rbboxes[:, :4])
    bboxes = center_to_minmax_2d(bboxes_center[:, :2], bboxes_center[:, 2:])
    return bboxes


def center_to_minmax_2d_0_5(centers, dims):
    return torch.cat([centers - dims / 2, centers + dims / 2], dim=-1)


def center_to_minmax_2d(centers, dims, origin=0.5):
    if origin == 0.5:
        return center_to_minmax_2d_0_5(centers, dims)
    corners = center_to_corner_box2d(centers, dims, origin=origin)
    return corners[:, [0, 2]].reshape([-1, 4])


def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """convert kitti locations, dimensions and angles to corners.
    format: center(xy), dims(xy), angles(clockwise when positive)

    Args:
        centers (float array, shape=[N, 2]): locations in kitti label file.
        dims (float array, shape=[N, 2]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.

    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.reshape([-1, 1, 2])
    return corners


def rotation_2d(points, angles):
    """rotation 2d points based on origin point clockwise when angle positive.

    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle.

    Returns:
        float array: same shape as points
    """
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    rot_mat_T = torch.stack([[rot_cos, -rot_sin], [rot_sin, rot_cos]])
    return torch.einsum('aij,jka->aik', points, rot_mat_T)


def enlarge_box3d_lidar(boxes3d, extra_width):
    """Enlarge the length, width and height of input boxes

    Args:
        boxes3d (torch.float32 or numpy.float32): bottom_center with
            shape [N, 7], (x, y, z, w, l, h, ry) in LiDAR coords
        extra_width (float): a fix number to add

    Returns:
        torch.float32 or numpy.float32: enlarged boxes
    """
    if isinstance(boxes3d, np.ndarray):
        large_boxes3d = boxes3d.copy()
    else:
        large_boxes3d = boxes3d.clone()
    large_boxes3d[:, 3:6] += extra_width * 2
    large_boxes3d[:, 2] -= extra_width  # bottom center z minus extra_width
    return large_boxes3d


def boxes3d_to_corners3d_lidar_torch(boxes3d, bottom_center=True):
    """convert kitti center boxes to corners

        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1

    Args:
        boxes3d (FloatTensor): (N, 7) [x, y, z, w, l, h, ry] in LiDAR coords,
            see the definition of ry in KITTI dataset
        bottom_center (bool): whether z is on the bottom center of object.

    Returns:
        FloatTensor: box corners with shape (N, 8, 3)
    """
    boxes_num = boxes3d.shape[0]
    w, l, h = boxes3d[:, 3:4], boxes3d[:, 4:5], boxes3d[:, 5:6]
    ry = boxes3d[:, 6:7]

    zeros = boxes3d.new_zeros(boxes_num, 1)
    ones = boxes3d.new_ones(boxes_num, 1)
    x_corners = torch.cat(
        [w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.],
        dim=1)  # (N, 8)
    y_corners = torch.cat(
        [-l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2.],
        dim=1)  # (N, 8)
    if bottom_center:
        z_corners = torch.cat([zeros, zeros, zeros, zeros, h, h, h, h],
                              dim=1)  # (N, 8)
    else:
        z_corners = torch.cat([
            -h / 2., -h / 2., -h / 2., -h / 2., h / 2., h / 2., h / 2., h / 2.
        ],
                              dim=1)  # (N, 8)
    temp_corners = torch.cat(
        (x_corners.unsqueeze(dim=2), y_corners.unsqueeze(dim=2),
         z_corners.unsqueeze(dim=2)),
        dim=2)  # (N, 8, 3)

    cosa, sina = torch.cos(ry), torch.sin(ry)
    raw_1 = torch.cat([cosa, -sina, zeros], dim=1)  # (N, 3)
    raw_2 = torch.cat([sina, cosa, zeros], dim=1)  # (N, 3)
    raw_3 = torch.cat([zeros, zeros, ones], dim=1)  # (N, 3)
    R = torch.cat((raw_1.unsqueeze(dim=1), raw_2.unsqueeze(dim=1),
                   raw_3.unsqueeze(dim=1)),
                  dim=1)  # (N, 3, 3)

    rotated_corners = torch.matmul(temp_corners, R)  # (N, 8, 3)
    x_corners = rotated_corners[:, :, 0]
    y_corners = rotated_corners[:, :, 1]
    z_corners = rotated_corners[:, :, 2]
    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.view(-1, 1) + x_corners.view(-1, 8)
    y = y_loc.view(-1, 1) + y_corners.view(-1, 8)
    z = z_loc.view(-1, 1) + z_corners.view(-1, 8)
    corners = torch.cat((x.view(-1, 8, 1), y.view(-1, 8, 1), z.view(-1, 8, 1)),
                        dim=2)

    return corners
