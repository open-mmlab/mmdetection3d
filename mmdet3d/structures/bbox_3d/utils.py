# Copyright (c) OpenMMLab. All rights reserved.
from logging import warning

import numpy as np
import torch

from mmdet3d.utils.array_converter import array_converter


@array_converter(apply_to=('val', ))
def limit_period(val, offset=0.5, period=np.pi):
    """Limit the value into a period for periodic function.

    Args:
        val (torch.Tensor | np.ndarray): The value to be converted.
        offset (float, optional): Offset to set the value range.
            Defaults to 0.5.
        period ([type], optional): Period of the value. Defaults to np.pi.

    Returns:
        (torch.Tensor | np.ndarray): Value in the range of
            [-offset * period, (1-offset) * period]
    """
    limited_val = val - torch.floor(val / period + offset) * period
    return limited_val


@array_converter(apply_to=('points', 'angles'))
def rotation_3d_in_axis(points,
                        angles,
                        axis=0,
                        return_mat=False,
                        clockwise=False):
    """Rotate points by angles according to axis.

    Args:
        points (np.ndarray | torch.Tensor | list | tuple ):
            Points of shape (N, M, 3).
        angles (np.ndarray | torch.Tensor | list | tuple | float):
            Vector of angles in shape (N,)
        axis (int, optional): The axis to be rotated. Defaults to 0.
        return_mat: Whether or not return the rotation matrix (transposed).
            Defaults to False.
        clockwise: Whether the rotation is clockwise. Defaults to False.

    Raises:
        ValueError: when the axis is not in range [0, 1, 2], it will
            raise value error.

    Returns:
        (torch.Tensor | np.ndarray): Rotated points in shape (N, M, 3).
    """
    batch_free = len(points.shape) == 2
    if batch_free:
        points = points[None]

    if isinstance(angles, float) or len(angles.shape) == 0:
        angles = torch.full(points.shape[:1], angles)

    assert len(points.shape) == 3 and len(angles.shape) == 1 \
        and points.shape[0] == angles.shape[0], f'Incorrect shape of points ' \
        f'angles: {points.shape}, {angles.shape}'

    assert points.shape[-1] in [2, 3], \
        f'Points size should be 2 or 3 instead of {points.shape[-1]}'

    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)

    if points.shape[-1] == 3:
        if axis == 1 or axis == -2:
            rot_mat_T = torch.stack([
                torch.stack([rot_cos, zeros, -rot_sin]),
                torch.stack([zeros, ones, zeros]),
                torch.stack([rot_sin, zeros, rot_cos])
            ])
        elif axis == 2 or axis == -1:
            rot_mat_T = torch.stack([
                torch.stack([rot_cos, rot_sin, zeros]),
                torch.stack([-rot_sin, rot_cos, zeros]),
                torch.stack([zeros, zeros, ones])
            ])
        elif axis == 0 or axis == -3:
            rot_mat_T = torch.stack([
                torch.stack([ones, zeros, zeros]),
                torch.stack([zeros, rot_cos, rot_sin]),
                torch.stack([zeros, -rot_sin, rot_cos])
            ])
        else:
            raise ValueError(f'axis should in range '
                             f'[-3, -2, -1, 0, 1, 2], got {axis}')
    else:
        rot_mat_T = torch.stack([
            torch.stack([rot_cos, rot_sin]),
            torch.stack([-rot_sin, rot_cos])
        ])

    if clockwise:
        rot_mat_T = rot_mat_T.transpose(0, 1)

    if points.shape[0] == 0:
        points_new = points
    else:
        points_new = torch.einsum('aij,jka->aik', points, rot_mat_T)

    if batch_free:
        points_new = points_new.squeeze(0)

    if return_mat:
        rot_mat_T = torch.einsum('jka->ajk', rot_mat_T)
        if batch_free:
            rot_mat_T = rot_mat_T.squeeze(0)
        return points_new, rot_mat_T
    else:
        return points_new


@array_converter(apply_to=('boxes_xywhr', ))
def xywhr2xyxyr(boxes_xywhr):
    """Convert a rotated boxes in XYWHR format to XYXYR format.

    Args:
        boxes_xywhr (torch.Tensor | np.ndarray): Rotated boxes in XYWHR format.

    Returns:
        (torch.Tensor | np.ndarray): Converted boxes in XYXYR format.
    """
    boxes = torch.zeros_like(boxes_xywhr)
    half_w = boxes_xywhr[..., 2] / 2
    half_h = boxes_xywhr[..., 3] / 2

    boxes[..., 0] = boxes_xywhr[..., 0] - half_w
    boxes[..., 1] = boxes_xywhr[..., 1] - half_h
    boxes[..., 2] = boxes_xywhr[..., 0] + half_w
    boxes[..., 3] = boxes_xywhr[..., 1] + half_h
    boxes[..., 4] = boxes_xywhr[..., 4]
    return boxes


def get_box_type(box_type):
    """Get the type and mode of box structure.

    Args:
        box_type (str): The type of box structure.
            The valid value are "LiDAR", "Camera", or "Depth".

    Raises:
        ValueError: A ValueError is raised when `box_type`
            does not belong to the three valid types.

    Returns:
        tuple: Box type and box mode.
    """
    from .box_3d_mode import (Box3DMode, CameraInstance3DBoxes,
                              DepthInstance3DBoxes, LiDARInstance3DBoxes)
    box_type_lower = box_type.lower()
    if box_type_lower == 'lidar':
        box_type_3d = LiDARInstance3DBoxes
        box_mode_3d = Box3DMode.LIDAR
    elif box_type_lower == 'camera':
        box_type_3d = CameraInstance3DBoxes
        box_mode_3d = Box3DMode.CAM
    elif box_type_lower == 'depth':
        box_type_3d = DepthInstance3DBoxes
        box_mode_3d = Box3DMode.DEPTH
    else:
        raise ValueError('Only "box_type" of "camera", "lidar", "depth"'
                         f' are supported, got {box_type}')

    return box_type_3d, box_mode_3d


@array_converter(apply_to=('points_3d', 'proj_mat'))
def points_cam2img(points_3d, proj_mat, with_depth=False):
    """Project points in camera coordinates to image coordinates.

    Args:
        points_3d (torch.Tensor | np.ndarray): Points in shape (N, 3)
        proj_mat (torch.Tensor | np.ndarray):
            Transformation matrix between coordinates.
        with_depth (bool, optional): Whether to keep depth in the output.
            Defaults to False.

    Returns:
        (torch.Tensor | np.ndarray): Points in image coordinates,
            with shape [N, 2] if `with_depth=False`, else [N, 3].
    """
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1

    assert len(proj_mat.shape) == 2, 'The dimension of the projection'\
        f' matrix should be 2 instead of {len(proj_mat.shape)}.'
    d1, d2 = proj_mat.shape[:2]
    assert (d1 == 3 and d2 == 3) or (d1 == 3 and d2 == 4) or (
        d1 == 4 and d2 == 4), 'The shape of the projection matrix'\
        f' ({d1}*{d2}) is not supported.'
    if d1 == 3:
        proj_mat_expanded = torch.eye(
            4, device=proj_mat.device, dtype=proj_mat.dtype)
        proj_mat_expanded[:d1, :d2] = proj_mat
        proj_mat = proj_mat_expanded

    # previous implementation use new_zeros, new_one yields better results
    points_4 = torch.cat([points_3d, points_3d.new_ones(points_shape)], dim=-1)

    point_2d = points_4 @ proj_mat.T
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]

    if with_depth:
        point_2d_res = torch.cat([point_2d_res, point_2d[..., 2:3]], dim=-1)

    return point_2d_res


@array_converter(apply_to=('points', 'cam2img'))
def points_img2cam(points, cam2img):
    """Project points in image coordinates to camera coordinates.

    Args:
        points (torch.Tensor): 2.5D points in 2D images, [N, 3],
            3 corresponds with x, y in the image and depth.
        cam2img (torch.Tensor): Camera intrinsic matrix. The shape can be
            [3, 3], [3, 4] or [4, 4].

    Returns:
        torch.Tensor: points in 3D space. [N, 3],
            3 corresponds with x, y, z in 3D space.
    """
    assert cam2img.shape[0] <= 4
    assert cam2img.shape[1] <= 4
    assert points.shape[1] == 3

    xys = points[:, :2]
    depths = points[:, 2].view(-1, 1)
    unnormed_xys = torch.cat([xys * depths, depths], dim=1)

    pad_cam2img = torch.eye(4, dtype=xys.dtype, device=xys.device)
    pad_cam2img[:cam2img.shape[0], :cam2img.shape[1]] = cam2img
    inv_pad_cam2img = torch.inverse(pad_cam2img).transpose(0, 1)

    # Do operation in homogeneous coordinates.
    num_points = unnormed_xys.shape[0]
    homo_xys = torch.cat([unnormed_xys, xys.new_ones((num_points, 1))], dim=1)
    points3D = torch.mm(homo_xys, inv_pad_cam2img)[:, :3]

    return points3D


def mono_cam_box2vis(cam_box):
    """This is a post-processing function on the bboxes from Mono-3D task. If
    we want to perform projection visualization, we need to:

        1. rotate the box along x-axis for np.pi / 2 (roll)
        2. change orientation from local yaw to global yaw
        3. convert yaw by (np.pi / 2 - yaw)

    After applying this function, we can project and draw it on 2D images.

    Args:
        cam_box (:obj:`CameraInstance3DBoxes`): 3D bbox in camera coordinate
            system before conversion. Could be gt bbox loaded from dataset
            or network prediction output.

    Returns:
        :obj:`CameraInstance3DBoxes`: Box after conversion.
    """
    warning.warn('DeprecationWarning: The hack of yaw and dimension in the '
                 'monocular 3D detection on nuScenes has been removed. The '
                 'function mono_cam_box2vis will be deprecated.')
    from . import CameraInstance3DBoxes
    assert isinstance(cam_box, CameraInstance3DBoxes), \
        'input bbox should be CameraInstance3DBoxes!'

    loc = cam_box.gravity_center
    dim = cam_box.dims
    yaw = cam_box.yaw
    feats = cam_box.tensor[:, 7:]
    # rotate along x-axis for np.pi / 2
    # see also here: https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/datasets/nuscenes_mono_dataset.py#L557  # noqa
    dim[:, [1, 2]] = dim[:, [2, 1]]
    # change local yaw to global yaw for visualization
    # refer to https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/datasets/nuscenes_mono_dataset.py#L164-L166  # noqa
    yaw += torch.atan2(loc[:, 0], loc[:, 2])
    # convert yaw by (-yaw - np.pi / 2)
    # this is because mono 3D box class such as `NuScenesBox` has different
    # definition of rotation with our `CameraInstance3DBoxes`
    yaw = -yaw - np.pi / 2
    cam_box = torch.cat([loc, dim, yaw[:, None], feats], dim=1)
    cam_box = CameraInstance3DBoxes(
        cam_box, box_dim=cam_box.shape[-1], origin=(0.5, 0.5, 0.5))

    return cam_box


def get_proj_mat_by_coord_type(img_meta, coord_type):
    """Obtain image features using points.

    Args:
        img_meta (dict): Meta info.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
            Can be case-insensitive.

    Returns:
        torch.Tensor: transformation matrix.
    """
    coord_type = coord_type.upper()
    mapping = {'LIDAR': 'lidar2img', 'DEPTH': 'depth2img', 'CAMERA': 'cam2img'}
    assert coord_type in mapping.keys()
    return img_meta[mapping[coord_type]]


def yaw2local(yaw, loc):
    """Transform global yaw to local yaw (alpha in kitti) in camera
    coordinates, ranges from -pi to pi.

    Args:
        yaw (torch.Tensor): A vector with local yaw of each box.
            shape: (N, )
        loc (torch.Tensor): gravity center of each box.
            shape: (N, 3)

    Returns:
        torch.Tensor: local yaw (alpha in kitti).
    """
    local_yaw = yaw - torch.atan2(loc[:, 0], loc[:, 2])
    larger_idx = (local_yaw > np.pi).nonzero(as_tuple=False)
    small_idx = (local_yaw < -np.pi).nonzero(as_tuple=False)
    if len(larger_idx) != 0:
        local_yaw[larger_idx] -= 2 * np.pi
    if len(small_idx) != 0:
        local_yaw[small_idx] += 2 * np.pi

    return local_yaw


def get_lidar2img(cam2img, lidar2cam):
    """Get the projection matrix of lidar2img.

    Args:
        cam2img (torch.Tensor): A 3x3 or 4x4 projection matrix.
        lidar2cam (torch.Tensor): A 3x3 or 4x4 projection matrix.

    Returns:
        torch.Tensor: transformation matrix with shape 4x4.
    """
    if cam2img.shape == (3, 3):
        temp = cam2img.new_zeros(4, 4)
        temp[:3, :3] = cam2img
        cam2img = temp

    if lidar2cam.shape == (3, 3):
        temp = lidar2cam.new_zeros(4, 4)
        temp[:3, :3] = lidar2cam
        lidar2cam = temp
    return torch.matmul(cam2img, lidar2cam)
