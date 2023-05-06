# Copyright (c) OpenMMLab. All rights reserved.
from .base_points import BasePoints
from .cam_points import CameraPoints
from .depth_points import DepthPoints
from .lidar_points import LiDARPoints

__all__ = ['BasePoints', 'CameraPoints', 'DepthPoints', 'LiDARPoints']


def get_points_type(points_type: str) -> type:
    """Get the class of points according to coordinate type.

    Args:
        points_type (str): The type of points coordinate. The valid value are
            "CAMERA", "LIDAR" and "DEPTH".

    Returns:
        type: Points type.
    """
    points_type_upper = points_type.upper()
    if points_type_upper == 'CAMERA':
        points_cls = CameraPoints
    elif points_type_upper == 'LIDAR':
        points_cls = LiDARPoints
    elif points_type_upper == 'DEPTH':
        points_cls = DepthPoints
    else:
        raise ValueError('Only "points_type" of "CAMERA", "LIDAR" and "DEPTH" '
                         f'are supported, got {points_type}')

    return points_cls
