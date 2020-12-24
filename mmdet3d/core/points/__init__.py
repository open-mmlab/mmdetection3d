from .base_points import BasePoints
from .cam_points import CameraPoints
from .depth_points import DepthPoints
from .lidar_points import LiDARPoints

__all__ = ['BasePoints', 'CameraPoints', 'DepthPoints', 'LiDARPoints']


def get_points_type(points_type):
    """Get the class of points according to coordinate type.

    Args:
        points_type (str): The type of points coordinate.
            The valid value are "CAMERA", "LIDAR", or "DEPTH".

    Returns:
        class: Points type.
    """
    if points_type == 'CAMERA':
        points_cls = CameraPoints
    elif points_type == 'LIDAR':
        points_cls = LiDARPoints
    elif points_type == 'DEPTH':
        points_cls = DepthPoints
    else:
        raise ValueError('Only "points_type" of "CAMERA", "LIDAR", or "DEPTH"'
                         f' are supported, got {points_type}')

    return points_cls
