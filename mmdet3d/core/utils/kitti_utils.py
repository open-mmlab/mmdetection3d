import numpy as np
import scipy
import torch
from scipy.spatial import Delaunay


def in_hull(p, hull):
    """
    :param p: (N, K) test points
    :param hull: (M, K) M corners of a box
    :return (N) bool
    """
    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        flag = hull.find_simplex(p) >= 0
    except scipy.spatial.qhull.QhullError:
        print('Warning: not a hull %s' % str(hull))
        flag = np.zeros(p.shape[0], dtype=np.bool)

    return flag


def enlarge_box3d(boxes3d, extra_width):
    """
    :param boxes3d: (N, 7) [x, y, z, w, l, h, ry] in LiDAR coords
    """
    if isinstance(boxes3d, np.ndarray):
        large_boxes3d = boxes3d.copy()
    else:
        large_boxes3d = boxes3d.clone()
    large_boxes3d[:, 3:6] += extra_width * 2
    # bugfixed: here should be minus, not addion in LiDAR, 20190508
    large_boxes3d[:, 2] -= extra_width
    return large_boxes3d


def rotate_pc_along_z(pc, rot_angle):
    """
    params pc: (N, 3+C), (N, 3) is in the LiDAR coordinate
    params rot_angle: rad scalar
    Output pc: updated pc with XYZ rotated
    """
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, 0:2] = np.dot(pc[:, 0:2], rotmat)
    return pc


def rotate_pc_along_z_torch(pc, rot_angle):
    """
    :param pc: (N, 512, 3 + C) in the LiDAR coordinate
    :param rot_angle: (N)
    :return:
    TODO: merge with rotate_pc_along_y_torch in bbox_transform.py
    """
    cosa = torch.cos(rot_angle).view(-1, 1)  # (N, 1)
    sina = torch.sin(rot_angle).view(-1, 1)  # (N, 1)

    raw_1 = torch.cat([cosa, -sina], dim=1)  # (N, 2)
    raw_2 = torch.cat([sina, cosa], dim=1)  # (N, 2)
    R = torch.cat((raw_1.unsqueeze(dim=1), raw_2.unsqueeze(dim=1)),
                  dim=1)  # (N, 2, 2)

    pc_temp = pc[:, :, 0:2]  # (N, 512, 2)

    pc[:, :, 0:2] = torch.matmul(pc_temp, R)  # (N, 512, 2)
    return pc
