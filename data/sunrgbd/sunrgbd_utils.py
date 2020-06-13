# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Provides Python helper function to read My SUNRGBD dataset.

Author: Charles R. Qi
Date: October, 2017

Updated by Charles R. Qi
Date: December, 2018
Note: removed basis loading.
"""

import cv2
import numpy as np
import scipy.io as sio

type2class = {
    'bed': 0,
    'table': 1,
    'sofa': 2,
    'chair': 3,
    'toilet': 4,
    'desk': 5,
    'dresser': 6,
    'night_stand': 7,
    'bookshelf': 8,
    'bathtub': 9
}
class2type = {type2class[t]: t for t in type2class}


def flip_axis_to_camera(pc):
    """Flip axis to camera.

    Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward.

    Args:
        pc(ndarray): points in depth axis.

    Returns:
        ndarray: points in camera  axis.
    """
    pc2 = np.copy(pc)
    pc2[:, [0, 1, 2]] = pc2[:, [0, 2, 1]]  # cam X,Y,Z = depth X,-Z,Y
    pc2[:, 1] *= -1
    return pc2


def flip_axis_to_depth(pc):
    pc2 = np.copy(pc)
    pc2[:, [0, 1, 2]] = pc2[:, [0, 2, 1]]  # depth X,Y,Z = cam X,Z,-Y
    pc2[:, 2] *= -1
    return pc2


class SUNObject3d(object):

    def __init__(self, line):
        data = line.split(' ')
        data[1:] = [float(x) for x in data[1:]]
        self.classname = data[0]
        self.xmin = data[1]
        self.ymin = data[2]
        self.xmax = data[1] + data[3]
        self.ymax = data[2] + data[4]
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])
        self.centroid = np.array([data[5], data[6], data[7]])
        self.unused_dimension = np.array([data[8], data[9], data[10]])
        self.width = data[8]
        self.length = data[9]
        self.height = data[10]
        self.orientation = np.zeros((3, ))
        self.orientation[0] = data[11]
        self.orientation[1] = data[12]
        self.heading_angle = -1 * np.arctan2(self.orientation[1],
                                             self.orientation[0])


class SUNRGBD_Calibration(object):
    """Calibration matrices and utils

    We define five coordinate system in SUN RGBD dataset:

        camera coodinate:
            Z is forward, Y is downward, X is rightward.

        depth coordinate:
            Just change axis order and flip up-down axis from camera coord.

        upright depth coordinate: tilted depth coordinate by Rtilt such that
            Z is gravity direction, Z is up-axis, Y is forward,
            X is right-ward.

        upright camera coordinate:
            Just change axis order and flip up-down axis from upright.
                depth coordinate

        image coordinate:
            ----> x-axis (u)
           |
           v
            y-axis (v)

        depth points are stored in upright depth coordinate.
        labels for 3d box (basis, centroid, size) are in upright
            depth coordinate.
        2d boxes are in image coordinate

        We generate frustum point cloud and 3d box
        in upright camera coordinate.

    Args:
        calib_filepath(str): Path of the calib file.
    """

    def __init__(self, calib_filepath):
        lines = [line.rstrip() for line in open(calib_filepath)]
        Rtilt = np.array([float(x) for x in lines[0].split(' ')])
        self.Rtilt = np.reshape(Rtilt, (3, 3), order='F')
        K = np.array([float(x) for x in lines[1].split(' ')])
        self.K = np.reshape(K, (3, 3), order='F')
        self.f_u = self.K[0, 0]
        self.f_v = self.K[1, 1]
        self.c_u = self.K[0, 2]
        self.c_v = self.K[1, 2]

    def project_upright_depth_to_camera(self, pc):
        """Convert pc coordinate from depth to image.

        Args:
            pc(ndarray): Point cloud in depth coordinate.

        Returns:
            pc(ndarray): Point cloud in camera coordinate.
        """
        # Project upright depth to depth coordinate
        pc2 = np.dot(np.transpose(self.Rtilt), np.transpose(pc[:,
                                                               0:3]))  # (3,n)
        return flip_axis_to_camera(np.transpose(pc2))

    def project_upright_depth_to_image(self, pc):
        """Convert pc coordinate from depth to image.

        Args:
            pc(ndarray): Point cloud in depth coordinate.

        Returns:
            ndarray: [N, 2] uv.
            ndarray: [n,] depth.
        """
        pc2 = self.project_upright_depth_to_camera(pc)
        uv = np.dot(pc2, np.transpose(self.K))  # (n,3)
        uv[:, 0] /= uv[:, 2]
        uv[:, 1] /= uv[:, 2]
        return uv[:, 0:2], pc2[:, 2]

    def project_image_to_camera(self, uv_depth):
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v
        pts_3d_camera = np.zeros((n, 3))
        pts_3d_camera[:, 0] = x
        pts_3d_camera[:, 1] = y
        pts_3d_camera[:, 2] = uv_depth[:, 2]
        return pts_3d_camera


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def read_sunrgbd_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [SUNObject3d(line) for line in lines]
    return objects


def load_image(img_filename):
    return cv2.imread(img_filename)


def load_depth_points(depth_filename):
    depth = np.loadtxt(depth_filename)
    return depth


def load_depth_points_mat(depth_filename):
    depth = sio.loadmat(depth_filename)['instance']
    return depth


def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def extract_pc_in_box3d(pc, box3d):
    """Extract point cloud in box3d.

    Args:
        pc(ndarray): [N, 3] Point cloud.
        box3d(ndarray): [8,3] 3d box.

    Returns:
        ndarray: Selected point cloud.
        ndarray: Indices of selected point cloud.
    """
    box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
    return pc[box3d_roi_inds, :], box3d_roi_inds


def my_compute_box_3d(center, size, heading_angle):
    R = rotz(-1 * heading_angle)
    l, w, h = size
    x_corners = [-l, l, l, -l, -l, l, l, -l]
    y_corners = [w, w, -w, -w, w, w, -w, -w]
    z_corners = [h, h, h, h, -h, -h, -h, -h]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] += center[0]
    corners_3d[1, :] += center[1]
    corners_3d[2, :] += center[2]
    return np.transpose(corners_3d)


def compute_box_3d(obj, calib):
    """Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.

    Args:
        obj(SUNObject3d): Instance of SUNObject3d.
        calib(SUNRGBD_Calibration): Instance of SUNRGBD_Calibration.

    Returns:
        ndarray: [8,2] array in image coord.
        corners_3d: [8,3] array in in upright depth coord.
    """
    center = obj.centroid

    # compute rotational matrix around yaw axis
    R = rotz(-1 * obj.heading_angle)

    # 3d bounding box dimensions
    length = obj.length  # along heading arrow
    width = obj.width  # perpendicular to heading arrow
    height = obj.height

    # rotate and translate 3d bounding box
    x_corners = [
        -length, length, length, -length, -length, length, length, -length
    ]
    y_corners = [width, width, -width, -width, width, width, -width, -width]
    z_corners = [
        height, height, height, height, -height, -height, -height, -height
    ]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] += center[0]
    corners_3d[1, :] += center[1]
    corners_3d[2, :] += center[2]

    # project the 3d bounding box into the image plane
    corners_2d, _ = calib.project_upright_depth_to_image(
        np.transpose(corners_3d))
    return corners_2d, np.transpose(corners_3d)
