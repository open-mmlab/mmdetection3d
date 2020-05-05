import numpy as np
import torch

from .base_box3d import BaseInstance3DBoxes
from .utils import limit_period, rotation_3d_in_axis


class LiDARInstance3DBoxes(BaseInstance3DBoxes):
    """
    This structure stores a list of boxes as a Nx7 torch.Tensor.
    It supports some common methods about boxes
    (`area`, `clip`, `nonempty`, etc),
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all boxes)
    By default the (x, y, z) is the bottom center of a box

    Attributes:
        tensor (torch.Tensor): float matrix of N x box_dim.
        box_dim (int): integer indicates the dimension of a box
        Each row is (x, y, z, x_size, y_size, z_size, yaw, ...).
    """

    def gravity_center(self):
        """Calculate the gravity center of all the boxes.

        Returns:
            torch.Tensor: a tensor with center of each box.
        """
        bottom_center = self.bottom_center()
        gravity_center = torch.zeros_like(bottom_center)
        gravity_center[:, :2] = bottom_center[:, :2]
        gravity_center[:, 2] = bottom_center[:, 2] + bottom_center[:, 5] * 0.5
        return gravity_center

    def corners(self, origin=[0.5, 1.0, 0.5], axis=1):
        """Calculate the coordinates of corners of all the boxes.

        Convert the boxes to the form of
        (x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1)

        Args:
            origin (list[float]): origin point relate to smallest point.
                use [0.5, 1.0, 0.5] in camera and [0.5, 0.5, 0] in lidar.
            axis (int): rotation axis. 1 for camera and 2 for lidar.

        Returns:
            torch.Tensor: corners of each box with size (N, 8, 3)
        """
        dims = self.tensor[:, 3:6]
        corners_norm = torch.from_numpy(
            np.stack(np.unravel_index(np.arange(2**3), [2] * 3), axis=1)).to(
                device=dims.device, dtype=dims.dtype)

        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        corners_norm = corners_norm - dims.new_tensor(origin)
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 2**3, 3])

        corners = rotation_3d_in_axis(corners, self.tensor[:, 6], axis=axis)
        corners += self.tensor[:, :3].view(-1, 1, 3)
        return corners

    def nearset_bev(self):
        """Calculate the 2D bounding boxes in BEV without rotation

        Returns:
            torch.Tensor: a tensor of 2D BEV box of each box.
        """
        # Obtain BEV boxes with rotation in XYWHR format
        bev_rotated_boxes = self.tensor[:, [0, 1, 3, 4, 6]]
        # convert the rotation to a valid range
        rotations = bev_rotated_boxes[:, -1]
        normed_rotations = torch.abs(limit_period(rotations, 0.5, np.pi))

        # find the center of boxes
        conditions = (normed_rotations > np.pi / 4)[..., None]
        bboxes_xywh = torch.where(conditions, bev_rotated_boxes[:,
                                                                [0, 1, 3, 2]],
                                  bev_rotated_boxes[:, :4])

        centers = bboxes_xywh[:, :2]
        dims = bboxes_xywh[:, 2:]
        bev_boxes = torch.cat([centers - dims / 2, centers + dims / 2], dim=-1)
        return bev_boxes

    def rotate(self, angle):
        """Calculate whether the points is in any of the boxes

        Args:
            angles (float | torch.Tensor): rotation angle

        Returns:
            None if `return_rot_mat=False`,
            torch.Tensor if `return_rot_mat=True`
        """
        if not isinstance(angle, torch.Tensor):
            angle = self.tensor.new_tensor(angle)
        rot_sin = torch.sin(angle)
        rot_cos = torch.cos(angle)
        rot_mat_T = self.tensor.new_tensor([[rot_cos, -rot_sin, 0],
                                            [rot_sin, rot_cos, 0], [0, 0, 1]])

        self.tensor[:, :3] = self.tensor[:, :3] @ rot_mat_T
        self.tensor[:, 6] += angle

    def flip(self):
        self.tensor[:, 1::7] = -self.tensor[:, 1::7]
        self.tensor[:, 6] = -self.tensor[:, 6] + np.pi

    def translate(self, trans_vector):
        """Calculate whether the points is in any of the boxes

        Args:
            trans_vector (torch.Tensor): translation vector of size 1x3

        """
        if not isinstance(trans_vector, torch.Tensor):
            trans_vector = self.tensor.new_tensor(trans_vector)
        self.tensor[:, :3] += trans_vector

    def in_range_3d(self, box_range):
        """Check whether the boxes are in the given range

        Args:
            box_range (list | torch.Tensor): the range of box
                (x_min, y_min, z_min, x_max, y_max, z_max)

        Note:
            In the original implementation of SECOND, checking whether
            a box in the range checks whether the points are in a convex
            polygon, we try to reduce the burdun for simpler cases.
            TODO: check whether this will effect the performance

        Returns:
            a binary vector, indicating whether each box is inside
            the reference range.
        """
        in_range_flags = ((self.tensor[:, 0] > box_range[0])
                          & (self.tensor[:, 1] > box_range[1])
                          & (self.tensor[:, 2] > box_range[2])
                          & (self.tensor[:, 0] < box_range[3])
                          & (self.tensor[:, 1] < box_range[4])
                          & (self.tensor[:, 2] < box_range[5]))
        return in_range_flags

    def in_range_bev(self, box_range):
        """Check whether the boxes are in the given range

        Args:
            box_range (list | torch.Tensor): the range of box
                (x_min, y_min, x_max, y_max)

        Note:
            In the original implementation of SECOND, checking whether
            a box in the range checks whether the points are in a convex
            polygon, we try to reduce the burdun for simpler cases.
            TODO: check whether this will effect the performance

        Returns:
            a binary vector, indicating whether each box is inside
            the reference range.
        """
        in_range_flags = ((self.tensor[:, 0] > box_range[0])
                          & (self.tensor[:, 1] > box_range[1])
                          & (self.tensor[:, 0] < box_range[2])
                          & (self.tensor[:, 1] < box_range[3]))
        return in_range_flags

    def scale(self, scale_factor):
        """Scale the box with horizontal and vertical scaling factors

        Args:
            scale_factors (float):
                scale factors to scale the boxes.
        """
        self.tensor[:, :6] *= scale_factor
        self.tensor[:, 7:] *= scale_factor

    def limit_yaw(self, offset=0.5, period=np.pi):
        """Limit the yaw to a given period and offset

        Args:
            offset (float): the offset of the yaw
            period (float): the expected period
        """
        self.tensor[:, 6] = limit_period(self.tensor[:, 6], offset, period)
