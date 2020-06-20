import numpy as np
import torch

from .base_box3d import BaseInstance3DBoxes
from .utils import limit_period, rotation_3d_in_axis


class CameraInstance3DBoxes(BaseInstance3DBoxes):
    """3D boxes of instances in CAM coordinates

    Coordinates in camera:
    .. code-block:: none

                z front (yaw=0.5*pi)
               /
              /
             0 ------> x right (yaw=0)
             |
             |
             v
        down y

    The relative coordinate of bottom center in a CAM box is (0.5, 1.0, 0.5),
    and the yaw is around the y axis, thus the rotation axis=1.
    The yaw is 0 at the positive direction of x axis, and increases from
    the positive direction of x to the positive direction of z.

    Attributes:
        tensor (torch.Tensor): float matrix of N x box_dim.
        box_dim (int): integer indicates the dimension of a box
            Each row is (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): if True, the value of yaw will be set to 0 as minmax
            boxes.
    """

    @property
    def height(self):
        """Obtain the height of all the boxes.

        Returns:
            torch.Tensor: a vector with height of each box.
        """
        return self.tensor[:, 4]

    @property
    def top_height(self):
        """Obtain the top height of all the boxes.

        Returns:
            torch.Tensor: a vector with the top height of each box.
        """
        # the positive direction is down rather than up
        return self.bottom_height - self.height

    @property
    def bottom_height(self):
        """Obtain the bottom's height of all the boxes.

        Returns:
            torch.Tensor: a vector with bottom's height of each box.
        """
        return self.tensor[:, 1]

    @property
    def gravity_center(self):
        """Calculate the gravity center of all the boxes.

        Returns:
            torch.Tensor: a tensor with center of each box.
        """
        bottom_center = self.bottom_center
        gravity_center = torch.zeros_like(bottom_center)
        gravity_center[:, [0, 2]] = bottom_center[:, [0, 2]]
        gravity_center[:, 1] = bottom_center[:, 1] - self.tensor[:, 4] * 0.5
        return gravity_center

    @property
    def corners(self):
        """Calculate the coordinates of corners of all the boxes.

        Convert the boxes to  in clockwise order, in the form of
        (x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)

        .. code-block:: none

                         front z
                              /
                             /
               (x0, y0, z1) + -----------  + (x1, y0, z1)
                           /|            / |
                          / |           /  |
            (x0, y0, z0) + ----------- +   + (x1, y1, z1)
                         |  /      .   |  /
                         | / oriign    | /
            (x0, y1, z0) + ----------- + -------> x right
                         |             (x1, y1, z0)
                         |
                         v
                    down y

        Returns:
            torch.Tensor: corners of each box with size (N, 8, 3)
        """
        # TODO: rotation_3d_in_axis function do not support
        #  empty tensor currently.
        assert len(self.tensor) != 0
        dims = self.dims
        corners_norm = torch.from_numpy(
            np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)).to(
                device=dims.device, dtype=dims.dtype)

        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        # use relative origin [0.5, 1, 0.5]
        corners_norm = corners_norm - dims.new_tensor([0.5, 1, 0.5])
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

        # rotate around y axis
        corners = rotation_3d_in_axis(corners, self.tensor[:, 6], axis=1)
        corners += self.tensor[:, :3].view(-1, 1, 3)
        return corners

    @property
    def bev(self):
        """Calculate the 2D bounding boxes in BEV with rotation

        Returns:
            torch.Tensor: a nx5 tensor of 2D BEV box of each box.
                The box is in XYWHR format.
        """
        return self.tensor[:, [0, 2, 3, 5, 6]]

    @property
    def nearest_bev(self):
        """Calculate the 2D bounding boxes in BEV without rotation

        Returns:
            torch.Tensor: a tensor of 2D BEV box of each box.
        """
        # Obtain BEV boxes with rotation in XZWHR format
        bev_rotated_boxes = self.bev
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

    def rotate(self, angle, points=None):
        """Rotate boxes with points (optional) with the given angle.

        Args:
            angle (float, torch.Tensor): Rotation angle.
            points (torch.Tensor, numpy.ndarray, optional): Points to rotate.
                Defaults to None.

        Returns:
            tuple or None: When ``points`` is None, the function returns None,
                otherwise it returns the rotated points and the
                rotation matrix ``rot_mat_T``.
        """
        if not isinstance(angle, torch.Tensor):
            angle = self.tensor.new_tensor(angle)
        rot_sin = torch.sin(angle)
        rot_cos = torch.cos(angle)
        rot_mat_T = self.tensor.new_tensor([[rot_cos, 0, -rot_sin], [0, 1, 0],
                                            [rot_sin, 0, rot_cos]])

        self.tensor[:, :3] = self.tensor[:, :3] @ rot_mat_T
        self.tensor[:, 6] += angle

        if points is not None:
            if isinstance(points, torch.Tensor):
                points[:, :3] = points[:, :3] @ rot_mat_T
            elif isinstance(points, np.ndarray):
                rot_mat_T = rot_mat_T.numpy()
                points[:, :3] = np.dot(points[:, :3], rot_mat_T)
            else:
                raise ValueError
            return points, rot_mat_T

    def flip(self, bev_direction='horizontal', points=None):
        """Flip the boxes in BEV along given BEV direction

        In CAM coordinates, it flips the x (horizontal) or z (vertical) axis.

        Args:
            bev_direction (str): Flip direction (horizontal or vertical).
            points (torch.Tensor, numpy.ndarray, None): Points to flip.
                Defaults to None.

        Returns:
            torch.Tensor, numpy.ndarray or None: Flipped points.
        """
        assert bev_direction in ('horizontal', 'vertical')
        if bev_direction == 'horizontal':
            self.tensor[:, 0::7] = -self.tensor[:, 0::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6] + np.pi
        elif bev_direction == 'vertical':
            self.tensor[:, 2::7] = -self.tensor[:, 2::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6]

        if points is not None:
            assert isinstance(points, (torch.Tensor, np.ndarray))
            if bev_direction == 'horizontal':
                points[:, 0] = -points[:, 0]
            elif bev_direction == 'vertical':
                points[:, 2] = -points[:, 2]
            return points

    def in_range_bev(self, box_range):
        """Check whether the boxes are in the given range

        Args:
            box_range (list | torch.Tensor): the range of box
                (x_min, z_min, x_max, z_max)

        Note:
            In the original implementation of SECOND, checking whether
            a box in the range checks whether the points are in a convex
            polygon, we try to reduce the burdun for simpler cases.
            TODO: check whether this will effect the performance

        Returns:
            torch.Tensor: Indicating whether each box is inside
                the reference range.
        """
        in_range_flags = ((self.tensor[:, 0] > box_range[0])
                          & (self.tensor[:, 2] > box_range[1])
                          & (self.tensor[:, 0] < box_range[2])
                          & (self.tensor[:, 2] < box_range[3]))
        return in_range_flags

    @classmethod
    def height_overlaps(cls, boxes1, boxes2, mode='iou'):
        """Calculate height overlaps of two boxes

        Note:
            This function calculate the height overlaps between boxes1 and
            boxes2,  boxes1 and boxes2 should be in the same type.

        Args:
            boxes1 (:obj:BaseInstanceBoxes): boxes 1 contain N boxes
            boxes2 (:obj:BaseInstanceBoxes): boxes 2 contain M boxes
            mode (str, optional): mode of iou calculation. Defaults to 'iou'.

        Returns:
            torch.Tensor: Calculated iou of boxes
        """
        assert isinstance(boxes1, BaseInstance3DBoxes)
        assert isinstance(boxes2, BaseInstance3DBoxes)
        assert type(boxes1) == type(boxes2), '"boxes1" and "boxes2" should' \
            f'be in the same type, got {type(boxes1)} and {type(boxes2)}.'

        boxes1_top_height = boxes1.top_height.view(-1, 1)
        boxes1_bottom_height = boxes1.bottom_height.view(-1, 1)
        boxes2_top_height = boxes2.top_height.view(1, -1)
        boxes2_bottom_height = boxes2.bottom_height.view(1, -1)

        # In camera coordinate system
        # from up to down is the positive direction
        heighest_of_bottom = torch.min(boxes1_bottom_height,
                                       boxes2_bottom_height)
        lowest_of_top = torch.max(boxes1_top_height, boxes2_top_height)
        overlaps_h = torch.clamp(heighest_of_bottom - lowest_of_top, min=0)
        return overlaps_h

    def convert_to(self, dst, rt_mat=None):
        """Convert self to `dst` mode.

        Args:
            dst (BoxMode): the target Box mode
            rt_mat (np.ndarray | torch.Tensor): The rotation and translation
                matrix between different coordinates. Defaults to None.
                The conversion from `src` coordinates to `dst` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.

        Returns:
            BaseInstance3DBoxes:
                The converted box of the same type in the `dst` mode.
        """
        from .box_3d_mode import Box3DMode
        return Box3DMode.convert(
            box=self, src=Box3DMode.CAM, dst=dst, rt_mat=rt_mat)
