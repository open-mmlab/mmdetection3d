# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from ...points import BasePoints
from .base_box3d import BaseInstance3DBoxes
from .utils import rotation_3d_in_axis, yaw2local


class CameraInstance3DBoxes(BaseInstance3DBoxes):
    """3D boxes of instances in CAM coordinates.

    Coordinates in camera:

    .. code-block:: none

                z front (yaw=-0.5*pi)
               /
              /
             0 ------> x right (yaw=0)
             |
             |
             v
        down y

    The relative coordinate of bottom center in a CAM box is (0.5, 1.0, 0.5),
    and the yaw is around the y axis, thus the rotation axis=1.
    The yaw is 0 at the positive direction of x axis, and decreases from
    the positive direction of x to the positive direction of z.

    Attributes:
        tensor (torch.Tensor): Float matrix in shape (N, box_dim).
        box_dim (int): Integer indicating the dimension of a box
            Each row is (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as
            axis-aligned boxes tightly enclosing the original boxes.
    """
    YAW_AXIS = 1

    def __init__(self,
                 tensor,
                 box_dim=7,
                 with_yaw=True,
                 origin=(0.5, 1.0, 0.5)):
        if isinstance(tensor, torch.Tensor):
            device = tensor.device
        else:
            device = torch.device('cpu')
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that
            # does not depend on the inputs (and consequently confuses jit)
            tensor = tensor.reshape((0, box_dim)).to(
                dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == box_dim, tensor.size()

        if tensor.shape[-1] == 6:
            # If the dimension of boxes is 6, we expand box_dim by padding
            # 0 as a fake yaw and set with_yaw to False.
            assert box_dim == 6
            fake_rot = tensor.new_zeros(tensor.shape[0], 1)
            tensor = torch.cat((tensor, fake_rot), dim=-1)
            self.box_dim = box_dim + 1
            self.with_yaw = False
        else:
            self.box_dim = box_dim
            self.with_yaw = with_yaw
        self.tensor = tensor.clone()

        if origin != (0.5, 1.0, 0.5):
            dst = self.tensor.new_tensor((0.5, 1.0, 0.5))
            src = self.tensor.new_tensor(origin)
            self.tensor[:, :3] += self.tensor[:, 3:6] * (dst - src)

    @property
    def height(self):
        """torch.Tensor: A vector with height of each box in shape (N, )."""
        return self.tensor[:, 4]

    @property
    def top_height(self):
        """torch.Tensor:
            A vector with the top height of each box in shape (N, )."""
        # the positive direction is down rather than up
        return self.bottom_height - self.height

    @property
    def bottom_height(self):
        """torch.Tensor:
            A vector with bottom's height of each box in shape (N, )."""
        return self.tensor[:, 1]

    @property
    def local_yaw(self):
        """torch.Tensor:
            A vector with local yaw of each box in shape (N, ).
            local_yaw equals to alpha in kitti, which is commonly
            used in monocular 3D object detection task, so only
            :obj:`CameraInstance3DBoxes` has the property.
        """
        yaw = self.yaw
        loc = self.gravity_center
        local_yaw = yaw2local(yaw, loc)

        return local_yaw

    @property
    def gravity_center(self):
        """torch.Tensor: A tensor with center of each box in shape (N, 3)."""
        bottom_center = self.bottom_center
        gravity_center = torch.zeros_like(bottom_center)
        gravity_center[:, [0, 2]] = bottom_center[:, [0, 2]]
        gravity_center[:, 1] = bottom_center[:, 1] - self.tensor[:, 4] * 0.5
        return gravity_center

    @property
    def corners(self):
        """torch.Tensor: Coordinates of corners of all the boxes in
                         shape (N, 8, 3).

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
                         | / origin    | /
            (x0, y1, z0) + ----------- + -------> x right
                         |             (x1, y1, z0)
                         |
                         v
                    down y
        """
        if self.tensor.numel() == 0:
            return torch.empty([0, 8, 3], device=self.tensor.device)

        dims = self.dims
        corners_norm = torch.from_numpy(
            np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)).to(
                device=dims.device, dtype=dims.dtype)

        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        # use relative origin [0.5, 1, 0.5]
        corners_norm = corners_norm - dims.new_tensor([0.5, 1, 0.5])
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

        corners = rotation_3d_in_axis(
            corners, self.tensor[:, 6], axis=self.YAW_AXIS)
        corners += self.tensor[:, :3].view(-1, 1, 3)
        return corners

    @property
    def bev(self):
        """torch.Tensor: 2D BEV box of each box with rotation
            in XYWHR format, in shape (N, 5)."""
        bev = self.tensor[:, [0, 2, 3, 5, 6]].clone()
        # positive direction of the gravity axis
        # in cam coord system points to the earth
        # so the bev yaw angle needs to be reversed
        bev[:, -1] = -bev[:, -1]
        return bev

    def rotate(self, angle, points=None):
        """Rotate boxes with points (optional) with the given angle or rotation
        matrix.

        Args:
            angle (float | torch.Tensor | np.ndarray):
                Rotation angle or rotation matrix.
            points (torch.Tensor | np.ndarray | :obj:`BasePoints`, optional):
                Points to rotate. Defaults to None.

        Returns:
            tuple or None: When ``points`` is None, the function returns
                None, otherwise it returns the rotated points and the
                rotation matrix ``rot_mat_T``.
        """
        if not isinstance(angle, torch.Tensor):
            angle = self.tensor.new_tensor(angle)

        assert angle.shape == torch.Size([3, 3]) or angle.numel() == 1, \
            f'invalid rotation angle shape {angle.shape}'

        if angle.numel() == 1:
            self.tensor[:, 0:3], rot_mat_T = rotation_3d_in_axis(
                self.tensor[:, 0:3],
                angle,
                axis=self.YAW_AXIS,
                return_mat=True)
        else:
            rot_mat_T = angle
            rot_sin = rot_mat_T[2, 0]
            rot_cos = rot_mat_T[0, 0]
            angle = np.arctan2(rot_sin, rot_cos)
            self.tensor[:, 0:3] = self.tensor[:, 0:3] @ rot_mat_T

        self.tensor[:, 6] += angle

        if points is not None:
            if isinstance(points, torch.Tensor):
                points[:, :3] = points[:, :3] @ rot_mat_T
            elif isinstance(points, np.ndarray):
                rot_mat_T = rot_mat_T.cpu().numpy()
                points[:, :3] = np.dot(points[:, :3], rot_mat_T)
            elif isinstance(points, BasePoints):
                points.rotate(rot_mat_T)
            else:
                raise ValueError
            return points, rot_mat_T

    def flip(self, bev_direction='horizontal', points=None):
        """Flip the boxes in BEV along given BEV direction.

        In CAM coordinates, it flips the x (horizontal) or z (vertical) axis.

        Args:
            bev_direction (str): Flip direction (horizontal or vertical).
            points (torch.Tensor | np.ndarray | :obj:`BasePoints`, optional):
                Points to flip. Defaults to None.

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
            assert isinstance(points, (torch.Tensor, np.ndarray, BasePoints))
            if isinstance(points, (torch.Tensor, np.ndarray)):
                if bev_direction == 'horizontal':
                    points[:, 0] = -points[:, 0]
                elif bev_direction == 'vertical':
                    points[:, 2] = -points[:, 2]
            elif isinstance(points, BasePoints):
                points.flip(bev_direction)
            return points

    @classmethod
    def height_overlaps(cls, boxes1, boxes2, mode='iou'):
        """Calculate height overlaps of two boxes.

        This function calculates the height overlaps between ``boxes1`` and
        ``boxes2``, where ``boxes1`` and ``boxes2`` should be in the same type.

        Args:
            boxes1 (:obj:`CameraInstance3DBoxes`): Boxes 1 contain N boxes.
            boxes2 (:obj:`CameraInstance3DBoxes`): Boxes 2 contain M boxes.
            mode (str, optional): Mode of iou calculation. Defaults to 'iou'.

        Returns:
            torch.Tensor: Calculated iou of boxes' heights.
        """
        assert isinstance(boxes1, CameraInstance3DBoxes)
        assert isinstance(boxes2, CameraInstance3DBoxes)

        boxes1_top_height = boxes1.top_height.view(-1, 1)
        boxes1_bottom_height = boxes1.bottom_height.view(-1, 1)
        boxes2_top_height = boxes2.top_height.view(1, -1)
        boxes2_bottom_height = boxes2.bottom_height.view(1, -1)

        # positive direction of the gravity axis
        # in cam coord system points to the earth
        heighest_of_bottom = torch.min(boxes1_bottom_height,
                                       boxes2_bottom_height)
        lowest_of_top = torch.max(boxes1_top_height, boxes2_top_height)
        overlaps_h = torch.clamp(heighest_of_bottom - lowest_of_top, min=0)
        return overlaps_h

    def convert_to(self, dst, rt_mat=None):
        """Convert self to ``dst`` mode.

        Args:
            dst (:obj:`Box3DMode`): The target Box mode.
            rt_mat (np.ndarray | torch.Tensor, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None.
                The conversion from ``src`` coordinates to ``dst`` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.

        Returns:
            :obj:`BaseInstance3DBoxes`:
                The converted box of the same type in the ``dst`` mode.
        """
        from .box_3d_mode import Box3DMode
        return Box3DMode.convert(
            box=self, src=Box3DMode.CAM, dst=dst, rt_mat=rt_mat)

    def points_in_boxes_part(self, points, boxes_override=None):
        """Find the box in which each point is.

        Args:
            points (torch.Tensor): Points in shape (1, M, 3) or (M, 3),
                3 dimensions are (x, y, z) in LiDAR or depth coordinate.
            boxes_override (torch.Tensor, optional): Boxes to override
                `self.tensor `. Defaults to None.

        Returns:
            torch.Tensor: The index of the box in which
                each point is, in shape (M, ). Default value is -1
                (if the point is not enclosed by any box).
        """
        from .coord_3d_mode import Coord3DMode

        points_lidar = Coord3DMode.convert(points, Coord3DMode.CAM,
                                           Coord3DMode.LIDAR)
        if boxes_override is not None:
            boxes_lidar = boxes_override
        else:
            boxes_lidar = Coord3DMode.convert(self.tensor, Coord3DMode.CAM,
                                              Coord3DMode.LIDAR)

        box_idx = super().points_in_boxes_part(points_lidar, boxes_lidar)
        return box_idx

    def points_in_boxes_all(self, points, boxes_override=None):
        """Find all boxes in which each point is.

        Args:
            points (torch.Tensor): Points in shape (1, M, 3) or (M, 3),
                3 dimensions are (x, y, z) in LiDAR or depth coordinate.
            boxes_override (torch.Tensor, optional): Boxes to override
                `self.tensor `. Defaults to None.

        Returns:
            torch.Tensor: The index of all boxes in which each point is,
                in shape (B, M, T).
        """
        from .coord_3d_mode import Coord3DMode

        points_lidar = Coord3DMode.convert(points, Coord3DMode.CAM,
                                           Coord3DMode.LIDAR)
        if boxes_override is not None:
            boxes_lidar = boxes_override
        else:
            boxes_lidar = Coord3DMode.convert(self.tensor, Coord3DMode.CAM,
                                              Coord3DMode.LIDAR)

        box_idx = super().points_in_boxes_all(points_lidar, boxes_lidar)
        return box_idx
