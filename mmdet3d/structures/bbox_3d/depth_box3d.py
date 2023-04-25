# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from mmdet3d.structures.points import BasePoints
from .base_box3d import BaseInstance3DBoxes
from .utils import rotation_3d_in_axis


class DepthInstance3DBoxes(BaseInstance3DBoxes):
    """3D boxes of instances in DEPTH coordinates.

    Coordinates in Depth:

    .. code-block:: none

        up z    y front (yaw=0.5*pi)
           ^   ^
           |  /
           | /
           0 ------> x right (yaw=0)

    The relative coordinate of bottom center in a Depth box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2. The yaw is 0 at
    the positive direction of x axis, and increases from the positive direction
    of x to the positive direction of y.

    Attributes:
        tensor (Tensor): Float matrix with shape (N, box_dim).
        box_dim (int): Integer indicating the dimension of a box. Each row is
            (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """
    YAW_AXIS = 2

    @property
    def corners(self) -> Tensor:
        """Convert boxes to corners in clockwise order, in the form of (x0y0z0,
        x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0).

        .. code-block:: none

                                        up z
                         front y           ^
                              /            |
                             /             |
               (x0, y1, z1) + -----------  + (x1, y1, z1)
                           /|            / |
                          / |           /  |
            (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                         |  /      .   |  /
                         | / origin    | /
            (x0, y0, z0) + ----------- + --------> right x
                                       (x1, y0, z0)

        Returns:
            Tensor: A tensor with 8 corners of each box in shape (N, 8, 3).
        """
        if self.tensor.numel() == 0:
            return torch.empty([0, 8, 3], device=self.tensor.device)

        dims = self.dims
        corners_norm = torch.from_numpy(
            np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)).to(
                device=dims.device, dtype=dims.dtype)

        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        # use relative origin (0.5, 0.5, 0)
        corners_norm = corners_norm - dims.new_tensor([0.5, 0.5, 0])
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

        # rotate around z axis
        corners = rotation_3d_in_axis(
            corners, self.tensor[:, 6], axis=self.YAW_AXIS)
        corners += self.tensor[:, :3].view(-1, 1, 3)
        return corners

    def rotate(
        self,
        angle: Union[Tensor, np.ndarray, float],
        points: Optional[Union[Tensor, np.ndarray, BasePoints]] = None
    ) -> Union[Tuple[Tensor, Tensor], Tuple[np.ndarray, np.ndarray], Tuple[
            BasePoints, Tensor], None]:
        """Rotate boxes with points (optional) with the given angle or rotation
        matrix.

        Args:
            angle (Tensor or np.ndarray or float): Rotation angle or rotation
                matrix.
            points (Tensor or np.ndarray or :obj:`BasePoints`, optional):
                Points to rotate. Defaults to None.

        Returns:
            tuple or None: When ``points`` is None, the function returns None,
            otherwise it returns the rotated points and the rotation matrix
            ``rot_mat_T``.
        """
        if not isinstance(angle, Tensor):
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
            rot_sin = rot_mat_T[0, 1]
            rot_cos = rot_mat_T[0, 0]
            angle = np.arctan2(rot_sin, rot_cos)
            self.tensor[:, 0:3] = self.tensor[:, 0:3] @ rot_mat_T

        if self.with_yaw:
            self.tensor[:, 6] += angle
        else:
            # for axis-aligned boxes, we take the new
            # enclosing axis-aligned boxes after rotation
            corners_rot = self.corners @ rot_mat_T
            new_x_size = corners_rot[..., 0].max(
                dim=1, keepdim=True)[0] - corners_rot[..., 0].min(
                    dim=1, keepdim=True)[0]
            new_y_size = corners_rot[..., 1].max(
                dim=1, keepdim=True)[0] - corners_rot[..., 1].min(
                    dim=1, keepdim=True)[0]
            self.tensor[:, 3:5] = torch.cat((new_x_size, new_y_size), dim=-1)

        if points is not None:
            if isinstance(points, Tensor):
                points[:, :3] = points[:, :3] @ rot_mat_T
            elif isinstance(points, np.ndarray):
                rot_mat_T = rot_mat_T.cpu().numpy()
                points[:, :3] = np.dot(points[:, :3], rot_mat_T)
            elif isinstance(points, BasePoints):
                points.rotate(rot_mat_T)
            else:
                raise ValueError
            return points, rot_mat_T

    def flip(
        self,
        bev_direction: str = 'horizontal',
        points: Optional[Union[Tensor, np.ndarray, BasePoints]] = None
    ) -> Union[Tensor, np.ndarray, BasePoints, None]:
        """Flip the boxes in BEV along given BEV direction.

        In Depth coordinates, it flips the x (horizontal) or y (vertical) axis.

        Args:
            bev_direction (str): Direction by which to flip. Can be chosen from
                'horizontal' and 'vertical'. Defaults to 'horizontal'.
            points (Tensor or np.ndarray or :obj:`BasePoints`, optional):
                Points to flip. Defaults to None.

        Returns:
            Tensor or np.ndarray or :obj:`BasePoints` or None: When ``points``
            is None, the function returns None, otherwise it returns the
            flipped points.
        """
        assert bev_direction in ('horizontal', 'vertical')
        if bev_direction == 'horizontal':
            self.tensor[:, 0::7] = -self.tensor[:, 0::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6] + np.pi
        elif bev_direction == 'vertical':
            self.tensor[:, 1::7] = -self.tensor[:, 1::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6]

        if points is not None:
            assert isinstance(points, (Tensor, np.ndarray, BasePoints))
            if isinstance(points, (Tensor, np.ndarray)):
                if bev_direction == 'horizontal':
                    points[:, 0] = -points[:, 0]
                elif bev_direction == 'vertical':
                    points[:, 1] = -points[:, 1]
            elif isinstance(points, BasePoints):
                points.flip(bev_direction)
            return points

    def convert_to(self,
                   dst: int,
                   rt_mat: Optional[Union[Tensor, np.ndarray]] = None,
                   correct_yaw: bool = False) -> 'BaseInstance3DBoxes':
        """Convert self to ``dst`` mode.

        Args:
            dst (int): The target Box mode.
            rt_mat (Tensor or np.ndarray, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None. The conversion from ``src`` coordinates to
                ``dst`` coordinates usually comes along the change of sensors,
                e.g., from camera to LiDAR. This requires a transformation
                matrix.
            correct_yaw (bool): Whether to convert the yaw angle to the target
                coordinate. Defaults to False.

        Returns:
            :obj:`BaseInstance3DBoxes`: The converted box of the same type in
            the ``dst`` mode.
        """
        from .box_3d_mode import Box3DMode
        return Box3DMode.convert(
            box=self,
            src=Box3DMode.DEPTH,
            dst=dst,
            rt_mat=rt_mat,
            correct_yaw=correct_yaw)

    def enlarged_box(
            self, extra_width: Union[float, Tensor]) -> 'DepthInstance3DBoxes':
        """Enlarge the length, width and height of boxes.

        Args:
            extra_width (float or Tensor): Extra width to enlarge the box.

        Returns:
            :obj:`DepthInstance3DBoxes`: Enlarged boxes.
        """
        enlarged_boxes = self.tensor.clone()
        enlarged_boxes[:, 3:6] += extra_width * 2
        # bottom center z minus extra_width
        enlarged_boxes[:, 2] -= extra_width
        return self.new_box(enlarged_boxes)

    def get_surface_line_center(self) -> Tuple[Tensor, Tensor]:
        """Compute surface and line center of bounding boxes.

        Returns:
            Tuple[Tensor, Tensor]: Surface and line center of bounding boxes.
        """
        obj_size = self.dims
        center = self.gravity_center.view(-1, 1, 3)
        batch_size = center.shape[0]

        rot_sin = torch.sin(-self.yaw)
        rot_cos = torch.cos(-self.yaw)
        rot_mat_T = self.yaw.new_zeros(tuple(list(self.yaw.shape) + [3, 3]))
        rot_mat_T[..., 0, 0] = rot_cos
        rot_mat_T[..., 0, 1] = -rot_sin
        rot_mat_T[..., 1, 0] = rot_sin
        rot_mat_T[..., 1, 1] = rot_cos
        rot_mat_T[..., 2, 2] = 1

        # Get the object surface center
        offset = obj_size.new_tensor([[0, 0, 1], [0, 0, -1], [0, 1, 0],
                                      [0, -1, 0], [1, 0, 0], [-1, 0, 0]])
        offset = offset.view(1, 6, 3) / 2
        surface_3d = (offset *
                      obj_size.view(batch_size, 1, 3).repeat(1, 6, 1)).reshape(
                          -1, 3)

        # Get the object line center
        offset = obj_size.new_tensor([[1, 0, 1], [-1, 0, 1], [0, 1, 1],
                                      [0, -1, 1], [1, 0, -1], [-1, 0, -1],
                                      [0, 1, -1], [0, -1, -1], [1, 1, 0],
                                      [1, -1, 0], [-1, 1, 0], [-1, -1, 0]])
        offset = offset.view(1, 12, 3) / 2

        line_3d = (offset *
                   obj_size.view(batch_size, 1, 3).repeat(1, 12, 1)).reshape(
                       -1, 3)

        surface_rot = rot_mat_T.repeat(6, 1, 1)
        surface_3d = torch.matmul(surface_3d.unsqueeze(-2),
                                  surface_rot).squeeze(-2)
        surface_center = center.repeat(1, 6, 1).reshape(-1, 3) + surface_3d

        line_rot = rot_mat_T.repeat(12, 1, 1)
        line_3d = torch.matmul(line_3d.unsqueeze(-2), line_rot).squeeze(-2)
        line_center = center.repeat(1, 12, 1).reshape(-1, 3) + line_3d

        return surface_center, line_center
