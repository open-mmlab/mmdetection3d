# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

import numpy as np
from torch import Tensor

from .base_points import BasePoints


class LiDARPoints(BasePoints):
    """Points of instances in LIDAR coordinates.

    Args:
        tensor (Tensor or np.ndarray or Sequence[Sequence[float]]): The points
            data with shape (N, points_dim).
        points_dim (int): Integer indicating the dimension of a point. Each row
            is (x, y, z, ...). Defaults to 3.
        attribute_dims (dict, optional): Dictionary to indicate the meaning of
            extra dimension. Defaults to None.

    Attributes:
        tensor (Tensor): Float matrix with shape (N, points_dim).
        points_dim (int): Integer indicating the dimension of a point. Each row
            is (x, y, z, ...).
        attribute_dims (dict, optional): Dictionary to indicate the meaning of
            extra dimension. Defaults to None.
        rotation_axis (int): Default rotation axis for points rotation.
    """

    def __init__(self,
                 tensor: Union[Tensor, np.ndarray, Sequence[Sequence[float]]],
                 points_dim: int = 3,
                 attribute_dims: Optional[dict] = None) -> None:
        super(LiDARPoints, self).__init__(
            tensor, points_dim=points_dim, attribute_dims=attribute_dims)
        self.rotation_axis = 2

    def flip(self, bev_direction: str = 'horizontal') -> None:
        """Flip the points along given BEV direction.

        Args:
            bev_direction (str): Flip direction (horizontal or vertical).
                Defaults to 'horizontal'.
        """
        assert bev_direction in ('horizontal', 'vertical')
        if bev_direction == 'horizontal':
            self.tensor[:, 1] = -self.tensor[:, 1]
        elif bev_direction == 'vertical':
            self.tensor[:, 0] = -self.tensor[:, 0]

    def convert_to(self,
                   dst: int,
                   rt_mat: Optional[Union[Tensor,
                                          np.ndarray]] = None) -> 'BasePoints':
        """Convert self to ``dst`` mode.

        Args:
            dst (int): The target Point mode.
            rt_mat (Tensor or np.ndarray, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None. The conversion from ``src`` coordinates to
                ``dst`` coordinates usually comes along the change of sensors,
                e.g., from camera to LiDAR. This requires a transformation
                matrix.

        Returns:
            :obj:`BasePoints`: The converted point of the same type in the
            ``dst`` mode.
        """
        from mmdet3d.structures.bbox_3d import Coord3DMode
        return Coord3DMode.convert_point(
            point=self, src=Coord3DMode.LIDAR, dst=dst, rt_mat=rt_mat)
