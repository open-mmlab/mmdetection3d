# Copyright (c) OpenMMLab. All rights reserved.
from .base_points import BasePoints


class CameraPoints(BasePoints):
    """Points of instances in CAM coordinates.

    Args:
        tensor (torch.Tensor | np.ndarray | list): a N x points_dim matrix.
        points_dim (int, optional): Number of the dimension of a point.
            Each row is (x, y, z). Defaults to 3.
        attribute_dims (dict, optional): Dictionary to indicate the
            meaning of extra dimension. Defaults to None.

    Attributes:
        tensor (torch.Tensor): Float matrix of N x points_dim.
        points_dim (int): Integer indicating the dimension of a point.
            Each row is (x, y, z, ...).
        attribute_dims (bool): Dictionary to indicate the meaning of extra
            dimension. Defaults to None.
        rotation_axis (int): Default rotation axis for points rotation.
    """

    def __init__(self, tensor, points_dim=3, attribute_dims=None):
        super(CameraPoints, self).__init__(
            tensor, points_dim=points_dim, attribute_dims=attribute_dims)
        self.rotation_axis = 1

    def flip(self, bev_direction='horizontal'):
        """Flip the points along given BEV direction.

        Args:
            bev_direction (str): Flip direction (horizontal or vertical).
        """
        if bev_direction == 'horizontal':
            self.tensor[:, 0] = -self.tensor[:, 0]
        elif bev_direction == 'vertical':
            self.tensor[:, 2] = -self.tensor[:, 2]

    @property
    def bev(self):
        """torch.Tensor: BEV of the points in shape (N, 2)."""
        return self.tensor[:, [0, 2]]

    def convert_to(self, dst, rt_mat=None):
        """Convert self to ``dst`` mode.

        Args:
            dst (:obj:`CoordMode`): The target Point mode.
            rt_mat (np.ndarray | torch.Tensor, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None.
                The conversion from `src` coordinates to `dst` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.

        Returns:
            :obj:`BasePoints`: The converted point of the same type
                in the `dst` mode.
        """
        from mmdet3d.core.bbox import Coord3DMode
        return Coord3DMode.convert_point(
            point=self, src=Coord3DMode.CAM, dst=dst, rt_mat=rt_mat)
