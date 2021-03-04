from .base_points import BasePoints


class DepthPoints(BasePoints):
    """Points of instances in DEPTH coordinates.

    Args:
        tensor (torch.Tensor | np.ndarray | list): a N x points_dim matrix.
        points_dim (int): Number of the dimension of a point.
            Each row is (x, y, z). Default to 3.
        attribute_dims (dict): Dictionary to indicate the meaning of extra
            dimension. Default to None.

    Attributes:
        tensor (torch.Tensor): Float matrix of N x points_dim.
        points_dim (int): Integer indicating the dimension of a point.
            Each row is (x, y, z, ...).
        attribute_dims (bool): Dictionary to indicate the meaning of extra
            dimension. Default to None.
        rotation_axis (int): Default rotation axis for points rotation.
    """

    def __init__(self, tensor, points_dim=3, attribute_dims=None):
        super(DepthPoints, self).__init__(
            tensor, points_dim=points_dim, attribute_dims=attribute_dims)
        self.rotation_axis = 2

    def flip(self, bev_direction='horizontal'):
        """Flip the boxes in BEV along given BEV direction."""
        if bev_direction == 'horizontal':
            self.tensor[:, 0] = -self.tensor[:, 0]
        elif bev_direction == 'vertical':
            self.tensor[:, 1] = -self.tensor[:, 1]

    def in_range_bev(self, point_range):
        """Check whether the points are in the given range.

        Args:
            point_range (list | torch.Tensor): The range of point
                in order of (x_min, y_min, x_max, y_max).

        Returns:
            torch.Tensor: Indicating whether each point is inside \
                the reference range.
        """
        in_range_flags = ((self.tensor[:, 0] > point_range[0])
                          & (self.tensor[:, 1] > point_range[1])
                          & (self.tensor[:, 0] < point_range[2])
                          & (self.tensor[:, 1] < point_range[3]))
        return in_range_flags

    def convert_to(self, dst, rt_mat=None):
        """Convert self to ``dst`` mode.

        Args:
            dst (:obj:`CoordMode`): The target Point mode.
            rt_mat (np.ndarray | torch.Tensor): The rotation and translation
                matrix between different coordinates. Defaults to None.
                The conversion from `src` coordinates to `dst` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.

        Returns:
            :obj:`BasePoints`: The converted point of the same type \
                in the `dst` mode.
        """
        from mmdet3d.core.bbox import Coord3DMode
        return Coord3DMode.convert_point(
            point=self, src=Coord3DMode.DEPTH, dst=dst, rt_mat=rt_mat)
