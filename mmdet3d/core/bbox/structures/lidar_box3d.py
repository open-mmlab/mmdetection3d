import numpy as np
import torch

from mmdet3d.ops.iou3d import iou3d_cuda
from .base_box3d import BaseInstance3DBoxes
from .utils import limit_period, rotation_3d_in_axis


class LiDARInstance3DBoxes(BaseInstance3DBoxes):
    """3D boxes of instances in LIDAR coordinates

    Coordinates in LiDAR:
    .. code-block:: none

                    up z    x front
                       ^   ^
                       |  /
                       | /
        left y <------ 0

    The relative coordinate of bottom center in a LiDAR box is [0.5, 0.5, 0],
    and the yaw is around the z axis, thus the rotation axis=2.

    Attributes:
        tensor (torch.Tensor): float matrix of N x box_dim.
        box_dim (int): integer indicates the dimension of a box
        Each row is (x, y, z, x_size, y_size, z_size, yaw, ...).
    """

    @property
    def gravity_center(self):
        """Calculate the gravity center of all the boxes.

        Returns:
            torch.Tensor: a tensor with center of each box.
        """
        bottom_center = self.bottom_center
        gravity_center = torch.zeros_like(bottom_center)
        gravity_center[:, :2] = bottom_center[:, :2]
        gravity_center[:, 2] = bottom_center[:, 2] + self.tensor[:, 5] * 0.5
        return gravity_center

    @property
    def corners(self):
        """Calculate the coordinates of corners of all the boxes.

        Convert the boxes to corners in clockwise order, in form of
        (x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z0, x1y1z1)

        .. code-block:: none

                                           up z
                            front x           ^
                                 /            |
                                /             |
                  (x1, y0, z1) + -----------  + (x1, y1, z1)
                              /|            / |
                             / |           /  |
               (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                            |  /      .   |  /
                            | / oriign    | /
            left y<-------- + ----------- + (x0, y1, z0)
                (x0, y0, z0)

        Returns:
            torch.Tensor: corners of each box with size (N, 8, 3)
        """
        dims = self.dims
        corners_norm = torch.from_numpy(
            np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)).to(
                device=dims.device, dtype=dims.dtype)

        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        # use relative origin [0.5, 0.5, 0]
        corners_norm = corners_norm - dims.new_tensor([0.5, 0.5, 0])
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

        # rotate around z axis
        corners = rotation_3d_in_axis(corners, self.tensor[:, 6], axis=2)
        corners += self.tensor[:, :3].view(-1, 1, 3)
        return corners

    @property
    def bev(self, mode='XYWHR'):
        """Calculate the 2D bounding boxes in BEV with rotation

        Args:
            mode (str): The mode of BEV boxes. Default to 'XYWHR'.

        Returns:
            torch.Tensor: a nx5 tensor of 2D BEV box of each box.
        """
        boxes_xywhr = self.tensor[:, [0, 1, 3, 4, 6]]
        if mode == 'XYWHR':
            return boxes_xywhr
        elif mode == 'XYXYR':
            boxes = torch.zeros_like(boxes_xywhr)
            boxes[:, 0] = boxes_xywhr[:, 0] - boxes_xywhr[2]
            boxes[:, 1] = boxes_xywhr[:, 1] - boxes_xywhr[3]
            boxes[:, 2] = boxes_xywhr[:, 0] + boxes_xywhr[2]
            boxes[:, 3] = boxes_xywhr[:, 1] + boxes_xywhr[3]
            boxes[:, 4] = boxes_xywhr[:, 4]
            return boxes
        else:
            raise ValueError(
                'Only support mode to be either "XYWHR" or "XYXYR",'
                f'got {mode}')

    @property
    def nearset_bev(self):
        """Calculate the 2D bounding boxes in BEV without rotation

        Returns:
            torch.Tensor: a tensor of 2D BEV box of each box.
        """
        # Obtain BEV boxes with rotation in XYWHR format
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
        """Flip the boxes in horizontal direction

        In LIDAR coordinates, it flips the y axis.
        """
        self.tensor[:, 1::7] = -self.tensor[:, 1::7]
        self.tensor[:, 6] = -self.tensor[:, 6] + np.pi

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

    @classmethod
    def overlaps(cls, boxes1, boxes2, mode='iou'):
        """Calculate overlaps of two boxes

        Args:
            boxes1 (:obj:LiDARInstanceBoxes): boxes 1 contain N boxes
            boxes2 (:obj:LiDARInstanceBoxes): boxes 2 contain M boxes
            mode (str, optional): mode of iou calculation. Defaults to 'iou'.

        Returns:
            torch.Tensor: Calculated iou of boxes
        """
        assert isinstance(boxes1, LiDARInstance3DBoxes)
        assert isinstance(boxes2, LiDARInstance3DBoxes)
        assert mode in ['iou', 'iof']

        # height overlap
        boxes1_height_max = (boxes1.tensor[:, 2] + boxes1.height).view(-1, 1)
        boxes1_height_min = boxes1.tensor[:, 2].view(-1, 1)
        boxes2_height_max = (boxes2.tensor[:, 2] + boxes2.height).view(1, -1)
        boxes2_height_min = boxes2.tensor[:, 2].view(1, -1)

        max_of_min = torch.max(boxes1_height_min, boxes2_height_min)
        min_of_max = torch.min(boxes1_height_max, boxes2_height_max)
        overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

        # obtain BEV boxes in XYXYR format
        boxes1_bev = boxes1.bev(mode='XYXYR')
        boxes2_bev = boxes2.bev(mode='XYXYR')

        # bev overlap
        overlaps_bev = boxes1_bev.new_zeros(
            (boxes1_bev.shape[0], boxes2_bev.shape[0])).cuda()  # (N, M)
        iou3d_cuda.boxes_overlap_bev_gpu(boxes1_bev.contiguous().cuda(),
                                         boxes2_bev.contiguous().cuda(),
                                         overlaps_bev)

        # 3d iou
        overlaps_3d = overlaps_bev.to(boxes1.device) * overlaps_h

        volume1 = boxes1.volume.view(-1, 1)
        volume2 = boxes2.volume.view(1, -1)

        if mode == 'iou':
            # the clamp func is used to avoid division of 0
            iou3d = overlaps_3d / torch.clamp(
                volume1 + volume2 - overlaps_3d, min=1e-8)
        else:
            iou3d = overlaps_3d / torch.clamp(volume1, min=1e-8)

        return iou3d
