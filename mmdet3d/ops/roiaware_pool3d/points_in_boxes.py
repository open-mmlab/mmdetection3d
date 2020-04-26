import torch

from . import roiaware_pool3d_ext


def points_in_boxes_gpu(points, boxes):
    """
    Args:
        points (torch.Tensor): [B, M, 3], [x, y, z] in LiDAR coordinate
        boxes (torch.Tensor): [B, T, 7],
            num_valid_boxes <= T, [x, y, z, w, l, h, ry] in LiDAR coordinate,
            (x, y, z) is the bottom center
    Returns:
        box_idxs_of_pts (torch.Tensor): (B, M), default background = -1
    """
    assert boxes.shape[0] == points.shape[0]
    assert boxes.shape[2] == 7
    batch_size, num_points, _ = points.shape

    box_idxs_of_pts = points.new_zeros((batch_size, num_points),
                                       dtype=torch.int).fill_(-1)
    roiaware_pool3d_ext.points_in_boxes_gpu(boxes.contiguous(),
                                            points.contiguous(),
                                            box_idxs_of_pts)

    return box_idxs_of_pts


def points_in_boxes_cpu(points, boxes):
    """
    Args:
        points (torch.Tensor): [npoints, 3]
        boxes (torch.Tensor): [N, 7], in LiDAR coordinate,
            (x, y, z) is the bottom center
    Returns:
        point_indices (torch.Tensor): (N, npoints)
    """
    assert boxes.shape[1] == 7
    assert points.shape[1] == 3

    point_indices = points.new_zeros((boxes.shape[0], points.shape[0]),
                                     dtype=torch.int)
    roiaware_pool3d_ext.points_in_boxes_cpu(boxes.float().contiguous(),
                                            points.float().contiguous(),
                                            point_indices)

    return point_indices
