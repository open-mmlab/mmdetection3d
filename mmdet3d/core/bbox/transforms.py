import torch


def bbox3d_mapping_back(bboxes, scale_factor, flip_horizontal, flip_vertical):
    """Map bboxes from testing scale to original image scale"""
    new_bboxes = bboxes.clone()
    if flip_horizontal:
        new_bboxes.flip('horizontal')
    if flip_vertical:
        new_bboxes.flip('vertical')
    new_bboxes.scale(1 / scale_factor)

    return new_bboxes


def transform_lidar_to_cam(boxes_lidar):
    """
    Only transform format, not exactly in camera coords
    :param boxes_lidar: (N, 3 or 7) [x, y, z, w, l, h, ry] in LiDAR coords
    :return: boxes_cam: (N, 3 or 7) [x, y, z, h, w, l, ry] in camera coords
    """
    # boxes_cam = boxes_lidar.new_tensor(boxes_lidar.data)
    boxes_cam = boxes_lidar.clone().detach()
    boxes_cam[:, 0] = -boxes_lidar[:, 1]
    boxes_cam[:, 1] = -boxes_lidar[:, 2]
    boxes_cam[:, 2] = boxes_lidar[:, 0]
    if boxes_cam.shape[1] > 3:
        boxes_cam[:, [3, 4, 5]] = boxes_lidar[:, [5, 3, 4]]
    return boxes_cam


def boxes3d_to_bev_torch(boxes3d):
    """
    :param boxes3d: (N, 7) [x, y, z, h, w, l, ry] in camera coords
    :return:
        boxes_bev: (N, 5) [x1, y1, x2, y2, ry]
    """
    boxes_bev = boxes3d.new(torch.Size((boxes3d.shape[0], 5)))

    cu, cv = boxes3d[:, 0], boxes3d[:, 2]
    half_l, half_w = boxes3d[:, 5] / 2, boxes3d[:, 4] / 2
    boxes_bev[:, 0], boxes_bev[:, 1] = cu - half_l, cv - half_w
    boxes_bev[:, 2], boxes_bev[:, 3] = cu + half_l, cv + half_w
    boxes_bev[:, 4] = boxes3d[:, 6]
    return boxes_bev


def boxes3d_to_bev_torch_lidar(boxes3d):
    """
    :param boxes3d: (N, 7) [x, y, z, w, l, h, ry] in LiDAR coords
    :return:
        boxes_bev: (N, 5) [x1, y1, x2, y2, ry]
    """
    boxes_bev = boxes3d.new(torch.Size((boxes3d.shape[0], 5)))

    cu, cv = boxes3d[:, 0], boxes3d[:, 1]
    half_l, half_w = boxes3d[:, 4] / 2, boxes3d[:, 3] / 2
    boxes_bev[:, 0], boxes_bev[:, 1] = cu - half_w, cv - half_l
    boxes_bev[:, 2], boxes_bev[:, 3] = cu + half_w, cv + half_l
    boxes_bev[:, 4] = boxes3d[:, 6]
    return boxes_bev


def bbox3d2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, c), [batch_ind, x, y ...]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes], dim=-1)
        else:
            rois = torch.zeros_like(bboxes)
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def bbox3d2result(bboxes, scores, labels):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        scores (Tensor): shape (n, )

    Returns:
        dict(Tensor): bbox results in cpu mode
    """
    return dict(
        boxes_3d=bboxes.to('cpu'),
        scores_3d=scores.cpu(),
        labels_3d=labels.cpu())


def upright_depth_to_lidar_torch(points=None,
                                 bboxes=None,
                                 to_bottom_center=False):
    """Convert points and boxes in upright depth coordinate to lidar.

    Args:
        points (None | Tensor): points in upright depth coordinate.
        bboxes (None | Tensor): bboxes in upright depth coordinate.
        to_bottom_center (bool): covert bboxes to bottom center.

    Returns:
        tuple: points and bboxes in lidar coordinate.
    """
    if points is not None:
        points_lidar = points.clone()
        points_lidar = points_lidar[..., [1, 0, 2]]
        points_lidar[..., 1] *= -1
    else:
        points_lidar = None

    if bboxes is not None:
        bboxes_lidar = bboxes.clone()
        bboxes_lidar = bboxes_lidar[..., [1, 0, 2, 4, 3, 5, 6]]
        bboxes_lidar[..., 1] *= -1
        if to_bottom_center:
            bboxes_lidar[..., 2] -= 0.5 * bboxes_lidar[..., 5]
    else:
        bboxes_lidar = None

    return points_lidar, bboxes_lidar


def box3d_to_corner3d_upright_depth(boxes3d):
    """Convert box3d to corner3d in upright depth coordinate

    Args:
        boxes3d (Tensor): boxes with shape [n,7] in upright depth coordinate

    Returns:
        Tensor: boxes with [n, 8, 3] in upright depth coordinate
    """
    boxes_num = boxes3d.shape[0]
    ry = boxes3d[:, 6:7]
    l, w, h = boxes3d[:, 3:4], boxes3d[:, 4:5], boxes3d[:, 5:6]
    zeros = boxes3d.new_zeros((boxes_num, 1))
    ones = boxes3d.new_ones((boxes_num, 1))
    # zeros = torch.cuda.FloatTensor(boxes_num, 1).fill_(0)
    # ones = torch.cuda.FloatTensor(boxes_num, 1).fill_(1)
    x_corners = torch.cat(
        [-l / 2., l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2.],
        dim=1)  # (N, 8)
    y_corners = torch.cat(
        [w / 2., w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2.],
        dim=1)  # (N, 8)
    z_corners = torch.cat(
        [h / 2., h / 2., h / 2., h / 2., -h / 2., -h / 2., -h / 2., -h / 2.],
        dim=1)  # (N, 8)
    temp_corners = torch.cat(
        (x_corners.unsqueeze(dim=2), y_corners.unsqueeze(dim=2),
         z_corners.unsqueeze(dim=2)),
        dim=2)  # (N, 8, 3)

    cosa, sina = torch.cos(-ry), torch.sin(-ry)
    raw_1 = torch.cat([cosa, -sina, zeros], dim=1)  # (N, 3)
    raw_2 = torch.cat([sina, cosa, zeros], dim=1)  # (N, 3)
    raw_3 = torch.cat([zeros, zeros, ones], dim=1)  # (N, 3)
    R = torch.cat((raw_1.unsqueeze(dim=1), raw_2.unsqueeze(dim=1),
                   raw_3.unsqueeze(dim=1)),
                  dim=1)  # (N, 3, 3)
    rotated_corners = torch.matmul(temp_corners, R)  # (N, 8, 3)
    x_corners = rotated_corners[:, :, 0]
    y_corners = rotated_corners[:, :, 1]
    z_corners = rotated_corners[:, :, 2]
    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.view(-1, 1) + x_corners.view(-1, 8)
    y = y_loc.view(-1, 1) + y_corners.view(-1, 8)
    z = z_loc.view(-1, 1) + z_corners.view(-1, 8)
    corners3d = torch.cat(
        (x.view(-1, 8, 1), y.view(-1, 8, 1), z.view(-1, 8, 1)), dim=2)

    return corners3d
