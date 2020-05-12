import torch

from . import iou3d_cuda


def boxes_iou_bev(boxes_a, boxes_b):
    """
    :param boxes_a: (M, 5)
    :param boxes_b: (N, 5)
    :return:
        ans_iou: (M, N)
    """

    ans_iou = torch.cuda.FloatTensor(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()

    iou3d_cuda.boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(),
                                 ans_iou)

    return ans_iou


def boxes_iou3d_gpu_camera(boxes_a, boxes_b, mode='iou'):
    """Calculate 3d iou of boxes in camera coordinate

    Args:
        boxes_a (FloatTensor): (N, 7) [x, y, z, h, w, l, ry]
            in LiDAR coordinate
        boxes_b (FloatTensor): (M, 7) [x, y, z, h, w, l, ry]
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        FloatTensor: (M, N)
    """

    boxes_a_bev = boxes3d_to_bev_torch_camera(boxes_a)
    boxes_b_bev = boxes3d_to_bev_torch_camera(boxes_b)

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    iou3d_cuda.boxes_overlap_bev_gpu(boxes_a_bev.contiguous(),
                                     boxes_b_bev.contiguous(), overlaps_bev)

    # height overlap
    boxes_a_height_min = (boxes_a[:, 1] - boxes_a[:, 3]).view(-1, 1)
    boxes_a_height_max = boxes_a[:, 1].view(-1, 1)
    boxes_b_height_min = (boxes_b[:, 1] - boxes_b[:, 3]).view(1, -1)
    boxes_b_height_max = boxes_b[:, 1].view(1, -1)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    volume_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    volume_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    if mode == 'iou':
        # the clamp func is used to avoid division of 0
        iou3d = overlaps_3d / torch.clamp(
            volume_a + volume_b - overlaps_3d, min=1e-8)
    else:
        iou3d = overlaps_3d / torch.clamp(volume_a, min=1e-8)

    return iou3d


def boxes_iou3d_gpu_lidar(boxes_a, boxes_b, mode='iou'):
    """Calculate 3d iou of boxes in lidar coordinate

    Args:
        boxes_a (FloatTensor): (N, 7) [x, y, z, w, l, h, ry]
            in LiDAR coordinate
        boxes_b (FloatTensor): (M, 7) [x, y, z, w, l, h, ry]
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    :Returns:
        FloatTensor: (M, N)
    """
    boxes_a_bev = boxes3d_to_bev_torch_lidar(boxes_a)
    boxes_b_bev = boxes3d_to_bev_torch_lidar(boxes_b)
    # height overlap
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5]).view(-1, 1)
    boxes_a_height_min = boxes_a[:, 2].view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5]).view(1, -1)
    boxes_b_height_min = boxes_b[:, 2].view(1, -1)

    # bev overlap
    overlaps_bev = boxes_a.new_zeros(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0])))  # (N, M)
    iou3d_cuda.boxes_overlap_bev_gpu(boxes_a_bev.contiguous(),
                                     boxes_b_bev.contiguous(), overlaps_bev)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    volume_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    volume_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    if mode == 'iou':
        # the clamp func is used to avoid division of 0
        iou3d = overlaps_3d / torch.clamp(
            volume_a + volume_b - overlaps_3d, min=1e-8)
    else:
        iou3d = overlaps_3d / torch.clamp(volume_a, min=1e-8)

    return iou3d


def nms_gpu(boxes, scores, thresh):
    """
    :param boxes: (N, 5) [x1, y1, x2, y2, ry]
    :param scores: (N)
    :param thresh:
    :return:
    """
    # areas = (x2 - x1) * (y2 - y1)
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_cuda.nms_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous()


def nms_normal_gpu(boxes, scores, thresh):
    """
    :param boxes: (N, 5) [x1, y1, x2, y2, ry]
    :param scores: (N)
    :param thresh:
    :return:
    """
    # areas = (x2 - x1) * (y2 - y1)
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_cuda.nms_normal_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous()


def boxes3d_to_bev_torch_camera(boxes3d):
    """covert boxes3d to bev in in camera coords

    Args:
        boxes3d (FloartTensor): (N, 7) [x, y, z, h, w, l, ry] in camera coords

    Return:
        FloartTensor: (N, 5) [x1, y1, x2, y2, ry]
    """
    boxes_bev = boxes3d.new(torch.Size((boxes3d.shape[0], 5)))

    cu, cv = boxes3d[:, 0], boxes3d[:, 2]
    half_l, half_w = boxes3d[:, 5] / 2, boxes3d[:, 4] / 2
    boxes_bev[:, 0], boxes_bev[:, 1] = cu - half_l, cv - half_w
    boxes_bev[:, 2], boxes_bev[:, 3] = cu + half_l, cv + half_w
    boxes_bev[:, 4] = boxes3d[:, 6]
    return boxes_bev


def boxes3d_to_bev_torch_lidar(boxes3d):
    """covert boxes3d to bev in in LiDAR coords

    Args:
        boxes3d (FloartTensor): (N, 7) [x, y, z, w, l, h, ry] in LiDAR coords

    Returns:
        FloartTensor: (N, 5) [x1, y1, x2, y2, ry]
    """
    boxes_bev = boxes3d.new(torch.Size((boxes3d.shape[0], 5)))

    x, y = boxes3d[:, 0], boxes3d[:, 1]
    half_l, half_w = boxes3d[:, 4] / 2, boxes3d[:, 3] / 2
    boxes_bev[:, 0], boxes_bev[:, 1] = x - half_w, y - half_l
    boxes_bev[:, 2], boxes_bev[:, 3] = x + half_w, y + half_l
    boxes_bev[:, 4] = boxes3d[:, 6]
    return boxes_bev
