import torch

from . import iou3d_cuda


def boxes_iou_bev(boxes_a, boxes_b):
    """Get iou of the bev boxes.

    Args:
        boxes_a (torch.Tensor): Input boxes a with the shape of [M, 5].
        boxes_b (torch.Tensor): Input boxes b with the shape of [M, 5].

    Returns:
        torch.Tensor: Result iou with the shaoe of [M, N].
    """

    ans_iou = torch.cuda.FloatTensor(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()

    iou3d_cuda.boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(),
                                 ans_iou)

    return ans_iou


def nms_gpu(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
    """Nms function with gpu implementation.

    Args:
        boxes (torch.Tensor): Input boxes with the shape of [N, 5]
            ([x1, y1, x2, y2, ry]).
        scores (torch.Tensor): Scores of boxes with the shape of [N].
        thresh (int): Threshold.
        pre_maxsize (int): Max size of boxes before nms. Default: None.
        post_maxsize (int): Max size of boxes after nms. Default: None.

    Returns:
        torch.Tensor: Indexes after nms.
    """
    # areas = (x2 - x1) * (y2 - y1)
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))
    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    num_out = iou3d_cuda.nms_gpu(boxes, keep, thresh)
    if post_max_size is not None:
        keep = keep[:post_max_size]
    return order[keep[:num_out].cuda()].contiguous()


def nms_normal_gpu(boxes, scores, thresh):
    """Normal nms function with gpu implementation.

    Args:
        boxes (torch.Tensor): Input boxes with the shape of [N, 5]
            ([x1, y1, x2, y2, ry]).
        scores (torch.Tensor): Scores of boxes with the shape of [N].
        thresh (int): Threshold.

    Returns:
        torch.Tensor: Indexes after nms.
    """
    # areas = (x2 - x1) * (y2 - y1)
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_cuda.nms_normal_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous()
