import torch

from mmcv.ops import nms3d


def nms_iou3d(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
    """NMS function GPU implementation (using IoU3D)

     Args:
        boxes (Tensor): Input boxes with the shape of [N, 5]
            ([x1, y1, x2, y2, ry]).
        scores (Tensor): Scores of boxes with the shape of [N].
        thresh (float): Overlap threshold of NMS.
        pre_max_size (int, optional): Max size of boxes before NMS.
            Defaults to None.
        post_max_size (int, optional): Max size of boxes after NMS.
            Defaults to None.

    Returns:
        Tensor: Indexes after NMS.
    """

    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))

    if len(boxes) == 0:
        num_out = 0
    else:
        num_out = nms3d(boxes, keep, thresh)

    selected = order[keep[:num_out].to(scores.device())].contiguous()

    if post_max_size is not None:
        selected = selected[:post_max_size]

    return selected
