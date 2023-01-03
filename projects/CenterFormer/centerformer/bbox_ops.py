import torch
from mmcv.utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['iou3d_nms3d_forward'])


def nms_iou3d(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
    """NMS function GPU implementation (using IoU3D). The difference between
    this implementation and nms3d in MMCV is that we add `pre_maxsize` and
    `post_max_size` before and after NMS respectively.

     Args:
        boxes (Tensor): Input boxes with the shape of [N, 7]
            ([cx, cy, cz, l, w, h, theta]).
        scores (Tensor): Scores of boxes with the shape of [N].
        thresh (float): Overlap threshold of NMS.
        pre_max_size (int, optional): Max size of boxes before NMS.
            Defaults to None.
        post_max_size (int, optional): Max size of boxes after NMS.
            Defaults to None.

    Returns:
        Tensor: Indexes after NMS.
    """
    # TODO: directly refactor ``nms3d`` in MMCV
    assert boxes.size(1) == 7, 'Input boxes shape should be (N, 7)'
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    boxes = boxes[order].contiguous()

    keep = boxes.new_zeros(boxes.size(0), dtype=torch.long)
    num_out = boxes.new_zeros(size=(), dtype=torch.long)
    ext_module.iou3d_nms3d_forward(
        boxes, keep, num_out, nms_overlap_thresh=thresh)
    keep = order[keep[:num_out].to(boxes.device)].contiguous()

    if post_max_size is not None:
        keep = keep[:post_max_size]

    return keep
