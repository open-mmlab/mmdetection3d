from typing import Union

import torch
from torch import Tensor

from mmdet3d.registry import TASK_UTILS


@TASK_UTILS.register_module()
class BBox3DL1Cost(object):
    """BBox3DL1Cost.

    Args:
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """

    def __init__(self, weight: Union[float, int] = 1.):
        self.weight = weight

    def __call__(self, bbox_pred: Tensor, gt_bboxes: Tensor) -> Tensor:
        """Compute match cost.

        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx,cy,l,w,cz,h,sin(φ),cos(φ),v_x,v_y)
                which are all in range [0, 1] and shape [num_query, 10].
            gt_bboxes (Tensor): Ground truth boxes with `normalized`
                coordinates (cx,cy,l,w,cz,h,sin(φ),cos(φ),v_x,v_y).
                Shape [num_gt, 10].
        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight
