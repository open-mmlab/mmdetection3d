import torch

from mmdet3d.core import merge_aug_bboxes_3d
from mmdet.models import DETECTORS
from .two_stage import TwoStage3DDetector


@DETECTORS.register_module()
class PointRCNN(TwoStage3DDetector):
    r"""PointRCNN model.

      Please refer to the `paper <https://arxiv.org/abs/1812.04244>`_
    """
    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(PointRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

def forward_train(self,
                  points,
                  gt_bboxes_3d,
                  gt_labels_3d,
                  pts_semantic_mask=None,
                  gt_bboxes_ignore=None):
    """Forward of training.

    Args:
        points (list[torch.Tensor]): Points of each batch.
        gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): gt bboxes of each batch.
        gt_labels_3d (list[torch.Tensor]): gt class labels of each batch.
        pts_semantic_mask (None | list[torch.Tensor]): point-wise semantic
            label of each batch.
        gt_bboxes_ignore (None | list[torch.Tensor]): Specify
            which bounding.

    Returns:
        dict: Losses.
    """
    points_cat = torch.stack(points)
    x = self.extract_feat(points_cat)

    if self.with_rpn:
        rpn_outs = self.rpn_head(x)
        rpn_loss = self.rpn_head.loss(
          rpn_outs,
          gt_bboxes_ignore=gt_bboxes_ignore
        )
    losses.update(rpn_loss)

    return loss
)
