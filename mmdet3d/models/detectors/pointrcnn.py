import torch
from torch import nn as nn

from mmdet3d.core import merge_aug_bboxes_3d
from mmdet3d.core.bbox import bbox3d2result
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
                 fp_channels=None,
                 pretrained=None):
        super(PointRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

    def extract_feat(self, points, img_metas=None):
        """Directly extract features from the backbone+neck.

        Args:
            points (torch.Tensor): Input points.
        """
        x = self.backbone(points)

        if self.with_neck:
            x = self.neck(x)
        return x
    
    def 

    def forward_train(self,
                      points,
                      img_metas,
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
        losses = dict()
        points_cat = torch.stack(points)
        points_intensity = points_cat[:, :, 3]
        x = self.extract_feat(points_cat)
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss = self.rpn_head.loss(
                bbox_preds=rpn_outs,
                points=points,
                gt_bboxes_3d=gt_bboxes_3d,
                gt_labels_3d=gt_labels_3d,
                img_metas=img_metas,
                gt_bboxes_ignore=gt_bboxes_ignore)
            proposal_list = self.rpn_head.get_bboxes(
                points_cat, bbox_preds, img_metas, rescale=rescale)

        if self.with_rcnn:
            roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                     gt_bboxes_3d, gt_labels_3d)

            losses.update(rpn_loss)

        return losses

    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        """Forward of testing.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            img_metas (list): Image metas.
            rescale (bool): Whether to rescale results.

        Returns:
            list: Predicted 3d boxes.
        """
        points_cat = torch.stack(points)

        x = self.extract_feat(points_cat)
        bbox_preds = self.rpn_head(x)
        bbox_list = self.rpn_head.get_bboxes(
            points_cat, bbox_preds, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results
