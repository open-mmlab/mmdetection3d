# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.models import DETECTORS
from .two_stage import TwoStage3DDetector


@DETECTORS.register_module()
class PointRCNN(TwoStage3DDetector):
    r"""PointRCNN detector.

    Please refer to the `PointRCNN <https://arxiv.org/abs/1812.04244>`_

    Args:
        backbone (dict): Config dict of detector's backbone.
        neck (dict, optional): Config dict of neck. Defaults to None.
        rpn_head (dict, optional): Config of RPN head. Defaults to None.
        roi_head (dict, optional): Config of ROI head. Defaults to None.
        train_cfg (dict, optional): Train configs. Defaults to None.
        test_cfg (dict, optional): Test configs. Defaults to None.
        pretrained (str, optional): Model pretrained path. Defaults to None.
        init_cfg (dict, optional): Config of initialization. Defaults to None.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(PointRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

    def extract_feat(self, points):
        """Directly extract features from the backbone+neck.

        Args:
            points (torch.Tensor): Input points.

        Returns:
            dict: Features from the backbone+neck
        """
        x = self.backbone(points)

        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, points, img_metas, gt_bboxes_3d, gt_labels_3d):
        """Forward of training.

        Args:
            points (list[torch.Tensor]): Points of each batch.
            img_metas (list[dict]): Meta information of each sample.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): gt bboxes of each batch.
            gt_labels_3d (list[torch.Tensor]): gt class labels of each batch.

        Returns:
            dict: Losses.
        """
        losses = dict()
        points_cat = torch.stack(points)
        x = self.extract_feat(points_cat)

        # features for rcnn
        backbone_feats = x['fp_features'].clone()
        backbone_xyz = x['fp_xyz'].clone()
        rcnn_feats = {'features': backbone_feats, 'points': backbone_xyz}

        bbox_preds, cls_preds = self.rpn_head(x)

        rpn_loss = self.rpn_head.loss(
            bbox_preds=bbox_preds,
            cls_preds=cls_preds,
            points=points,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            img_metas=img_metas)
        losses.update(rpn_loss)

        bbox_list = self.rpn_head.get_bboxes(points_cat, bbox_preds, cls_preds,
                                             img_metas)
        proposal_list = [
            dict(
                boxes_3d=bboxes,
                scores_3d=scores,
                labels_3d=labels,
                cls_preds=preds_cls)
            for bboxes, scores, labels, preds_cls in bbox_list
        ]
        rcnn_feats.update({'points_cls_preds': cls_preds})

        roi_losses = self.roi_head.forward_train(rcnn_feats, img_metas,
                                                 proposal_list, gt_bboxes_3d,
                                                 gt_labels_3d)
        losses.update(roi_losses)

        return losses

    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        """Forward of testing.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            img_metas (list[dict]): Image metas.
            imgs (list[torch.Tensor], optional): Images of each sample.
                Defaults to None.
            rescale (bool, optional): Whether to rescale results.
                Defaults to False.

        Returns:
            list: Predicted 3d boxes.
        """
        points_cat = torch.stack(points)

        x = self.extract_feat(points_cat)
        # features for rcnn
        backbone_feats = x['fp_features'].clone()
        backbone_xyz = x['fp_xyz'].clone()
        rcnn_feats = {'features': backbone_feats, 'points': backbone_xyz}
        bbox_preds, cls_preds = self.rpn_head(x)
        rcnn_feats.update({'points_cls_preds': cls_preds})

        bbox_list = self.rpn_head.get_bboxes(
            points_cat, bbox_preds, cls_preds, img_metas, rescale=rescale)

        proposal_list = [
            dict(
                boxes_3d=bboxes,
                scores_3d=scores,
                labels_3d=labels,
                cls_preds=preds_cls)
            for bboxes, scores, labels, preds_cls in bbox_list
        ]
        bbox_results = self.roi_head.simple_test(rcnn_feats, img_metas,
                                                 proposal_list)

        return bbox_results
