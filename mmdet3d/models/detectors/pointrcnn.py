import torch
from torch.nn import functional as F

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

    def extract_feat(self, points, img_metas=None):
        """Directly extract features from the backbone+neck.

        Args:
            points (torch.Tensor): Input points.
        """
        x = self.backbone(points)

        if self.with_neck:
            x = self.neck(x)
        return x

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
        x = self.extract_feat(points_cat)
        '''
        # features for rcnn
        backbone_feats = x['fp_features'][-1].clone()
        backbone_xyz = x['fp_xyz'][-1].clone()
        rcnn_feats = {'features': backbone_feats, 'points': backbone_xyz}
        '''
        if self.with_rpn:
            bbox_preds = self.rpn_head(x)
            rpn_loss = self.rpn_head.loss(
                bbox_preds=bbox_preds,
                points=points,
                gt_bboxes_3d=gt_bboxes_3d,
                gt_labels_3d=gt_labels_3d,
                img_metas=img_metas,
                gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_loss)
        '''
            sem_scores = F.sigmoid(bbox_preds['obj_scores']).transpose(
                1, 2).detach()
            obj_scores = sem_scores.max(-1)[0]
            is_training = True
            bbox_list = self.rpn_head.get_bboxes(points_cat, bbox_preds,
                                                 img_metas, is_training)
            proposal_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            rcnn_feats.update({'points_scores': obj_scores})

        roi_losses = self.roi_head.forward_train(rcnn_feats, img_metas,
                                                 proposal_list, gt_bboxes_3d,
                                                 gt_labels_3d)

        losses.update(roi_losses)
        '''
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
        # features for rcnn
        backbone_feats = x['fp_features'][-1].clone()
        backbone_xyz = x['fp_xyz'][-1].clone()
        rcnn_feats = {'features': backbone_feats, 'points': backbone_xyz}

        bbox_preds = self.rpn_head(x)
        sem_scores = F.sigmoid(bbox_preds['obj_scores']).detach()
        obj_scores = sem_scores.transpose(1, 2).max(-1)[0]
        rcnn_feats.update({'points_scores': obj_scores})

        bbox_list = self.rpn_head.get_bboxes(
            points_cat, bbox_preds, img_metas, rescale=rescale)
        from mmdet3d.core.bbox import bbox3d2result
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        '''
        proposal_list = [
            dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
            for bboxes, scores, labels in bbox_list
        ]
        bbox_results = self.roi_head.simple_test(rcnn_feats, img_metas,
                                                 proposal_list)
        '''
        return bbox_results
