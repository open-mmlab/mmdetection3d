from __future__ import division

import numpy as np
import torch
from mmcv.runner import force_fp32

from mmdet3d.core import limit_period, xywhr2xyxyr
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu, nms_normal_gpu
from mmdet.models import HEADS
from .anchor3d_head import Anchor3DHead


@HEADS.register_module()
class PartA2RPNHead(Anchor3DHead):
    """RPN head for PartA2.

    Note:
        The main difference between the PartA2 RPN head and the Anchor3DHead
        lies in their output during inference. PartA2 RPN head further returns
        the original classification score for the second stage since the bbox
        head in RoI head does not do classification task.

        Different from RPN heads in 2D detectors, this RPN head does
        multi-class classification task and uses FocalLoss like the SECOND and
        PointPillars do. But this head uses class agnostic nms rather than
        multi-class nms.

    Args:
        num_classes (int): Number of classes.
        in_channels (int): Number of channels in the input feature map.
        train_cfg (dict): Train configs.
        test_cfg (dict): Test configs.
        feat_channels (int): Number of channels of the feature map.
        use_direction_classifier (bool): Whether to add a direction classifier.
        anchor_generator(dict): Config dict of anchor generator.
        assigner_per_size (bool): Whether to do assignment for each separate
            anchor size.
        assign_per_class (bool): Whether to do assignment for each class.
        diff_rad_by_sin (bool): Whether to change the difference into sin
            difference for box regression loss.
        dir_offset (float | int): The offset of BEV rotation angles
            (TODO: may be moved into box coder)
        dir_limit_offset (float | int): The limited range of BEV
            rotation angles. (TODO: may be moved into box coder)
        bbox_coder (dict): Config dict of box coders.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_dir (dict): Config of direction classifier loss.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 train_cfg,
                 test_cfg,
                 feat_channels=256,
                 use_direction_classifier=True,
                 anchor_generator=dict(
                     type='Anchor3DRangeGenerator',
                     range=[0, -39.68, -1.78, 69.12, 39.68, -1.78],
                     strides=[2],
                     sizes=[[1.6, 3.9, 1.56]],
                     rotations=[0, 1.57],
                     custom_values=[],
                     reshape_out=False),
                 assigner_per_size=False,
                 assign_per_class=False,
                 diff_rad_by_sin=True,
                 dir_offset=0,
                 dir_limit_offset=1,
                 bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
                 loss_dir=dict(type='CrossEntropyLoss', loss_weight=0.2),
                 init_cfg=None):
        super().__init__(num_classes, in_channels, train_cfg, test_cfg,
                         feat_channels, use_direction_classifier,
                         anchor_generator, assigner_per_size, assign_per_class,
                         diff_rad_by_sin, dir_offset, dir_limit_offset,
                         bbox_coder, loss_cls, loss_bbox, loss_dir, init_cfg)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'dir_cls_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             dir_cls_preds,
             gt_bboxes,
             gt_labels,
             input_metas,
             gt_bboxes_ignore=None):
        """Calculate losses.

        Args:
            cls_scores (list[torch.Tensor]): Multi-level class scores.
            bbox_preds (list[torch.Tensor]): Multi-level bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Multi-level direction
                class predictions.
            gt_bboxes (list[:obj:`BaseInstance3DBoxes`]): Ground truth boxes \
                of each sample.
            gt_labels (list[torch.Tensor]): Labels of each sample.
            input_metas (list[dict]): Point cloud and image's meta info.
            gt_bboxes_ignore (None | list[torch.Tensor]): Specify
                which bounding.

        Returns:
            dict[str, list[torch.Tensor]]: Classification, bbox, and \
                direction losses of each level.

                - loss_rpn_cls (list[torch.Tensor]): Classification losses.
                - loss_rpn_bbox (list[torch.Tensor]): Box regression losses.
                - loss_rpn_dir (list[torch.Tensor]): Direction classification \
                    losses.
        """
        loss_dict = super().loss(cls_scores, bbox_preds, dir_cls_preds,
                                 gt_bboxes, gt_labels, input_metas,
                                 gt_bboxes_ignore)
        # change the loss key names to avoid conflict
        return dict(
            loss_rpn_cls=loss_dict['loss_cls'],
            loss_rpn_bbox=loss_dict['loss_bbox'],
            loss_rpn_dir=loss_dict['loss_dir'])

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          dir_cls_preds,
                          mlvl_anchors,
                          input_meta,
                          cfg,
                          rescale=False):
        """Get bboxes of single branch.

        Args:
            cls_scores (torch.Tensor): Class score in single batch.
            bbox_preds (torch.Tensor): Bbox prediction in single batch.
            dir_cls_preds (torch.Tensor): Predictions of direction class
                in single batch.
            mlvl_anchors (List[torch.Tensor]): Multi-level anchors
                in single batch.
            input_meta (list[dict]): Contain pcd and img's meta info.
            cfg (None | :obj:`ConfigDict`): Training or testing config.
            rescale (list[torch.Tensor]): whether th rescale bbox.

        Returns:
            dict: Predictions of single batch containing the following keys:

                - boxes_3d (:obj:`BaseInstance3DBoxes`): Predicted 3d bboxes.
                - scores_3d (torch.Tensor): Score of each bbox.
                - labels_3d (torch.Tensor): Label of each bbox.
                - cls_preds (torch.Tensor): Class score of each bbox.
        """
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_max_scores = []
        mlvl_label_pred = []
        mlvl_dir_scores = []
        mlvl_cls_score = []
        for cls_score, bbox_pred, dir_cls_pred, anchors in zip(
                cls_scores, bbox_preds, dir_cls_preds, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            assert cls_score.size()[-2:] == dir_cls_pred.size()[-2:]
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]

            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.num_classes)

            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2,
                                          0).reshape(-1, self.box_code_size)

            nms_pre = cfg.get('nms_pre', -1)
            if self.use_sigmoid_cls:
                max_scores, pred_labels = scores.max(dim=1)
            else:
                max_scores, pred_labels = scores[:, :-1].max(dim=1)
            # get topk
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                topk_scores, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                max_scores = topk_scores
                cls_score = scores[topk_inds, :]
                dir_cls_score = dir_cls_score[topk_inds]
                pred_labels = pred_labels[topk_inds]

            bboxes = self.bbox_coder.decode(anchors, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_max_scores.append(max_scores)
            mlvl_cls_score.append(cls_score)
            mlvl_label_pred.append(pred_labels)
            mlvl_dir_scores.append(dir_cls_score)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_bboxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d'](
            mlvl_bboxes, box_dim=self.box_code_size).bev)
        mlvl_max_scores = torch.cat(mlvl_max_scores)
        mlvl_label_pred = torch.cat(mlvl_label_pred)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)
        # shape [k, num_class] before sigmoid
        # PartA2 need to keep raw classification score
        # becase the bbox head in the second stage does not have
        # classification branch,
        # roi head need this score as classification score
        mlvl_cls_score = torch.cat(mlvl_cls_score)

        score_thr = cfg.get('score_thr', 0)
        result = self.class_agnostic_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
                                         mlvl_max_scores, mlvl_label_pred,
                                         mlvl_cls_score, mlvl_dir_scores,
                                         score_thr, cfg.nms_post, cfg,
                                         input_meta)

        return result

    def class_agnostic_nms(self, mlvl_bboxes, mlvl_bboxes_for_nms,
                           mlvl_max_scores, mlvl_label_pred, mlvl_cls_score,
                           mlvl_dir_scores, score_thr, max_num, cfg,
                           input_meta):
        """Class agnostic nms for single batch.

        Args:
            mlvl_bboxes (torch.Tensor): Bboxes from Multi-level.
            mlvl_bboxes_for_nms (torch.Tensor): Bboxes for nms
                (bev or minmax boxes) from Multi-level.
            mlvl_max_scores (torch.Tensor): Max scores of Multi-level bbox.
            mlvl_label_pred (torch.Tensor): Class predictions
                of Multi-level bbox.
            mlvl_cls_score (torch.Tensor): Class scores of
                Multi-level bbox.
            mlvl_dir_scores (torch.Tensor): Direction scores of
                Multi-level bbox.
            score_thr (int): Score threshold.
            max_num (int): Max number of bboxes after nms.
            cfg (None | :obj:`ConfigDict`): Training or testing config.
            input_meta (dict): Contain pcd and img's meta info.

        Returns:
            dict: Predictions of single batch. Contain the keys:

                - boxes_3d (:obj:`BaseInstance3DBoxes`): Predicted 3d bboxes.
                - scores_3d (torch.Tensor): Score of each bbox.
                - labels_3d (torch.Tensor): Label of each bbox.
                - cls_preds (torch.Tensor): Class score of each bbox.
        """
        bboxes = []
        scores = []
        labels = []
        dir_scores = []
        cls_scores = []
        score_thr_inds = mlvl_max_scores > score_thr
        _scores = mlvl_max_scores[score_thr_inds]
        _bboxes_for_nms = mlvl_bboxes_for_nms[score_thr_inds, :]
        if cfg.use_rotate_nms:
            nms_func = nms_gpu
        else:
            nms_func = nms_normal_gpu
        selected = nms_func(_bboxes_for_nms, _scores, cfg.nms_thr)

        _mlvl_bboxes = mlvl_bboxes[score_thr_inds, :]
        _mlvl_dir_scores = mlvl_dir_scores[score_thr_inds]
        _mlvl_label_pred = mlvl_label_pred[score_thr_inds]
        _mlvl_cls_score = mlvl_cls_score[score_thr_inds]

        if len(selected) > 0:
            bboxes.append(_mlvl_bboxes[selected])
            scores.append(_scores[selected])
            labels.append(_mlvl_label_pred[selected])
            cls_scores.append(_mlvl_cls_score[selected])
            dir_scores.append(_mlvl_dir_scores[selected])
            dir_rot = limit_period(bboxes[-1][..., 6] - self.dir_offset,
                                   self.dir_limit_offset, np.pi)
            bboxes[-1][..., 6] = (
                dir_rot + self.dir_offset +
                np.pi * dir_scores[-1].to(bboxes[-1].dtype))

        if bboxes:
            bboxes = torch.cat(bboxes, dim=0)
            scores = torch.cat(scores, dim=0)
            cls_scores = torch.cat(cls_scores, dim=0)
            labels = torch.cat(labels, dim=0)
            dir_scores = torch.cat(dir_scores, dim=0)
            if bboxes.shape[0] > max_num:
                _, inds = scores.sort(descending=True)
                inds = inds[:max_num]
                bboxes = bboxes[inds, :]
                labels = labels[inds]
                scores = scores[inds]
                cls_scores = cls_scores[inds]
            bboxes = input_meta['box_type_3d'](
                bboxes, box_dim=self.box_code_size)
            return dict(
                boxes_3d=bboxes,
                scores_3d=scores,
                labels_3d=labels,
                cls_preds=cls_scores  # raw scores [max_num, cls_num]
            )
        else:
            return dict(
                boxes_3d=input_meta['box_type_3d'](
                    mlvl_bboxes.new_zeros([0, self.box_code_size]),
                    box_dim=self.box_code_size),
                scores_3d=mlvl_bboxes.new_zeros([0]),
                labels_3d=mlvl_bboxes.new_zeros([0]),
                cls_preds=mlvl_bboxes.new_zeros([0, mlvl_cls_score.shape[-1]]))
