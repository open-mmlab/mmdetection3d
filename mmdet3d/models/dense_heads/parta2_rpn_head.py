# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import numpy as np
import torch
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet3d.models.layers import nms_bev, nms_normal_bev
from mmdet3d.registry import MODELS
from mmdet3d.structures import limit_period, xywhr2xyxyr
from mmdet3d.utils.typing_utils import InstanceList
from ...structures.det3d_data_sample import SampleList
from .anchor3d_head import Anchor3DHead


@MODELS.register_module()
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
                 num_classes: int,
                 in_channels: int,
                 train_cfg: ConfigDict,
                 test_cfg: ConfigDict,
                 feat_channels: int = 256,
                 use_direction_classifier: bool = True,
                 anchor_generator: Dict = dict(
                     type='Anchor3DRangeGenerator',
                     range=[0, -39.68, -1.78, 69.12, 39.68, -1.78],
                     strides=[2],
                     sizes=[[3.9, 1.6, 1.56]],
                     rotations=[0, 1.57],
                     custom_values=[],
                     reshape_out=False),
                 assigner_per_size: bool = False,
                 assign_per_class: bool = False,
                 diff_rad_by_sin: bool = True,
                 dir_offset: float = -np.pi / 2,
                 dir_limit_offset: float = 0,
                 bbox_coder: Dict = dict(type='DeltaXYZWLHRBBoxCoder'),
                 loss_cls: Dict = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox: Dict = dict(
                     type='mmdet.SmoothL1Loss',
                     beta=1.0 / 9.0,
                     loss_weight=2.0),
                 loss_dir: Dict = dict(
                     type='mmdet.CrossEntropyLoss', loss_weight=0.2),
                 init_cfg: Dict = None) -> None:
        super().__init__(num_classes, in_channels, feat_channels,
                         use_direction_classifier, anchor_generator,
                         assigner_per_size, assign_per_class, diff_rad_by_sin,
                         dir_offset, dir_limit_offset, bbox_coder, loss_cls,
                         loss_bbox, loss_dir, train_cfg, test_cfg, init_cfg)

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                dir_cls_pred_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                input_meta: List[dict],
                                cfg: ConfigDict,
                                rescale: List[Tensor] = False):
        """Get bboxes of single branch.

        Args:
            cls_score_list (torch.Tensor): Class score in single batch.
            bbox_pred_list (torch.Tensor): Bbox prediction in single batch.
            dir_cls_pred_list (torch.Tensor): Predictions of direction class
                in single batch.
            mlvl_priors (List[torch.Tensor]): Multi-level anchors
                in single batch.
            input_meta (list[dict]): Contain pcd and img's meta info.
            cfg (:obj:`ConfigDict`): Training or testing config.
            rescale (list[torch.Tensor]): whether th rescale bbox.

        Returns:
            dict: Predictions of single batch containing the following keys:

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Predicted 3d bboxes.
            - scores_3d (torch.Tensor): Score of each bbox.
            - labels_3d (torch.Tensor): Label of each bbox.
            - cls_preds (torch.Tensor): Class score of each bbox.
        """
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_priors)
        mlvl_bboxes = []
        mlvl_max_scores = []
        mlvl_label_pred = []
        mlvl_dir_scores = []
        mlvl_cls_score = []
        for cls_score, bbox_pred, dir_cls_pred, anchors in zip(
                cls_score_list, bbox_pred_list, dir_cls_pred_list,
                mlvl_priors):
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
        # because the bbox head in the second stage does not have
        # classification branch,
        # roi head need this score as classification score
        mlvl_cls_score = torch.cat(mlvl_cls_score)

        score_thr = cfg.get('score_thr', 0)
        result = self.class_agnostic_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
                                         mlvl_max_scores, mlvl_label_pred,
                                         mlvl_cls_score, mlvl_dir_scores,
                                         score_thr, cfg, input_meta)
        return result

    def loss_and_predict(self,
                         feats_dict: Dict,
                         batch_data_samples: SampleList,
                         proposal_cfg: ConfigDict = None,
                         **kwargs) -> Tuple[dict, InstanceList]:
        """Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples.

        Args:
            feats_dict (dict): Contains features from the first stage.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.
            proposal_cfg (ConfigDict, optional): Proposal config.

        Returns:
            tuple: the return value is a tuple contains:

            - losses: (dict[str, Tensor]): A dictionary of loss components.
            - predictions (list[:obj:`InstanceData`]): Detection
              results of each sample after the post process.
        """
        batch_gt_instances_3d = []
        batch_gt_instances_ignore = []
        batch_input_metas = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
            batch_gt_instances_ignore.append(
                data_sample.get('ignored_instances', None))

        outs = self(feats_dict['neck_feats'])

        loss_inputs = outs + (batch_gt_instances_3d, batch_input_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)

        predictions = self.predict_by_feat(
            *outs, batch_input_metas=batch_input_metas, cfg=proposal_cfg)
        return losses, predictions

    def loss_by_feat(self,
                     cls_scores: List[Tensor],
                     bbox_preds: List[Tensor],
                     dir_cls_preds: List[Tensor],
                     batch_gt_instances_3d: InstanceList,
                     batch_input_metas: List[dict],
                     batch_gt_instances_ignore: InstanceList = None) -> Dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[torch.Tensor]): Multi-level class scores.
            bbox_preds (list[torch.Tensor]): Multi-level bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Multi-level direction
                class predictions.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances. It usually includes ``bboxes_3d`` and
                ``labels_3d`` attributes.
            batch_input_metas (list[dict]): Contain pcd and img's meta info.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, list[torch.Tensor]]: Classification, bbox, and
                direction losses of each level.

            - loss_rpn_cls (list[torch.Tensor]): Classification losses.
            - loss_rpn_bbox (list[torch.Tensor]): Box regression losses.
            - loss_rpn_dir (list[torch.Tensor]): Direction classification
                losses.
        """
        loss_dict = super().loss_by_feat(cls_scores, bbox_preds, dir_cls_preds,
                                         batch_gt_instances_3d,
                                         batch_input_metas,
                                         batch_gt_instances_ignore)
        # change the loss key names to avoid conflict
        return dict(
            loss_rpn_cls=loss_dict['loss_cls'],
            loss_rpn_bbox=loss_dict['loss_bbox'],
            loss_rpn_dir=loss_dict['loss_dir'])

    def class_agnostic_nms(self, mlvl_bboxes: Tensor,
                           mlvl_bboxes_for_nms: Tensor,
                           mlvl_max_scores: Tensor, mlvl_label_pred: Tensor,
                           mlvl_cls_score: Tensor, mlvl_dir_scores: Tensor,
                           score_thr: int, cfg: ConfigDict,
                           input_meta: dict) -> Dict:
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
            cfg (:obj:`ConfigDict`): Training or testing config.
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
            nms_func = nms_bev
        else:
            nms_func = nms_normal_bev
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
            if bboxes.shape[0] > cfg.nms_post:
                _, inds = scores.sort(descending=True)
                inds = inds[:cfg.nms_post]
                bboxes = bboxes[inds, :]
                labels = labels[inds]
                scores = scores[inds]
                cls_scores = cls_scores[inds]
            bboxes = input_meta['box_type_3d'](
                bboxes, box_dim=self.box_code_size)
            result = InstanceData()
            result.bboxes_3d = bboxes
            result.scores_3d = scores
            result.labels_3d = labels
            result.cls_preds = cls_scores
            return result
        else:
            result = InstanceData()
            result.bboxes_3d = input_meta['box_type_3d'](
                mlvl_bboxes.new_zeros([0, self.box_code_size]),
                box_dim=self.box_code_size)
            result.scores_3d = mlvl_bboxes.new_zeros([0])
            result.labels_3d = mlvl_bboxes.new_zeros([0])
            result.cls_preds = mlvl_bboxes.new_zeros(
                [0, mlvl_cls_score.shape[-1]])
            return result

    def predict(self, feats_dict: Dict,
                batch_data_samples: SampleList) -> InstanceList:
        """Perform forward propagation of the 3D detection head and predict
        detection results on the features of the upstream network.

        Args:
            feats_dict (dict): Contains features from the first stage.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each sample
            after the post process.
            Each item usually contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (BaseInstance3DBoxes): Prediction of bboxes,
              contains a tensor with shape (num_instances, C), where
              C >= 7.
        """
        batch_input_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        rpn_outs = self(feats_dict['neck_feats'])
        proposal_cfg = self.test_cfg

        proposal_list = self.predict_by_feat(
            *rpn_outs, cfg=proposal_cfg, batch_input_metas=batch_input_metas)
        return proposal_list
