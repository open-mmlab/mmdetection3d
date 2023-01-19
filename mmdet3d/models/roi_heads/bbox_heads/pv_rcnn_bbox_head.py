# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from mmcv.cnn import ConvModule
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.models.utils import multi_apply
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import nn as nn

from mmdet3d.models.layers import nms_bev, nms_normal_bev
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures.bbox_3d import (LiDARInstance3DBoxes,
                                        rotation_3d_in_axis, xywhr2xyxyr)
from mmdet3d.utils import InstanceList


@MODELS.register_module()
class PVRCNNBBoxHead(BaseModule):
    """PVRCNN BBox head.

    Args:
        in_channels (int): The number of input channel.
        grid_size (int): The number of grid points in roi bbox.
        num_classes (int): The number of classes.
        class_agnostic (bool): Whether generate class agnostic prediction.
            Defaults to True.
        shared_fc_channels (tuple(int)): Out channels of each shared fc layer.
            Defaults to (256, 256).
        cls_channels (tuple(int)): Out channels of each classification layer.
            Defaults to (256, 256).
        reg_channels (tuple(int)): Out channels of each regression layer.
            Defaults to (256, 256).
        dropout_ratio (float): Ratio of dropout layer. Defaults to 0.5.
        with_corner_loss (bool): Whether to use corner loss or not.
            Defaults to True.
        bbox_coder (:obj:`BaseBBoxCoder`): Bbox coder for box head.
            Defaults to dict(type='DeltaXYZWLHRBBoxCoder').
        norm_cfg (dict): Type of normalization method.
            Defaults to dict(type='BN1d', eps=1e-5, momentum=0.1)
        loss_bbox (dict): Config dict of box regression loss.
        loss_cls (dict): Config dict of classifacation loss.
        init_cfg (dict, optional): Initialize config of
            model.
    """

    def __init__(
        self,
        in_channels: int,
        grid_size: int,
        num_classes: int,
        class_agnostic: bool = True,
        shared_fc_channels: Tuple[int] = (256, 256),
        cls_channels: Tuple[int] = (256, 256),
        reg_channels: Tuple[int] = (256, 256),
        dropout_ratio: float = 0.3,
        with_corner_loss: bool = True,
        bbox_coder: dict = dict(type='DeltaXYZWLHRBBoxCoder'),
        norm_cfg: dict = dict(type='BN2d', eps=1e-5, momentum=0.1),
        loss_bbox: dict = dict(
            type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_cls: dict = dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='none',
            loss_weight=1.0),
        init_cfg: Optional[dict] = dict(
            type='Xavier', layer=['Conv2d', 'Conv1d'], distribution='uniform')
    ) -> None:
        super(PVRCNNBBoxHead, self).__init__(init_cfg=init_cfg)
        self.init_cfg = init_cfg
        self.num_classes = num_classes
        self.with_corner_loss = with_corner_loss
        self.class_agnostic = class_agnostic
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_cls = MODELS.build(loss_cls)
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)

        cls_out_channels = 1 if class_agnostic else num_classes
        self.reg_out_channels = self.bbox_coder.code_size * cls_out_channels
        if self.use_sigmoid_cls:
            self.cls_out_channels = cls_out_channels
        else:
            self.cls_out_channels = cls_out_channels + 1

        self.dropout_ratio = dropout_ratio
        self.grid_size = grid_size

        # PVRCNNBBoxHead model in_channels is num of grid points in roi box.
        in_channels *= (self.grid_size**3)

        self.in_channels = in_channels

        self.shared_fc_layer = self._make_fc_layers(
            in_channels, shared_fc_channels,
            range(len(shared_fc_channels) - 1), norm_cfg)
        self.cls_layer = self._make_fc_layers(
            shared_fc_channels[-1],
            cls_channels,
            range(1),
            norm_cfg,
            out_channels=self.cls_out_channels)
        self.reg_layer = self._make_fc_layers(
            shared_fc_channels[-1],
            reg_channels,
            range(1),
            norm_cfg,
            out_channels=self.reg_out_channels)

    def _make_fc_layers(self,
                        in_channels: int,
                        fc_channels: list,
                        dropout_indices: list,
                        norm_cfg: dict,
                        out_channels: Optional[int] = None) -> torch.nn.Module:
        """Initial a full connection layer.

        Args:
            in_channels (int): Module in channels.
            fc_channels (list): Full connection layer channels.
            dropout_indices (list): Dropout indices.
            norm_cfg (dict): Type of normalization method.
            out_channels (int, optional): Module out channels.
        """
        fc_layers = []
        pre_channel = in_channels
        for k in range(len(fc_channels)):
            fc_layers.append(
                ConvModule(
                    pre_channel,
                    fc_channels[k],
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    norm_cfg=norm_cfg,
                    conv_cfg=dict(type='Conv2d'),
                    bias=False,
                    inplace=True))
            pre_channel = fc_channels[k]
            if self.dropout_ratio >= 0 and k in dropout_indices:
                fc_layers.append(nn.Dropout(self.dropout_ratio))
        if out_channels is not None:
            fc_layers.append(
                nn.Conv2d(fc_channels[-1], out_channels, 1, bias=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers

    def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor]:
        """Forward pvrcnn bbox head.

        Args:
            feats (torch.Tensor): Batch point-wise features.

        Returns:
            tuple[torch.Tensor]: Score of class and bbox predictions.
        """
        # (B * N, 6, 6, 6, C)
        rcnn_batch_size = feats.shape[0]
        feats = feats.permute(0, 4, 1, 2,
                              3).contiguous().view(rcnn_batch_size, -1, 1, 1)
        # (BxN, C*6*6*6)
        shared_feats = self.shared_fc_layer(feats)
        cls_score = self.cls_layer(shared_feats).transpose(
            1, 2).contiguous().view(-1, self.cls_out_channels)  # (B, 1)
        bbox_pred = self.reg_layer(shared_feats).transpose(
            1, 2).contiguous().view(-1, self.reg_out_channels)  # (B, C)
        return cls_score, bbox_pred

    def loss(self, cls_score: torch.Tensor, bbox_pred: torch.Tensor,
             rois: torch.Tensor, labels: torch.Tensor,
             bbox_targets: torch.Tensor, pos_gt_bboxes: torch.Tensor,
             reg_mask: torch.Tensor, label_weights: torch.Tensor,
             bbox_weights: torch.Tensor) -> Dict:
        """Coumputing losses.

        Args:
            cls_score (torch.Tensor): Scores of each roi.
            bbox_pred (torch.Tensor): Predictions of bboxes.
            rois (torch.Tensor): Roi bboxes.
            labels (torch.Tensor): Labels of class.
            bbox_targets (torch.Tensor): Target of positive bboxes.
            pos_gt_bboxes (torch.Tensor): Ground truths of positive bboxes.
            reg_mask (torch.Tensor): Mask for positive bboxes.
            label_weights (torch.Tensor): Weights of class loss.
            bbox_weights (torch.Tensor): Weights of bbox loss.

        Returns:
             dict: Computed losses.

             - loss_cls (torch.Tensor): Loss of classes.
             - loss_bbox (torch.Tensor): Loss of bboxes.
             - loss_corner (torch.Tensor): Loss of corners.
        """
        losses = dict()
        rcnn_batch_size = cls_score.shape[0]

        # calculate class loss
        cls_flat = cls_score.view(-1)
        loss_cls = self.loss_cls(cls_flat, labels, label_weights)
        losses['loss_cls'] = loss_cls

        # calculate regression loss
        code_size = self.bbox_coder.code_size
        pos_inds = (reg_mask > 0)
        if pos_inds.any() == 0:
            # fake a part loss
            losses['loss_bbox'] = 0 * bbox_pred.sum()
            if self.with_corner_loss:
                losses['loss_corner'] = 0 * bbox_pred.sum()
        else:
            pos_bbox_pred = bbox_pred.view(rcnn_batch_size, -1)[pos_inds]
            bbox_weights_flat = bbox_weights[pos_inds].view(-1, 1).repeat(
                1, pos_bbox_pred.shape[-1])
            loss_bbox = self.loss_bbox(
                pos_bbox_pred.unsqueeze(dim=0), bbox_targets.unsqueeze(dim=0),
                bbox_weights_flat.unsqueeze(dim=0))
            losses['loss_bbox'] = loss_bbox

            if self.with_corner_loss:
                pos_roi_boxes3d = rois[..., 1:].view(-1, code_size)[pos_inds]
                pos_roi_boxes3d = pos_roi_boxes3d.view(-1, code_size)
                batch_anchors = pos_roi_boxes3d.clone().detach()
                pos_rois_rotation = pos_roi_boxes3d[..., 6].view(-1)
                roi_xyz = pos_roi_boxes3d[..., 0:3].view(-1, 3)
                batch_anchors[..., 0:3] = 0
                # decode boxes
                pred_boxes3d = self.bbox_coder.decode(
                    batch_anchors,
                    pos_bbox_pred.view(-1, code_size)).view(-1, code_size)

                pred_boxes3d[..., 0:3] = rotation_3d_in_axis(
                    pred_boxes3d[..., 0:3].unsqueeze(1),
                    pos_rois_rotation,
                    axis=2).squeeze(1)

                pred_boxes3d[:, 0:3] += roi_xyz

                # calculate corner loss
                loss_corner = self.get_corner_loss_lidar(
                    pred_boxes3d, pos_gt_bboxes)
                losses['loss_corner'] = loss_corner.mean()

        return losses

    def get_targets(self,
                    sampling_results: SamplingResult,
                    rcnn_train_cfg: dict,
                    concat: bool = True) -> Tuple[torch.Tensor]:
        """Generate targets.

        Args:
            sampling_results (list[:obj:`SamplingResult`]):
                Sampled results from rois.
            rcnn_train_cfg (:obj:`ConfigDict`): Training config of rcnn.
            concat (bool): Whether to concatenate targets between batches.

        Returns:
            tuple[torch.Tensor]: Targets of boxes and class prediction.
        """
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        iou_list = [res.iou for res in sampling_results]
        targets = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            pos_gt_bboxes_list,
            iou_list,
            cfg=rcnn_train_cfg)

        (label, bbox_targets, pos_gt_bboxes, reg_mask, label_weights,
         bbox_weights) = targets

        if concat:
            label = torch.cat(label, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            pos_gt_bboxes = torch.cat(pos_gt_bboxes, 0)
            reg_mask = torch.cat(reg_mask, 0)

            label_weights = torch.cat(label_weights, 0)
            label_weights /= torch.clamp(label_weights.sum(), min=1.0)

            bbox_weights = torch.cat(bbox_weights, 0)
            bbox_weights /= torch.clamp(bbox_weights.sum(), min=1.0)

        return (label, bbox_targets, pos_gt_bboxes, reg_mask, label_weights,
                bbox_weights)

    def _get_target_single(self, pos_bboxes: torch.Tensor,
                           pos_gt_bboxes: torch.Tensor, ious: torch.Tensor,
                           cfg: dict) -> Tuple[torch.Tensor]:
        """Generate training targets for a single sample.

        Args:
            pos_bboxes (torch.Tensor): Positive boxes with shape
                (N, 7).
            pos_gt_bboxes (torch.Tensor): Ground truth boxes with shape
                (M, 7).
            ious (torch.Tensor): IoU between `pos_bboxes` and `pos_gt_bboxes`
                in shape (N, M).
            cfg (dict): Training configs.

        Returns:
            tuple[torch.Tensor]: Target for positive boxes.
                (label, bbox_targets, pos_gt_bboxes, reg_mask, label_weights,
                bbox_weights)
        """
        cls_pos_mask = ious > cfg.cls_pos_thr
        cls_neg_mask = ious < cfg.cls_neg_thr
        interval_mask = (cls_pos_mask == 0) & (cls_neg_mask == 0)

        # iou regression target
        label = (cls_pos_mask > 0).float()
        label[interval_mask] = ious[interval_mask] * 2 - 0.5
        # label weights
        label_weights = (label >= 0).float()

        # box regression target
        reg_mask = pos_bboxes.new_zeros(ious.size(0)).long()
        reg_mask[0:pos_gt_bboxes.size(0)] = 1
        bbox_weights = (reg_mask > 0).float()
        if reg_mask.bool().any():
            pos_gt_bboxes_ct = pos_gt_bboxes.clone().detach()
            roi_center = pos_bboxes[..., 0:3]
            roi_ry = pos_bboxes[..., 6] % (2 * np.pi)

            # canonical transformation
            pos_gt_bboxes_ct[..., 0:3] -= roi_center
            pos_gt_bboxes_ct[..., 6] -= roi_ry
            pos_gt_bboxes_ct[..., 0:3] = rotation_3d_in_axis(
                pos_gt_bboxes_ct[..., 0:3].unsqueeze(1), -roi_ry,
                axis=2).squeeze(1)

            # flip orientation if rois have opposite orientation
            ry_label = pos_gt_bboxes_ct[..., 6] % (2 * np.pi)  # 0 ~ 2pi
            opposite_flag = (ry_label > np.pi * 0.5) & (ry_label < np.pi * 1.5)
            ry_label[opposite_flag] = (ry_label[opposite_flag] + np.pi) % (
                2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
            flag = ry_label > np.pi
            ry_label[flag] = ry_label[flag] - np.pi * 2  # (-pi/2, pi/2)
            ry_label = torch.clamp(ry_label, min=-np.pi / 2, max=np.pi / 2)
            pos_gt_bboxes_ct[..., 6] = ry_label

            rois_anchor = pos_bboxes.clone().detach()
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            bbox_targets = self.bbox_coder.encode(rois_anchor,
                                                  pos_gt_bboxes_ct)
        else:
            # no fg bbox
            bbox_targets = pos_gt_bboxes.new_empty((0, 7))

        return (label, bbox_targets, pos_gt_bboxes, reg_mask, label_weights,
                bbox_weights)

    def get_corner_loss_lidar(self,
                              pred_bbox3d: torch.Tensor,
                              gt_bbox3d: torch.Tensor,
                              delta: float = 1.0) -> torch.Tensor:
        """Calculate corner loss of given boxes.

        Args:
            pred_bbox3d (torch.FloatTensor): Predicted boxes in shape (N, 7).
            gt_bbox3d (torch.FloatTensor): Ground truth boxes in shape (N, 7).
            delta (float, optional): huber loss threshold. Defaults to 1.0

        Returns:
            torch.FloatTensor: Calculated corner loss in shape (N).
        """
        assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

        # This is a little bit hack here because we assume the box for
        # Part-A2 is in LiDAR coordinates
        gt_boxes_structure = LiDARInstance3DBoxes(gt_bbox3d)
        pred_box_corners = LiDARInstance3DBoxes(pred_bbox3d).corners
        gt_box_corners = gt_boxes_structure.corners

        # This flip only changes the heading direction of GT boxes
        gt_bbox3d_flip = gt_boxes_structure.clone()
        gt_bbox3d_flip.tensor[:, 6] += np.pi
        gt_box_corners_flip = gt_bbox3d_flip.corners

        corner_dist = torch.min(
            torch.norm(pred_box_corners - gt_box_corners, dim=2),
            torch.norm(pred_box_corners - gt_box_corners_flip,
                       dim=2))  # (N, 8)
        # huber loss
        abs_error = torch.abs(corner_dist)
        corner_loss = torch.where(abs_error < delta,
                                  0.5 * abs_error**2 / delta,
                                  abs_error - 0.5 * delta)
        return corner_loss.mean(dim=1)

    def get_results(self,
                    rois: torch.Tensor,
                    cls_preds: torch.Tensor,
                    bbox_reg: torch.Tensor,
                    class_labels: torch.Tensor,
                    input_metas: List[dict],
                    test_cfg: dict = None) -> InstanceList:
        """Generate bboxes from bbox head predictions.

        Args:
            rois (torch.Tensor): Roi bounding boxes.
            cls_preds (torch.Tensor): Scores of bounding boxes.
            bbox_reg (torch.Tensor): Bounding boxes predictions
            class_labels (torch.Tensor): Label of classes
            input_metas (list[dict]): Point cloud meta info.
            test_cfg (:obj:`ConfigDict`): Testing config.

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
        roi_batch_id = rois[..., 0]
        roi_boxes = rois[..., 1:]  # boxes without batch id
        batch_size = int(roi_batch_id.max().item() + 1)

        # decode boxes
        roi_ry = roi_boxes[..., 6].view(-1)
        roi_xyz = roi_boxes[..., 0:3].view(-1, 3)
        local_roi_boxes = roi_boxes.clone().detach()
        local_roi_boxes[..., 0:3] = 0
        batch_box_preds = self.bbox_coder.decode(local_roi_boxes, bbox_reg)
        batch_box_preds[..., 0:3] = rotation_3d_in_axis(
            batch_box_preds[..., 0:3].unsqueeze(1), roi_ry, axis=2).squeeze(1)
        batch_box_preds[:, 0:3] += roi_xyz

        # post processing
        result_list = []
        for batch_id in range(batch_size):
            cur_cls_preds = cls_preds[roi_batch_id == batch_id]
            box_preds = batch_box_preds[roi_batch_id == batch_id]
            label_preds = class_labels[batch_id]

            cur_cls_preds = cur_cls_preds.sigmoid()
            cur_cls_preds, _ = torch.max(cur_cls_preds, dim=-1)
            selected = self.class_agnostic_nms(
                scores=cur_cls_preds,
                bbox_preds=box_preds,
                input_meta=input_metas[batch_id],
                nms_cfg=test_cfg)

            selected_bboxes = box_preds[selected]
            selected_label_preds = label_preds[selected]
            selected_scores = cur_cls_preds[selected]

            results = InstanceData()
            results.bboxes_3d = input_metas[batch_id]['box_type_3d'](
                selected_bboxes, self.bbox_coder.code_size)
            results.scores_3d = selected_scores
            results.labels_3d = selected_label_preds

            result_list.append(results)
        return result_list

    def class_agnostic_nms(self, scores: torch.Tensor,
                           bbox_preds: torch.Tensor, nms_cfg: dict,
                           input_meta: dict) -> Tuple[torch.Tensor]:
        """Class agnostic NMS for box head.

        Args:
            scores (torch.Tensor): Object score of bounding boxes.
            bbox_preds (torch.Tensor): Predicted bounding boxes.
            nms_cfg (dict): NMS config dict.
            input_meta (dict): Contain pcd and img's meta info.

        Returns:
            tuple[torch.Tensor]: Bounding boxes, scores and labels.
        """
        obj_scores = scores.clone()
        if nms_cfg.use_rotate_nms:
            nms_func = nms_bev
        else:
            nms_func = nms_normal_bev

        bbox = input_meta['box_type_3d'](
            bbox_preds.clone(),
            box_dim=bbox_preds.shape[-1],
            with_yaw=True,
            origin=(0.5, 0.5, 0.5))

        if nms_cfg.score_thr is not None:
            scores_mask = (obj_scores >= nms_cfg.score_thr)
            obj_scores = obj_scores[scores_mask]
            bbox = bbox[scores_mask]
        selected = []
        if obj_scores.shape[0] > 0:
            box_scores_nms, indices = torch.topk(
                obj_scores, k=min(4096, obj_scores.shape[0]))
            bbox_bev = bbox.bev[indices]
            bbox_for_nms = xywhr2xyxyr(bbox_bev)

            keep = nms_func(bbox_for_nms, box_scores_nms, nms_cfg.nms_thr)
            selected = indices[keep]
        if nms_cfg.score_thr is not None:
            original_idxs = scores_mask.nonzero().view(-1)
            selected = original_idxs[selected]
        return selected
