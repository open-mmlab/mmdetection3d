# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.runner import BaseModule
from torch import nn as nn

from mmdet3d.core.bbox.structures import rotation_3d_in_axis
from mmdet3d.models.builder import build_loss
from mmdet.core import multi_apply
from mmdet.models import HEADS


@HEADS.register_module()
class CenterPointBBoxHead(BaseModule):
    """Box head of the second stage CenterPoint.

    Args:
        input_channels (int): The number of input channels.
        shared_fc (list[int]): Output channels of the shared head.
        cls_fc (list[int]): Output channels of the classification head.
        reg_fc (list[int]): Output channels of the regression head.
        dp_ratio (float): Ratio of Dropout.
        code_size (int): Dimension of the encoded bbox.
        num_classes (int): The number of classes.
        loss_reg (dict): Config dict of regression loss.
        loss_cls (dict): Config dict of classification loss.
        init_cfg (dict): Initialization config dict.
    """

    def __init__(self,
                 input_channels=128 * 3 * 5,
                 shared_fc=[256, 256],
                 cls_fc=[256, 256],
                 reg_fc=[256, 256],
                 dp_ratio=0.3,
                 code_size=7,
                 num_classes=1,
                 loss_reg=dict(
                     type='L1Loss', reduction='none', loss_weight=1.0),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     reduction='none',
                     loss_weight=1.0),
                 init_cfg=None):
        super(CenterPointBBoxHead, self).__init__(init_cfg=init_cfg)

        self.input_channels = input_channels
        pre_channel = input_channels

        self.loss_reg = build_loss(loss_reg)
        self.loss_cls = build_loss(loss_cls)

        shared_fc_list = []
        for i in range(len(shared_fc)):
            shared_fc_list.extend([
                nn.Linear(pre_channel, shared_fc[i], bias=False),
                nn.BatchNorm1d(shared_fc[i]),
                nn.ReLU()
            ])
            pre_channel = shared_fc[i]

            if i != len(shared_fc) - 1 and dp_ratio > 0:
                shared_fc_list.append(nn.Dropout(dp_ratio))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.cls_layers = self.make_fc_layers(pre_channel, num_classes, cls_fc,
                                              dp_ratio)
        self.reg_layers = self.make_fc_layers(pre_channel, code_size, reg_fc,
                                              dp_ratio)

        self.num_classes = num_classes
        self.code_size = code_size

    def forward(self, roi_features):
        """Forward function for CenterPointBBoxHead.

        Args:
            roi_features (list[torch.Tensor]): Extracted features in roi.
            The shape of each roi feature is [N, 5*C].

        Return:
            list[dict[str, torch.Tensor]]: Contains predictions of bbox
                refinement head and classification head. The outer list
                indicates the prediction result in each batch.
                - cls (torch.Tensor): [N, num_class]
                - reg (torch.Tensor): [N, code_size]
        """
        pred_res = []

        for i in range(len(roi_features)):
            pred_res_batch = dict()
            shared_features = self.shared_fc_layer(roi_features[i])
            cls = self.cls_layers(shared_features)
            reg = self.reg_layers(shared_features)
            pred_res_batch.update(cls=cls)
            pred_res_batch.update(reg=reg)
            pred_res.append(pred_res_batch)

        return pred_res

    def loss(self, roi_features_sampled, sample_results, cfg):
        """Loss function for CenterPoint bbox head.

        Args:
            roi_features_sampled list[torch.Tensor]: roi features after
                sampling with shape of [N, 5*C]. cat([pos_box, neg_box])
            sample_results list[(:obj:`SamplingResult`)]: Sampled results used
                for training.
            cfg (:obj:`ConfigDict`): Training config.

        Returns:
            dict: Losses from CenterPointBBoxHead.
        """
        losses = dict()

        pred_res = self(roi_features_sampled)

        label, bbox_targets, pos_gt_bboxes, reg_mask, label_weights, \
            bbox_weights = self.get_targets(sample_results, cfg)

        cls_pred = torch.cat([pred_batch['cls'] for pred_batch in pred_res],
                             dim=0)
        label = torch.cat(list(label), dim=0).reshape(-1, self.num_classes)
        label_weights = torch.cat(list(label_weights), dim=0)
        loss_cls = self.loss_cls(cls_pred, label, label_weights)
        losses.update(loss_cls=loss_cls)

        bbox_pred = torch.cat([pred_batch['reg'] for pred_batch in pred_res],
                              dim=0)
        bbox_targets = torch.cat(
            list(bbox_targets), dim=0).reshape(-1, self.code_size)
        bbox_weights = torch.cat(list(bbox_weights), dim=0).reshape(-1, 1)
        loss_reg = self.loss_reg(bbox_pred, bbox_targets, bbox_weights)
        losses.update(loss_reg=loss_reg)

        return losses

    def get_bboxes(self, roi_features, img_metas, rois):
        """Generate bboxes from one-stage rois and bbox head predictions.
        Args:
            roi_features list[torch.Tensor]: Extracted features. The shape
                of each roi feature is [N, 5*C]
            imput_metas (list[dict]): Meta info of each input.
            rois (list[list[bboxes, scores, labels]]): Decoded bbox, scores
                 and labels.
                - bboxes (:obj:`BaseInstance3DBoxes`): Prediction bboxes
                    after nms
                - scores (torch.Tensor): Prediction scores after nms with the
                    shape of [N].
                - labels (torch.Tensor): Prediction labels after nms with the
                    shape of [N].

        Returns:
            list[list[boxes, scores, labels]]: final predicted bboxes
                    bboxes (:obj:`BaseInstance3DBoxes`): finale bboxes
                    scores (torch.Tensor): finale scores
                    labels (torch.Tensor): finale labels
        """

        pred_res = self(roi_features)

        res_lists = []
        batch_size = len(roi_features)
        for batch_idx in range(batch_size):
            # - calculate score
            bbox_head = pred_res[batch_idx]
            cls = bbox_head['cls']
            assert cls.shape[-1] == 1  # NOTE: ONLY surpport class agnostic now
            scores = torch.sqrt(
                torch.sigmoid(cls).reshape(-1) * rois[batch_idx][1])

            # box refinement
            reg = bbox_head['reg']
            dxyz = reg[:, :3].unsqueeze(1)  # [N, 3] -> [N, 1, 3]
            dxyz = rotation_3d_in_axis(
                dxyz, rois[batch_idx][0].yaw, axis=2).squeeze(1)
            reg[:, :3] = dxyz

            bboxes = reg + rois[batch_idx][
                0].tensor[:, :7]  # TODO: consider velocity

            res_lists.append([bboxes, scores, rois[batch_idx][2]])
        return res_lists

    def make_fc_layers(self, input_channels, output_channels, fc_list,
                       dp_ratio):
        fc_layers = []
        pre_channel = input_channels
        for i in range(0, len(fc_list)):
            fc_layers.extend([
                nn.Linear(pre_channel, fc_list[i], bias=False),
                nn.BatchNorm1d(fc_list[i]),
                nn.ReLU()
            ])
            pre_channel = fc_list[i]
            if dp_ratio >= 0 and i == 0:
                fc_layers.append(nn.Dropout(dp_ratio))
        fc_layers.append(nn.Linear(pre_channel, output_channels, bias=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers

    def get_targets(self, sampling_results, train_cfg):
        """Generate targets.

        Args:
            sampling_results (list[:obj:`SamplingResult`]):
                Sampled results from rois.
            train_cfg (:obj:`ConfigDict`): Training config.

        Returns:
            tuple[tuple[torch.Tensor]]: Targets of boxes and class prediction.
                The outer tuple indicates the different predictions of targets.
                The inner tuple indicates the targets' attributes in one batch.
        """
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        iou_list = [res.iou for res in sampling_results]
        targets = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            pos_gt_bboxes_list,
            iou_list,
            cfg=train_cfg)
        (label, bbox_targets, pos_gt_bboxes, reg_mask, label_weights,
         bbox_weights) = targets

        return (label, bbox_targets, pos_gt_bboxes, reg_mask, label_weights,
                bbox_weights)

    def _get_target_single(self, pos_bboxes, pos_gt_bboxes, ious, cfg):
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
        label[interval_mask] = (ious[interval_mask] - cfg.cls_neg_thr) / \
            (cfg.cls_pos_thr - cfg.cls_neg_thr)
        # label weights
        label_weights = (label >= 0).float()
        # box regression target
        reg_mask = ious > cfg.reg_pos_thr
        bbox_weights = (reg_mask > 0).float()
        if reg_mask.bool().any():
            pos_gt_bboxes_ct = pos_gt_bboxes.clone().detach()
            roi_ry = pos_bboxes[..., 6] % (2 * np.pi)

            # canonical transformation
            pos_gt_bboxes_ct[..., 0:6] -= pos_bboxes[..., 0:6]
            pos_gt_bboxes_ct[..., 6] -= roi_ry
            pos_gt_bboxes_ct[..., 0:3] = rotation_3d_in_axis(
                pos_gt_bboxes_ct[..., 0:3].unsqueeze(1), -(roi_ry),
                axis=2).squeeze(1)

            # flip orientation if gt have opposite orientation
            ry_label = pos_gt_bboxes_ct[..., 6] % (2 * np.pi)  # 0 ~ 2pi
            is_opposite = (ry_label > np.pi * 0.5) & (ry_label < np.pi * 1.5)
            ry_label[is_opposite] = (ry_label[is_opposite] + np.pi) % (
                2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
            flag = ry_label > np.pi
            ry_label[flag] = ry_label[flag] - np.pi * 2  # (-pi/2, pi/2)
            ry_label = torch.clamp(ry_label, min=-np.pi / 2, max=np.pi / 2)
            pos_gt_bboxes_ct[..., 6] = ry_label

            rois_anchor = pos_bboxes.clone().detach()
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            # Directly encode as (dx, dy, dz, dw, dl, dh, dtheta)
            bbox_targets = pos_gt_bboxes_ct
        else:
            # no fg bbox
            bbox_targets = pos_gt_bboxes.new_empty((0, 7))

        return (label, bbox_targets, pos_gt_bboxes, reg_mask, label_weights,
                bbox_weights)
