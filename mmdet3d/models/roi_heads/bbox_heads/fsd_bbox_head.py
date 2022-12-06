# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmengine.model import BaseModule
from torch import nn as nn
import torch.nn.functional as F

from mmdet3d.structures import LiDARInstance3DBoxes, rotation_3d_in_axis
from mmdet3d.structures import xywhr2xyxyr
from mmdet3d.models.builder import build_loss
from mmdet3d.models.layers.sst import build_mlp

from mmdet3d.structures.ops.iou3d_calculator import nms_gpu, nms_normal_gpu
from mmdet3d.models.task_modules.builder import build_bbox_coder
from mmdet.models.utils import multi_apply
from mmdet.utils import reduce_mean

from mmdet3d.models import builder
from mmdet3d.registry import MODELS


@MODELS.register_module()
class FullySparseBboxHead(BaseModule):

    def __init__(self,
                 num_classes,
                 num_blocks,
                 in_channels, 
                 feat_channels,
                 rel_mlp_hidden_dims,
                 rel_mlp_in_channels,
                 reg_mlp,
                 cls_mlp,
                 with_rel_mlp=True,
                 with_cluster_center=False,
                 with_distance=False,
                 mode='max',
                 xyz_normalizer=[20, 20, 4],
                 act='gelu',
                 geo_input=True,
                 with_corner_loss=False,
                 bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
                 norm_cfg=dict(type='LN', eps=1e-3, momentum=0.01),
                 corner_loss_weight=1.0,
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='none',
                     loss_weight=1.0),
                 dropout=0,
                 cls_dropout=0,
                 reg_dropout=0,
                 unique_once=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.with_corner_loss = with_corner_loss
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.box_code_size = self.bbox_coder.code_size
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_cls = build_loss(loss_cls)
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.geo_input = geo_input
        self.corner_loss_weight = corner_loss_weight

        self.num_blocks = num_blocks
        self.print_info = {}
        self.unique_once = unique_once
        
        block_list = []
        for i in range(num_blocks):
            return_point_feats = i != num_blocks-1
            kwargs = dict(
                type='SIRLayer',
                in_channels=in_channels[i],
                feat_channels=feat_channels[i],
                with_distance=with_distance,
                with_cluster_center=with_cluster_center,
                with_rel_mlp=with_rel_mlp,
                rel_mlp_hidden_dims=rel_mlp_hidden_dims[i],
                rel_mlp_in_channel=rel_mlp_in_channels[i],
                with_voxel_center=False,
                voxel_size=[0.1, 0.1, 0.1], # not used, placeholder
                point_cloud_range=[-74.88, -74.88, -2, 74.88, 74.88, 4], # not used, placeholder
                norm_cfg=norm_cfg,
                mode=mode,
                fusion_layer=None,
                return_point_feats=return_point_feats,
                return_inv=False,
                rel_dist_scaler=10.0,
                xyz_normalizer=xyz_normalizer,
                act=act,
                dropout=dropout,
            )
            encoder = builder.build_voxel_encoder(kwargs)
            block_list.append(encoder)
        self.block_list = nn.ModuleList(block_list)

        end_channel = 0
        for c in feat_channels:
            end_channel += sum(c)

        if cls_mlp is not None:
            self.conv_cls = build_mlp(end_channel, cls_mlp + [1,], norm_cfg, True, act=act, dropout=cls_dropout)
        else:
            self.conv_cls = nn.Linear(end_channel, 1)

        if reg_mlp is not None:
            self.conv_reg = build_mlp(end_channel, reg_mlp + [self.box_code_size,], norm_cfg, True, act=act, dropout=reg_dropout)
        else:
            self.conv_reg = nn.Linear(end_channel, self.box_code_size)


    def init_weights(self):
        super().init_weights()

    # @force_fp32(apply_to=('pts_features', 'rois'))
    def forward(self, pts_xyz, pts_features, pts_info, roi_inds, rois):
        """Forward pass.

        Args:
            seg_feats (torch.Tensor): Point-wise semantic features.
            part_feats (torch.Tensor): Point-wise part prediction features.

        Returns:
            tuple[torch.Tensor]: Score of class and bbox predictions.
        """
        assert pts_features.size(0) > 0

        rois_batch_idx = rois[:, 0]
        rois = rois[:, 1:]
        roi_centers = rois[:, :3]
        rel_xyz = pts_xyz[:, :3] - roi_centers[roi_inds] 

        if self.unique_once:
            new_coors, unq_inv = torch.unique(roi_inds, return_inverse=True, return_counts=False, dim=0)
        else:
            new_coors = unq_inv = None


        out_feats = pts_features
        f_cluster = torch.cat([pts_info['local_xyz'], pts_info['boundary_offset'], pts_info['is_in_margin'][:, None], rel_xyz], dim=-1)

        cluster_feat_list = []
        for i, block in enumerate(self.block_list):

            in_feats = torch.cat([pts_xyz, out_feats], 1)

            if self.geo_input:
                in_feats = torch.cat([in_feats, f_cluster / 10], 1)

            if i < self.num_blocks - 1:
                # return point features
                out_feats, out_cluster_feats = block(in_feats, roi_inds, f_cluster, unq_inv_once=unq_inv, new_coors_once=new_coors)
                cluster_feat_list.append(out_cluster_feats)
            if i == self.num_blocks - 1:
                # return group features
                out_cluster_feats, out_coors = block(in_feats, roi_inds, f_cluster, unq_inv_once=unq_inv, new_coors_once=new_coors)
                cluster_feat_list.append(out_cluster_feats)
            
        final_cluster_feats = torch.cat(cluster_feat_list, dim=1)

        if self.training and (out_coors == -1).any():
            assert out_coors[0].item() == -1, 'This should hold due to sorted=True in torch.unique'

        nonempty_roi_mask = self.get_nonempty_roi_mask(out_coors, len(rois))

        cls_score = self.conv_cls(final_cluster_feats)
        bbox_pred = self.conv_reg(final_cluster_feats)

        cls_score = self.align_roi_feature_and_rois(cls_score, out_coors, len(rois))
        bbox_pred = self.align_roi_feature_and_rois(bbox_pred, out_coors, len(rois))

        return cls_score, bbox_pred, nonempty_roi_mask
    
    def get_nonempty_roi_mask(self, out_coors, num_rois):
        if self.training:
            assert out_coors.max() + 1 <= num_rois
            assert out_coors.ndim == 1
            assert torch.unique(out_coors).size(0) == out_coors.size(0)
            assert (out_coors == torch.sort(out_coors)[0]).all()
        out_coors = out_coors[out_coors >= 0]
        nonempty_roi_mask = torch.zeros(num_rois, dtype=torch.bool, device=out_coors.device)
        nonempty_roi_mask[out_coors] = True
        return nonempty_roi_mask

    def align_roi_feature_and_rois(self, features, out_coors, num_rois):
        """
        1. The order of roi features obtained by dynamic pooling may not align with rois
        2. Even if we set sorted=True in torch.unique, the empty group (with idx -1) will be the first feature, causing misaligned
        So here we explicitly align them to make sure the sanity
        """
        new_feature = features.new_zeros((num_rois, features.size(1)))
        coors_mask = out_coors >= 0

        if not coors_mask.any():
            new_feature[:len(features), :] = features * 0 # pseudo gradient, avoid unused_parameters
            return new_feature

        nonempty_coors = out_coors[coors_mask]
        nonempty_feats = features[coors_mask]

        new_feature[nonempty_coors] = nonempty_feats

        return new_feature


    def loss(self, cls_score, bbox_pred, nonempty_roi_mask, rois, labels, bbox_targets, pos_batch_idx,
             pos_gt_bboxes, pos_gt_labels, reg_mask, label_weights, bbox_weights):
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
        num_total_samples = rcnn_batch_size = cls_score.shape[0]
        assert num_total_samples > 0

        # calculate class loss
        cls_flat = cls_score.view(-1) # only to classify foreground and background

        label_weights[~nonempty_roi_mask] = 0 # do not calculate cls loss for empty rois
        label_weights[nonempty_roi_mask] = 1 # we use avg_factor in loss_cls, so we need to set it to 1
        bbox_weights[...] = 1 # we use avg_factor in loss_bbox, so we need to set it to 1

        reg_mask[~nonempty_roi_mask] = 0 # do not calculate loss for empty rois

        cls_avg_factor = num_total_samples * 1.0
        if self.train_cfg.get('sync_cls_avg_factor', False):
            cls_avg_factor = reduce_mean(
                bbox_weights.new_tensor([cls_avg_factor]))

        loss_cls = self.loss_cls(cls_flat, labels, label_weights, avg_factor=cls_avg_factor)
        losses['loss_rcnn_cls'] = loss_cls

        # calculate regression loss
        pos_inds = (reg_mask > 0)
        losses['num_pos_rois'] = pos_inds.sum().float()
        losses['num_neg_rois'] = (reg_mask <= 0).sum().float()

        reg_avg_factor = pos_inds.sum().item()
        if self.train_cfg.get('sync_reg_avg_factor', False):
            reg_avg_factor = reduce_mean(
                bbox_weights.new_tensor([reg_avg_factor]))

        if pos_inds.any() == 0:
            # fake a bbox loss
            losses['loss_rcnn_bbox'] = bbox_pred.sum() * 0
            if self.with_corner_loss:
                losses['loss_rcnn_corner'] = bbox_pred.sum() * 0
        else:
            pos_bbox_pred = bbox_pred[pos_inds]
            # bbox_targets should have same size with pos_bbox_pred in normal case. But reg_mask is modified by nonempty_roi_mask. So it could be different.
            # filter bbox_targets per sample

            bbox_targets = self.filter_pos_assigned_but_empty_rois(bbox_targets, pos_batch_idx, pos_inds, rois[:, 0].int())

            assert not (pos_bbox_pred == -1).all(1).any()
            bbox_weights_flat = bbox_weights[pos_inds].view(-1, 1).repeat(1, pos_bbox_pred.shape[-1])


            if pos_bbox_pred.size(0) != bbox_targets.size(0):
                raise ValueError('Impossible after filtering bbox_targets')
                # I don't know why this happens
                losses['loss_rcnn_bbox'] = bbox_pred.sum() * 0
                if self.with_corner_loss:
                    losses['loss_rcnn_corner'] = bbox_pred.sum() * 0
                return losses

            assert bbox_targets.numel() > 0
            loss_bbox = self.loss_bbox(pos_bbox_pred, bbox_targets, bbox_weights_flat, avg_factor=reg_avg_factor)
            losses['loss_rcnn_bbox'] = loss_bbox

            if self.with_corner_loss:
                code_size = self.bbox_coder.code_size
                pos_roi_boxes3d = rois[..., 1:code_size + 1].view(-1, code_size)[pos_inds]
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
                    (pos_rois_rotation + np.pi / 2),
                    axis=2).squeeze(1)

                pred_boxes3d[:, 0:3] += roi_xyz

                # calculate corner loss
                assert pos_gt_bboxes.size(0) == pos_gt_labels.size(0)
                pos_gt_bboxes = self.filter_pos_assigned_but_empty_rois(pos_gt_bboxes, pos_batch_idx, pos_inds, rois[:, 0].int())
                pos_gt_labels = self.filter_pos_assigned_but_empty_rois(pos_gt_labels, pos_batch_idx, pos_inds, rois[:, 0].int())
                if self.train_cfg.get('corner_loss_only_car', True):
                    car_type_index = self.train_cfg['class_names'].index('Car')
                    car_mask = pos_gt_labels == car_type_index
                    pos_gt_bboxes = pos_gt_bboxes[car_mask]
                    pred_boxes3d = pred_boxes3d[car_mask]
                if len(pos_gt_bboxes) > 0:
                    loss_corner = self.get_corner_loss_lidar(
                        pred_boxes3d, pos_gt_bboxes) * self.corner_loss_weight
                else:
                    loss_corner = bbox_pred.sum() * 0

                losses['loss_rcnn_corner'] = loss_corner

        return losses
    
    def filter_pos_assigned_but_empty_rois(self, pos_data, pos_batch_idx, filtered_pos_mask, roi_batch_idx):
        real_bsz = roi_batch_idx.max().item() + 1
        filter_data_list = []
        for b_idx in range(real_bsz):
            roi_batch_mask = roi_batch_idx == b_idx
            data_batch_mask = pos_batch_idx == b_idx
            filter_data = pos_data[data_batch_mask][torch.nonzero(filtered_pos_mask[roi_batch_mask]).reshape(-1)]
            filter_data_list.append(filter_data)
        out = torch.cat(filter_data_list, 0)
        return out

    def get_targets(self, sampling_results, rcnn_train_cfg, concat=True):
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
        pos_label_list = [res.pos_gt_labels for res in sampling_results]
        targets = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            pos_gt_bboxes_list,
            iou_list,
            pos_label_list,
            cfg=rcnn_train_cfg)

        (label, bbox_targets, pos_gt_bboxes, reg_mask, label_weights,
         bbox_weights) = targets

        pos_gt_labels = pos_label_list
        bbox_target_batch_idx = []

        if concat:
            label = torch.cat(label, 0)
            bbox_target_batch_idx = torch.cat([t.new_ones(len(t), dtype=torch.int) * i for i, t in enumerate(bbox_targets)])
            bbox_targets = torch.cat(bbox_targets, 0)
            pos_gt_bboxes = torch.cat(pos_gt_bboxes, 0)
            pos_gt_labels = torch.cat(pos_gt_labels, 0)
            reg_mask = torch.cat(reg_mask, 0)

            label_weights = torch.cat(label_weights, 0)
            label_weights /= torch.clamp(label_weights.sum(), min=1.0)

            bbox_weights = torch.cat(bbox_weights, 0)
            bbox_weights /= torch.clamp(bbox_weights.sum(), min=1.0)

        return (label, bbox_targets, bbox_target_batch_idx, pos_gt_bboxes, pos_gt_labels, reg_mask, label_weights,
                bbox_weights)

    def _get_target_single(self, pos_bboxes, pos_gt_bboxes, ious, pos_labels, cfg):
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
        assert pos_gt_bboxes.size(1) in (7, 9, 10)
        if pos_gt_bboxes.size(1) in (9, 10):
            pos_bboxes = pos_bboxes[:, :7]
            pos_gt_bboxes = pos_gt_bboxes[:, :7]

        if self.num_classes > 1 or self.train_cfg.get('enable_multi_class_test', False):
            label, label_weights = self.get_multi_class_soft_label(ious, pos_labels, cfg)
        else:
            label, label_weights = self.get_single_class_soft_label(ious, cfg)

        # box regression target
        reg_mask = pos_bboxes.new_zeros(ious.size(0)).long()
        reg_mask[0:pos_gt_bboxes.size(0)] = 1
        bbox_weights = (reg_mask > 0).float()
        bbox_weights = self.get_class_wise_box_weights(bbox_weights, pos_labels, cfg)

        if reg_mask.bool().any():
            pos_gt_bboxes_ct = pos_gt_bboxes.clone().detach()
            roi_center = pos_bboxes[..., 0:3]
            roi_ry = pos_bboxes[..., 6] % (2 * np.pi)

            # canonical transformation
            pos_gt_bboxes_ct[..., 0:3] -= roi_center
            pos_gt_bboxes_ct[..., 6] -= roi_ry
            pos_gt_bboxes_ct[..., 0:3] = rotation_3d_in_axis(
                pos_gt_bboxes_ct[..., 0:3].unsqueeze(1),
                -(roi_ry + np.pi / 2),
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
    
    def get_class_wise_box_weights(self, weights, gt_labels, cfg):
        class_wise_weight = cfg.get('class_wise_box_weights', None)
        if class_wise_weight is None:
            return weights

        num_samples = len(weights)
        num_pos = len(gt_labels)
        all_gt_labels = torch.cat([gt_labels, gt_labels.new_full((num_samples - num_pos,), -1)], dim=0)
        for i in range(self.num_classes):
            this_cls_mask = (all_gt_labels == i)
            weights[this_cls_mask] *= class_wise_weight[i]

        return weights
    
    def get_single_class_soft_label(self, ious, cfg):

        cls_pos_mask = ious > cfg.cls_pos_thr
        cls_neg_mask = ious < cfg.cls_neg_thr
        interval_mask = (cls_pos_mask == 0) & (cls_neg_mask == 0)

        # iou regression target
        label = (cls_pos_mask > 0).float()
        # label[interval_mask] = ious[interval_mask] * 2 - 0.5
        label[interval_mask] = (ious[interval_mask] - cfg.cls_neg_thr) / (cfg.cls_pos_thr - cfg.cls_neg_thr)
        assert (label >= 0).all()
        # label weights
        label_weights = (label >= 0).float()
        return label, label_weights

    def get_multi_class_soft_label(self, ious, pos_gt_labels, cfg):
        pos_thrs = cfg.cls_pos_thr
        neg_thrs = cfg.cls_neg_thr

        if isinstance(pos_thrs, float):
            assert isinstance(neg_thrs, float)
            pos_thrs = [pos_thrs] * self.num_classes
            neg_thrs = [neg_thrs] * self.num_classes
        else:
            assert isinstance(pos_thrs, (list, tuple)) and isinstance(neg_thrs, (list, tuple))

        assert (pos_gt_labels >= 0).all()
        assert (pos_gt_labels < self.num_classes).all()
        num_samples = ious.size(0)
        num_pos = pos_gt_labels.size(0)

        # if num_pos > 0 and num_pos < len(ious):
        #     # all pos samples are in the left of ious array
        #     if not ious[num_pos-1].item() >= ious[num_pos].item():
        #         try:
        #             assert pos_gt_labels[-1].item() in (1, 2), 'The only resonable case is iou of positive Ped or Cyc less than positive Car'
        #         except AssertionError as e:
        #             print('Something werid happened')
        #             print('All ious: \n', ious)
        #             print('All labels: \n', pos_gt_labels)

            

        all_gt_labels = torch.cat([pos_gt_labels, pos_gt_labels.new_full((num_samples - num_pos,), -1)], dim=0)

        check = pos_gt_labels.new_zeros(ious.size(0)) - 1
        all_label = ious.new_zeros(ious.size(0))
        for i in range(self.num_classes):
            pos_thr_i = pos_thrs[i]
            neg_thr_i = neg_thrs[i]
            this_cls_mask = (all_gt_labels == i)
            check[this_cls_mask] += 1

            this_ious = ious[this_cls_mask]
            pos_mask = this_ious > pos_thr_i
            neg_mask = this_ious < neg_thr_i
            interval_mask = (pos_mask == 0) & (neg_mask == 0)
            this_label = (pos_mask > 0).float()
            this_label[interval_mask] = (this_ious[interval_mask] - neg_thr_i) / (pos_thr_i - neg_thr_i)
            all_label[this_cls_mask] = this_label


        assert (all_label >= 0).all()
        # label weights
        label_weights = (all_label >= 0).float()

        class_wise_weight = cfg.get('class_wise_cls_weights', None)
        if class_wise_weight is not None:
            for i in range(self.num_classes):
                this_cls_mask = (all_gt_labels == i)
                label_weights[this_cls_mask] *= class_wise_weight[i]

        assert (check[:num_pos] == 0).all()
        assert (check[num_pos:] == -1).all()
        return all_label, label_weights


    def get_corner_loss_lidar(self, pred_bbox3d, gt_bbox3d, delta=1):
        """Calculate corner loss of given boxes.

        Args:
            pred_bbox3d (torch.FloatTensor): Predicted boxes in shape (N, 7).
            gt_bbox3d (torch.FloatTensor): Ground truth boxes in shape (N, 7).

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
        quadratic = torch.clamp(abs_error, max=delta)
        linear = (abs_error - quadratic)
        corner_loss = 0.5 * quadratic**2 + delta * linear

        return corner_loss.mean()

    def get_bboxes(
        self,
        rois,
        cls_score,
        bbox_pred,
        valid_roi_mask,
        class_labels,
        class_pred,
        img_metas,
        cfg=None
    ):
        """Generate bboxes from bbox head predictions.

        Args:
            rois (torch.Tensor): Roi bounding boxes.
            cls_score (torch.Tensor): Scores of bounding boxes.
            bbox_pred (torch.Tensor): Bounding boxes predictions
            class_labels (torch.Tensor): Label of classes, from rpn.
            class_pred (torch.Tensor): Score for nms. From rpn
            img_metas (list[dict]): Point cloud and image's meta info.
            cfg (:obj:`ConfigDict`): Testing config.

        Returns:
            list[tuple]: Decoded bbox, scores and labels after nms.
        """
        assert rois.size(0) == cls_score.size(0) == bbox_pred.size(0)
        assert isinstance(class_labels, list) and isinstance(class_pred, list) and len(class_labels) == len(class_pred) == 1

        cls_score = cls_score.sigmoid()
        assert (class_pred[0] >= 0).all()

        if self.test_cfg.get('rcnn_score_nms', False):
            # assert class_pred[0].shape == cls_score.shape
            class_pred[0] = cls_score.squeeze(1)

        # regard empty bboxes as false positive
        rois = rois[valid_roi_mask]
        cls_score = cls_score[valid_roi_mask]
        bbox_pred = bbox_pred[valid_roi_mask]


        for i in range(len(class_labels)):
            class_labels[i] = class_labels[i][valid_roi_mask]
            class_pred[i] = class_pred[i][valid_roi_mask]

        if rois.numel() == 0:
            return [(
                img_metas[0]['box_type_3d'](rois[:, 1:], rois.size(1) - 1),
                class_pred[0],
                class_labels[0]
            ),]


        roi_batch_id = rois[..., 0]
        roi_boxes = rois[..., 1:]  # boxes without batch id
        batch_size = int(roi_batch_id.max().item() + 1)

        # decode boxes
        roi_ry = roi_boxes[..., 6].view(-1)
        roi_xyz = roi_boxes[..., 0:3].view(-1, 3)
        local_roi_boxes = roi_boxes.clone().detach()
        local_roi_boxes[..., 0:3] = 0

        assert local_roi_boxes.size(1) in (7, 9) # with or without velocity
        if local_roi_boxes.size(1) == 9:
            # fake zero predicted velocity, which means rcnn do not refine the velocity
            bbox_pred = F.pad(bbox_pred, (0, 2), 'constant', 0)

        rcnn_boxes3d = self.bbox_coder.decode(local_roi_boxes, bbox_pred)
        rcnn_boxes3d[..., 0:3] = rotation_3d_in_axis(
            rcnn_boxes3d[..., 0:3].unsqueeze(1), (roi_ry + np.pi / 2),
            axis=2).squeeze(1)
        rcnn_boxes3d[:, 0:3] += roi_xyz

        # post processing
        result_list = []
        if cfg.get('multi_class_nms', False) or self.num_classes > 1:
            nms_func = self.multi_class_nms
        else:
            nms_func = self.single_class_nms

        for batch_id in range(batch_size):
            cur_class_labels = class_labels[batch_id]
            if batch_size == 1:
                cur_cls_score = cls_score.view(-1)
                cur_rcnn_boxes3d = rcnn_boxes3d
            else:
                roi_batch_mask = roi_batch_id == batch_id
                cur_cls_score = cls_score[roi_batch_mask].view(-1)
                cur_rcnn_boxes3d = rcnn_boxes3d[roi_batch_mask]

            cur_box_prob = class_pred[batch_id]
            selected = nms_func(cur_box_prob, cur_class_labels, cur_rcnn_boxes3d,
                                cfg.score_thr, cfg.nms_thr,
                                img_metas[batch_id],
                                cfg.use_rotate_nms)
            selected_bboxes = cur_rcnn_boxes3d[selected]
            selected_label_preds = cur_class_labels[selected]
            selected_scores = cur_cls_score[selected]

            result_list.append(
                (img_metas[batch_id]['box_type_3d'](selected_bboxes, selected_bboxes.size(1)),
                selected_scores, selected_label_preds))
        return result_list

    def multi_class_nms(self,
                        box_probs,
                        labels, # labels from rpn
                        box_preds,
                        score_thr,
                        nms_thr,
                        input_meta,
                        use_rotate_nms=True):
        """Multi-class NMS for box head.

        Note:
            This function has large overlap with the `box3d_multiclass_nms`
            implemented in `mmdet3d.core.post_processing`. We are considering
            merging these two functions in the future.

        Args:
            box_probs (torch.Tensor): Predicted boxes probabitilies in
                shape (N,).
            box_preds (torch.Tensor): Predicted boxes in shape (N, 7+C).
            score_thr (float): Threshold of scores.
            nms_thr (float): Threshold for NMS.
            input_meta (dict): Meta informations of the current sample.
            use_rotate_nms (bool, optional): Whether to use rotated nms.
                Defaults to True.

        Returns:
            torch.Tensor: Selected indices.
        """
        if use_rotate_nms:
            nms_func = nms_gpu
        else:
            nms_func = nms_normal_gpu

        assert box_probs.ndim == 1

        selected_list = []
        boxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d'](
            box_preds, box_preds.size(1)).bev)

        score_thresh = score_thr if isinstance(
            score_thr, (list, tuple)) else [score_thr for x in range(self.num_classes)]
        nms_thresh = nms_thr if isinstance(
            nms_thr, (list, tuple)) else [nms_thr for x in range(self.num_classes)]

        for k in range(0, self.num_classes):
            class_scores_keep = (box_probs >= score_thresh[k]) & (labels == k)

            if class_scores_keep.any():
                original_idxs = class_scores_keep.nonzero(
                    as_tuple=False).view(-1)
                cur_boxes_for_nms = boxes_for_nms[class_scores_keep]
                cur_rank_scores = box_probs[class_scores_keep]

                cur_selected = nms_func(cur_boxes_for_nms, cur_rank_scores,
                                        nms_thresh[k])

                if cur_selected.shape[0] == 0:
                    continue
                selected_list.append(original_idxs[cur_selected])

        selected = torch.cat(
            selected_list, dim=0) if len(selected_list) > 0 else []
        return selected

    def single_class_nms(self,
                        box_probs,
                        labels, # labels from rpn
                        box_preds,
                        score_thr,
                        nms_thr,
                        input_meta,
                        use_rotate_nms=True):

        if use_rotate_nms:
            nms_func = nms_gpu
        else:
            nms_func = nms_normal_gpu

        assert box_probs.ndim == 1

        selected_list = []
        boxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d'](
            box_preds, box_preds.size(1)).bev)

        assert isinstance(score_thr, float)
        score_thresh = score_thr
        nms_thresh = nms_thr
        class_scores_keep = box_probs >= score_thresh

        if class_scores_keep.int().sum() > 0:
            original_idxs = class_scores_keep.nonzero(
                as_tuple=False).view(-1)
            cur_boxes_for_nms = boxes_for_nms[class_scores_keep]
            cur_rank_scores = box_probs[class_scores_keep]

            if nms_thresh is not None:
                cur_selected = nms_func(cur_boxes_for_nms, cur_rank_scores, nms_thresh)
            else:
                cur_selected = torch.arange(len(original_idxs), device=original_idxs.device, dtype=torch.long)

            if len(cur_selected) > 0:
                selected_list.append(original_idxs[cur_selected])

        selected = torch.cat(selected_list, dim=0) if len(selected_list) > 0 else []
        return selected