import torch
from mmcv.cnn import ConvModule
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.core.bbox.structures import (DepthInstance3DBoxes,
                                          LiDARInstance3DBoxes,
                                          rotation_3d_in_axis)
from mmdet3d.core.post_processing import aligned_3d_nms
from mmdet3d.models.builder import build_loss
from mmdet3d.ops import PointSAModuleMSG
from mmdet.core import build_bbox_coder, multi_apply
from mmdet.models import HEADS


@HEADS.register_module()
class SSD3DHead(nn.Module):
    r"""Bbox head of `3DSSD <https://arxiv.org/abs/2002.10187>`_.

    Args:
        num_classes (int): The number of class.
        bbox_coder (:obj:`BaseBBoxCoder`): Bbox coder for encoding and
            decoding boxes.
        num_candidates (int): The number of candidate points.
        in_channels (int): The number of input feature channel.
        train_cfg (dict): Config for training.
        test_cfg (dict): Config for testing.
        vote_aggregation_cfg (dict): Config of vote aggregation layer.
        candidate_points_mlps (tuple[int]): Config of mlps for generating
            candidate points.
        feat_channels (tuple[int]): Convolution channels for aggregation
            multi-scale grouping features.
        cls_pred_mlps (tuple[int]): Convolution channels of class
            prediction layer.
        reg_pred_mlps (tuple[int]): Convolution channels of regression
            prediction layer.
        conv_cfg (dict): Config of convolution in prediction layer.
        norm_cfg (dict): Config of BN in prediction layer.
        act_cfg (dict): Config of activation in prediction layer.
        objectness_loss (dict): Config of objectness loss.
        center_loss (dict): Config of center loss.
        dir_class_loss (dict): Config of direction classification loss.
        dir_res_loss (dict): Config of direction residual regression loss.
        size_res_loss (dict): Config of size residual regression loss.
        corner_loss (dict): Config of bbox corners regression loss.
        vote_loss (dict): Config of candidate points regression loss.
    """

    def __init__(self,
                 num_classes,
                 bbox_coder,
                 num_candidates=256,
                 in_channels=256,
                 train_cfg=None,
                 test_cfg=None,
                 vote_aggregation_cfg=None,
                 candidate_points_mlps=[128],
                 feat_channels=(128, ),
                 cls_pred_mlps=(128, ),
                 reg_pred_mlps=(128, ),
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU'),
                 objectness_loss=None,
                 center_loss=None,
                 dir_class_loss=None,
                 dir_res_loss=None,
                 size_res_loss=None,
                 corner_loss=None,
                 vote_loss=None):
        super(SSD3DHead, self).__init__()
        self.num_classes = num_classes
        self.num_candidates = num_candidates
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_proposal = vote_aggregation_cfg['num_point']

        self.objectness_loss = build_loss(objectness_loss)
        self.center_loss = build_loss(center_loss)
        self.dir_class_loss = build_loss(dir_class_loss)
        self.dir_res_loss = build_loss(dir_res_loss)
        self.size_res_loss = build_loss(size_res_loss)
        self.corner_loss = build_loss(corner_loss)
        self.vote_loss = build_loss(vote_loss)

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.num_dir_bins = self.bbox_coder.num_dir_bins

        candidate_points_mlps = [in_channels] + list(candidate_points_mlps)
        self.candidate_points_layers = nn.Sequential()
        for i in range(len(candidate_points_mlps) - 1):
            self.candidate_points_layers.add_module(
                f'layer{i}',
                ConvModule(
                    candidate_points_mlps[i],
                    candidate_points_mlps[i + 1],
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    bias=True,
                    inplace=True))
        self.candidate_points_layers.add_module(
            'pred', nn.Conv1d(candidate_points_mlps[-1], 3, 1))

        self.vote_aggregation = PointSAModuleMSG(**vote_aggregation_cfg)

        prev_channel = sum(
            [_[-1] for _ in vote_aggregation_cfg['mlp_channels']])
        feats_aggregation_list = list()
        for k in range(len(feat_channels)):
            feats_aggregation_list.append(
                ConvModule(
                    prev_channel,
                    feat_channels[k],
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    bias=True,
                    inplace=True))
            prev_channel = feat_channels[k]
        self.feats_aggregation_layers = nn.Sequential(*feats_aggregation_list)

        # Bbox classification
        prev_channel = feat_channels[-1]
        cls_pred_list = list()
        for k in range(len(cls_pred_mlps)):
            cls_pred_list.append(
                ConvModule(
                    prev_channel,
                    cls_pred_mlps[k],
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    bias=True,
                    inplace=True))
            prev_channel = cls_pred_mlps[k]
        self.cls_pred = nn.Sequential(*cls_pred_list)
        self.cls_pred.add_module('cls_pred',
                                 nn.Conv1d(prev_channel, num_classes, 1))

        # Bbox regression
        prev_channel = feat_channels[-1]
        regression_pred_list = list()
        for k in range(len(reg_pred_mlps)):
            regression_pred_list.append(
                ConvModule(
                    prev_channel,
                    reg_pred_mlps[k],
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    bias=True,
                    inplace=True))
            prev_channel = reg_pred_mlps[k]
        self.reg_pred = nn.Sequential(*regression_pred_list)
        # (center residual (3), size regression (3)
        # heading class+residual (num_dir_bins*2)),
        conv_out_channel = (3 + 3 + self.num_dir_bins * 2)
        self.reg_pred.add_module('reg_pred',
                                 nn.Conv1d(prev_channel, conv_out_channel, 1))

    def init_weights(self):
        """Initialize weights of VoteHead."""
        pass

    def forward(self, feat_dict):
        """Forward pass.

        Note:
            The forward of SSDHead is devided into 4 steps:

                1. Generate candidate points from seed_points.
                2. Aggregate candidate points and features.
                3. Predict bbox and score.
                4. Decode predictions.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            dict: Predictions of vote head.
        """
        seed_points = feat_dict['sa_xyz']
        seed_features = feat_dict['sa_features']
        seed_indices = feat_dict['sa_indices']

        # 1. generate candidate points from seed_points
        candidate_offset = self.candidate_points_layers(
            seed_features[:, :, :self.num_candidates])

        limited_candidate_offset = candidate_offset.clone().transpose(1, 2)
        for axis in range(len(self.test_cfg.max_translate_range)):
            limited_candidate_offset[..., axis].clamp(
                min=-self.test_cfg.max_translate_range[axis],
                max=self.test_cfg.max_translate_range[axis])

        candidate_points = seed_points[:, :self.num_candidates] + \
            limited_candidate_offset

        results = dict(
            seed_points=seed_points[:, :self.num_candidates],
            seed_indices=seed_indices,
            candidate_points=candidate_points,
            candidate_offset=candidate_offset)

        # 2. aggregate vote_points
        vote_aggregation_ret = self.vote_aggregation(
            seed_points, seed_features, target_xyz=candidate_points)
        aggregated_points, features, aggregated_indices = vote_aggregation_ret
        results['aggregated_points'] = aggregated_points
        results['aggregated_indices'] = aggregated_indices

        # 3. predict bbox and score
        features = self.feats_aggregation_layers(features)
        cls_predictions = self.cls_pred(features)
        reg_predictions = self.reg_pred(features)
        results['obj_scores'] = cls_predictions

        # 4. decode predictions
        decode_res = self.bbox_coder.split_pred(reg_predictions,
                                                aggregated_points)
        results.update(decode_res)

        return results

    def loss(self,
             bbox_preds,
             points,
             gt_bboxes_3d,
             gt_labels_3d,
             pts_semantic_mask=None,
             pts_instance_mask=None,
             img_metas=None,
             gt_bboxes_ignore=None):
        """Compute loss.

        Args:
            bbox_preds (dict): Predictions from forward of vote head.
            points (list[torch.Tensor]): Input points.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth \
                bboxes of each sample.
            gt_labels_3d (list[torch.Tensor]): Labels of each sample.
            pts_semantic_mask (None | list[torch.Tensor]): Point-wise
                semantic mask.
            pts_instance_mask (None | list[torch.Tensor]): Point-wise
                instance mask.
            img_metas (list[dict]): Contain pcd and img's meta info.
            gt_bboxes_ignore (None | list[torch.Tensor]): Specify
                which bounding.

        Returns:
            dict: Losses of Votenet.
        """
        targets = self.get_targets(points, gt_bboxes_3d, gt_labels_3d,
                                   pts_semantic_mask, pts_instance_mask,
                                   bbox_preds)
        (vote_targets, center_targets, size_res_targets, dir_class_targets,
         dir_res_targets, mask_targets, centerness_targets, corner3d_targets,
         vote_mask, positive_mask, negative_mask, centerness_weights,
         box_loss_weights, heading_res_loss_weight) = targets

        # calculate centerness loss
        centerness_loss = self.objectness_loss(
            bbox_preds['obj_scores'].transpose(2, 1),
            centerness_targets,
            weight=centerness_weights)

        # calculate center loss
        center_loss = self.center_loss(
            bbox_preds['center'],
            center_targets,
            weight=box_loss_weights.unsqueeze(-1))

        # calculate direction class loss
        dir_class_loss = self.dir_class_loss(
            bbox_preds['dir_class'].transpose(1, 2),
            dir_class_targets,
            weight=box_loss_weights)

        # calculate direction residual loss
        dir_res_loss = self.dir_res_loss(
            bbox_preds['dir_res'],
            dir_res_targets.unsqueeze(-1).repeat(1, 1, self.num_dir_bins),
            weight=heading_res_loss_weight)

        # calculate size residual loss
        size_loss = self.size_res_loss(
            bbox_preds['size'],
            size_res_targets,
            weight=box_loss_weights.unsqueeze(-1))

        # calculate corner loss
        pred_bbox3d = self.bbox_coder.decode(bbox_preds)
        pred_bbox3d = pred_bbox3d.reshape(-1, pred_bbox3d.shape[-1])
        pred_bbox3d = img_metas[0]['box_type_3d'](
            pred_bbox3d.clone(),
            box_dim=pred_bbox3d.shape[-1],
            with_yaw=self.bbox_coder.with_rot,
            origin=(0.5, 0.5, 0.5))
        pred_corners3d = pred_bbox3d.corners.reshape(-1, 8, 3)
        corner_loss = self.corner_loss(
            pred_corners3d,
            corner3d_targets.reshape(-1, 8, 3),
            weight=box_loss_weights.view(-1, 1, 1))

        # calculate vote loss
        vote_loss = self.vote_loss(
            bbox_preds['candidate_offset'].transpose(1, 2),
            vote_targets,
            weight=vote_mask.unsqueeze(-1))

        losses = dict(
            centerness_loss=centerness_loss,
            center_loss=center_loss,
            dir_class_loss=dir_class_loss,
            dir_res_loss=dir_res_loss,
            size_res_loss=size_loss,
            corner_loss=corner_loss,
            vote_loss=vote_loss)

        return losses

    def get_targets(self,
                    points,
                    gt_bboxes_3d,
                    gt_labels_3d,
                    pts_semantic_mask=None,
                    pts_instance_mask=None,
                    bbox_preds=None):
        """Generate targets of vote head.

        Args:
            points (list[torch.Tensor]): Points of each batch.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth \
                bboxes of each batch.
            gt_labels_3d (list[torch.Tensor]): Labels of each batch.
            pts_semantic_mask (None | list[torch.Tensor]): Point-wise semantic
                label of each batch.
            pts_instance_mask (None | list[torch.Tensor]): Point-wise instance
                label of each batch.
            bbox_preds (torch.Tensor): Bounding box predictions of vote head.

        Returns:
            tuple[torch.Tensor]: Targets of vote head.
        """
        # find empty example
        for index in range(len(gt_labels_3d)):
            if len(gt_labels_3d[index]) == 0:
                fake_box = gt_bboxes_3d[index].tensor.new_zeros(
                    1, gt_bboxes_3d[index].tensor.shape[-1])
                gt_bboxes_3d[index] = gt_bboxes_3d[index].new_box(fake_box)
                gt_labels_3d[index] = gt_labels_3d[index].new_zeros(1)

        if pts_semantic_mask is None:
            pts_semantic_mask = [None for i in range(len(gt_labels_3d))]
            pts_instance_mask = [None for i in range(len(gt_labels_3d))]

        aggregated_points = [
            bbox_preds['aggregated_points'][i].detach()
            for i in range(len(gt_labels_3d))
        ]

        seed_points = [
            bbox_preds['seed_points'][i].detach()
            for i in range(len(gt_labels_3d))
        ]

        (vote_targets, center_targets, size_res_targets, dir_class_targets,
         dir_res_targets, mask_targets, centerness_targets, corner3d_targets,
         vote_mask, positive_mask, negative_mask) = multi_apply(
             self.get_targets_single, points, gt_bboxes_3d, gt_labels_3d,
             pts_semantic_mask, pts_instance_mask, aggregated_points,
             seed_points)

        center_targets = torch.stack(center_targets)
        positive_mask = torch.stack(positive_mask)
        negative_mask = torch.stack(negative_mask)
        dir_class_targets = torch.stack(dir_class_targets)
        dir_res_targets = torch.stack(dir_res_targets)
        size_res_targets = torch.stack(size_res_targets)
        mask_targets = torch.stack(mask_targets)
        centerness_targets = torch.stack(centerness_targets)
        corner3d_targets = torch.stack(corner3d_targets)
        vote_targets = torch.stack(vote_targets)
        vote_mask = torch.stack(vote_mask)

        centerness_weights = (positive_mask +
                              negative_mask).unsqueeze(-1).repeat(
                                  1, 1, self.num_classes).float()
        centerness_weights = centerness_weights / \
            (centerness_weights.sum() + 1e-6)
        vote_mask = vote_mask / (vote_mask.sum() + 1e-6)

        box_loss_weights = positive_mask / (positive_mask.sum() + 1e-6)

        batch_size, proposal_num = dir_class_targets.shape[:2]
        heading_label_one_hot = dir_class_targets.new_zeros(
            (batch_size, proposal_num, self.num_dir_bins))
        heading_label_one_hot.scatter_(2, dir_class_targets.unsqueeze(-1), 1)
        heading_res_loss_weight = heading_label_one_hot * \
            box_loss_weights.unsqueeze(-1)

        return (vote_targets, center_targets, size_res_targets,
                dir_class_targets, dir_res_targets, mask_targets,
                centerness_targets, corner3d_targets, vote_mask, positive_mask,
                negative_mask, centerness_weights, box_loss_weights,
                heading_res_loss_weight)

    def get_targets_single(self,
                           points,
                           gt_bboxes_3d,
                           gt_labels_3d,
                           pts_semantic_mask=None,
                           pts_instance_mask=None,
                           aggregated_points=None,
                           seed_points=None):
        """Generate targets of vote head for single batch.

        Args:
            points (torch.Tensor): Points of each batch.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): Ground truth \
                boxes of each batch.
            gt_labels_3d (torch.Tensor): Labels of each batch.
            pts_semantic_mask (None | torch.Tensor): Point-wise semantic
                label of each batch.
            pts_instance_mask (None | torch.Tensor): Point-wise instance
                label of each batch.
            aggregated_points (torch.Tensor): Aggregated points from
                candidate points layer.
            seed_points (torch.Tensor): Seed points of candidate points.

        Returns:
            tuple[torch.Tensor]: Targets of vote head.
        """
        assert self.bbox_coder.with_rot or pts_semantic_mask is not None
        gt_bboxes_3d = gt_bboxes_3d.to(points.device)
        valid_gt = gt_labels_3d != -1
        gt_bboxes_3d = gt_bboxes_3d[valid_gt]
        gt_labels_3d = gt_labels_3d[valid_gt]
        gt_corner3d = gt_bboxes_3d.corners

        (center_targets, size_targets, dir_class_targets,
         dir_res_targets) = self.bbox_coder.encode(gt_bboxes_3d, gt_labels_3d)

        points_mask, assignment = self._assign_targets_by_points_inside(
            gt_bboxes_3d, aggregated_points)

        center_targets = center_targets[assignment]
        size_res_targets = size_targets[assignment]
        mask_targets = gt_labels_3d[assignment]
        dir_class_targets = dir_class_targets[assignment]
        dir_res_targets = dir_res_targets[assignment]
        corner3d_targets = gt_corner3d[assignment]

        dist = torch.norm(aggregated_points - center_targets, dim=1)
        dist_mask = dist < self.train_cfg.pos_distance_thr
        positive_mask = (points_mask.max(1)[0] > 0) * dist_mask
        negative_mask = (points_mask.max(1)[0] == 0)

        # Centerness loss targets
        canonical_xyz = aggregated_points - center_targets
        if self.bbox_coder.with_rot:
            # TODO:
            # Align LiDARInstance3DBoxes and DepthInstance3DBoxes
            # for points rotation
            canonical_xyz = rotation_3d_in_axis(
                canonical_xyz.unsqueeze(0).transpose(0, 1),
                -gt_bboxes_3d.yaw[assignment], 2).squeeze(1)
        distance_front = torch.clamp(
            size_res_targets[:, 0] - canonical_xyz[:, 0], min=0)
        distance_back = torch.clamp(
            size_res_targets[:, 0] + canonical_xyz[:, 0], min=0)
        distance_left = torch.clamp(
            size_res_targets[:, 1] - canonical_xyz[:, 1], min=0)
        distance_right = torch.clamp(
            size_res_targets[:, 1] + canonical_xyz[:, 1], min=0)
        distance_top = torch.clamp(
            size_res_targets[:, 2] - canonical_xyz[:, 2], min=0)
        distance_bottom = torch.clamp(
            size_res_targets[:, 2] + canonical_xyz[:, 2], min=0)

        centerness_l = torch.min(distance_front, distance_back) / torch.max(
            distance_front, distance_back)
        centerness_w = torch.min(distance_left, distance_right) / torch.max(
            distance_left, distance_right)
        centerness_h = torch.min(distance_bottom, distance_top) / torch.max(
            distance_bottom, distance_top)
        centerness_targets = torch.clamp(
            centerness_l * centerness_w * centerness_h, min=0)
        centerness_targets = centerness_targets.pow(1 / 3.0)
        centerness_targets = torch.clamp(centerness_targets, min=0, max=1)

        proposal_num = centerness_targets.shape[0]
        one_hot_centerness_targets = centerness_targets.new_zeros(
            (proposal_num, self.num_classes))
        one_hot_centerness_targets.scatter_(1, mask_targets.unsqueeze(-1), 1)
        centerness_targets = centerness_targets.unsqueeze(
            1) * one_hot_centerness_targets

        # Vote loss targets
        enlarged_gt_bboxes_3d = gt_bboxes_3d.enlarged_box(
            self.train_cfg.expand_dims_length)
        vote_mask, vote_assignment = self._assign_targets_by_points_inside(
            enlarged_gt_bboxes_3d, seed_points)

        vote_targets = enlarged_gt_bboxes_3d.gravity_center
        vote_targets = vote_targets[vote_assignment] - seed_points
        vote_mask = vote_mask.max(1)[0] > 0

        return (vote_targets, center_targets, size_res_targets,
                dir_class_targets, dir_res_targets, mask_targets,
                centerness_targets, corner3d_targets, vote_mask, positive_mask,
                negative_mask)

    def get_bboxes(self, points, bbox_preds, input_metas, rescale=False):
        """Generate bboxes from sdd 3d head predictions.

        Args:
            points (torch.Tensor): Input points.
            bbox_preds (dict): Predictions from sdd 3d head.
            input_metas (list[dict]): Point cloud and image's meta info.
            rescale (bool): Whether to rescale bboxes.

        Returns:
            list[tuple[torch.Tensor]]: Bounding boxes, scores and labels.
        """
        # decode boxes
        sem_scores = F.sigmoid(bbox_preds['obj_scores']).transpose(1, 2)
        obj_scores = sem_scores.max(-1)[0]
        bbox3d = self.bbox_coder.decode(bbox_preds)

        batch_size = bbox3d.shape[0]
        results = list()
        for b in range(batch_size):
            bbox_selected, score_selected, labels = self.multiclass_nms_single(
                obj_scores[b], sem_scores[b], bbox3d[b], points[b, ..., :3],
                input_metas[b])
            bbox = input_metas[b]['box_type_3d'](
                bbox_selected.clone(),
                box_dim=bbox_selected.shape[-1],
                with_yaw=self.bbox_coder.with_rot)
            results.append((bbox, score_selected, labels))

        return results

    def multiclass_nms_single(self, obj_scores, sem_scores, bbox, points,
                              input_meta):
        """Multi-class nms in single batch.

        Args:
            obj_scores (torch.Tensor): Objectness score of bounding boxes.
            sem_scores (torch.Tensor): semantic class score of bounding boxes.
            bbox (torch.Tensor): Predicted bounding boxes.
            points (torch.Tensor): Input points.
            input_meta (dict): Point cloud and image's meta info.

        Returns:
            tuple[torch.Tensor]: Bounding boxes, scores and labels.
        """
        num_bbox = bbox.shape[0]
        bbox = input_meta['box_type_3d'](
            bbox.clone(),
            box_dim=bbox.shape[-1],
            with_yaw=self.bbox_coder.with_rot,
            origin=(0.5, 0.5, 1.0))
        if isinstance(bbox, LiDARInstance3DBoxes):
            box_idx = bbox.points_in_boxes(points)
            box_indices = box_idx.new_zeros([num_bbox + 1])
            box_idx[box_idx == -1] = num_bbox
            box_indices.scatter_add_(0, box_idx.long(),
                                     box_idx.new_ones(box_idx.shape))
            box_indices = box_indices[:-1]
            nonempty_box_mask = box_indices > 0
        elif isinstance(bbox, DepthInstance3DBoxes):
            box_indices = bbox.points_in_boxes(points)
            nonempty_box_mask = box_indices.T.sum(1) > 0
        else:
            raise NotImplementedError('Unsupported bbox type!')

        corner3d = bbox.corners
        minmax_box3d = corner3d.new(torch.Size((corner3d.shape[0], 6)))
        minmax_box3d[:, :3] = torch.min(corner3d, dim=1)[0]
        minmax_box3d[:, 3:] = torch.max(corner3d, dim=1)[0]

        bbox_classes = torch.argmax(sem_scores, -1)
        nms_selected = aligned_3d_nms(minmax_box3d[nonempty_box_mask],
                                      obj_scores[nonempty_box_mask],
                                      bbox_classes[nonempty_box_mask],
                                      self.test_cfg.nms_thr)

        # filter empty boxes and boxes with low score
        scores_mask = (obj_scores > self.test_cfg.score_thr)
        nonempty_box_inds = torch.nonzero(nonempty_box_mask).flatten()
        nonempty_mask = torch.zeros_like(bbox_classes).scatter(
            0, nonempty_box_inds[nms_selected], 1)
        selected = (nonempty_mask.bool() & scores_mask.bool())

        if self.test_cfg.per_class_proposal:
            bbox_selected, score_selected, labels = [], [], []
            for k in range(sem_scores.shape[-1]):
                bbox_selected.append(bbox[selected].tensor)
                score_selected.append(obj_scores[selected] *
                                      sem_scores[selected][:, k])
                labels.append(
                    torch.zeros_like(bbox_classes[selected]).fill_(k))
            bbox_selected = torch.cat(bbox_selected, 0)
            score_selected = torch.cat(score_selected, 0)
            labels = torch.cat(labels, 0)
        else:
            bbox_selected = bbox[selected].tensor
            score_selected = obj_scores[selected]
            labels = bbox_classes[selected]

        return bbox_selected, score_selected, labels

    def _assign_targets_by_points_inside(self, bboxes_3d, points):
        """Compute assignment by checking whether point is inside bbox.

        Args:
            bboxes_3d (BaseInstance3DBoxes): Instance of bounding boxes.
            points (torch.Tensor): Points of a batch.

        Returns:
            tuple[torch.Tensor]: Flags indicating whether each point is
                inside bbox and the index of box where each point are in.
        """
        # TODO: align points_in_boxes function in each box_structures
        num_bbox = bboxes_3d.tensor.shape[0]
        if isinstance(bboxes_3d, LiDARInstance3DBoxes):
            assignment = bboxes_3d.points_in_boxes(points).long()
            points_mask = assignment.new_zeros(
                [assignment.shape[0], num_bbox + 1])
            assignment[assignment == -1] = num_bbox
            points_mask.scatter_(1, assignment.unsqueeze(1), 1)
            points_mask = points_mask[:, :-1]
            assignment[assignment == num_bbox] = num_bbox - 1
        elif isinstance(bboxes_3d, DepthInstance3DBoxes):
            points_mask = bboxes_3d.points_in_boxes(points)
            assignment = points_mask.argmax(dim=-1)
        else:
            raise NotImplementedError('Unsupported bbox type!')
        return points_mask, assignment
