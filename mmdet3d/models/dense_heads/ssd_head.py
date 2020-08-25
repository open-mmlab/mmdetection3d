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
    r"""Bbox head of `Votenet <https://arxiv.org/abs/1904.09664>`_.

    Args:
        num_classes (int): The number of class.
        bbox_coder (:obj:`BaseBBoxCoder`): Bbox coder for encoding and
            decoding boxes.
        train_cfg (dict): Config for training.
        test_cfg (dict): Config for testing.
        vote_moudule_cfg (dict): Config of VoteModule for point-wise votes.
        vote_aggregation_cfg (dict): Config of vote aggregation layer.
        feat_channels (tuple[int]): Convolution channels of
            prediction layer.
        conv_cfg (dict): Config of convolution in prediction layer.
        norm_cfg (dict): Config of BN in prediction layer.
        objectness_loss (dict): Config of objectness loss.
        center_loss (dict): Config of center loss.
        dir_class_loss (dict): Config of direction classification loss.
        dir_res_loss (dict): Config of direction residual regression loss.
        size_class_loss (dict): Config of size classification loss.
        size_res_loss (dict): Config of size residual regression loss.
        semantic_loss (dict): Config of point-wise semantic segmentation loss.
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
                 size_class_loss=None,
                 size_res_loss=None,
                 semantic_loss=None):
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
        self.size_class_loss = build_loss(size_class_loss)
        self.size_res_loss = build_loss(size_res_loss)
        self.semantic_loss = build_loss(semantic_loss)

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.num_regressions = self.bbox_coder.num_regressions
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
        # heading class+residual (num_dir_bins*2)) * num_regressions,
        conv_out_channel = self.num_regressions * \
            (3 + 3 + self.num_dir_bins * 2)
        self.reg_pred.add_module('reg_pred',
                                 nn.Conv1d(prev_channel, conv_out_channel, 1))

    def init_weights(self):
        """Initialize weights of VoteHead."""
        pass

    def forward(self, feat_dict, sample_mod):
        """Forward pass.

        Note:
            The forward of VoteHead is devided into 4 steps:

                1. Generate candidate points from seed_points.
                2. Aggregate vote_points.
                3. Predict bbox and score.
                4. Decode predictions.

        Args:
            feat_dict (dict): Feature dict from backbone.
            sample_mod (str): Sample mode for vote aggregation layer.
                valid modes are "vote", "seed" and "random".

        Returns:
            dict: Predictions of vote head.
        """
        assert sample_mod in ['vote', 'seed', 'random']

        seed_points = feat_dict['sa_xyz']
        seed_features = feat_dict['sa_features']
        seed_indices = feat_dict['sa_indices']

        # 1. generate vote_points from seed_points
        candidate_offset = self.candidate_points_layers(
            seed_features[:, :, :self.num_candidates])

        candidate_points = seed_points[:, :self.num_candidates] + \
            candidate_offset.transpose(1, 2)

        results = dict(
            seed_points=seed_points,
            seed_indices=seed_indices,
            candidate_points=candidate_points)

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
        (center_targets, size_res_targets, dir_class_targets, dir_res_targets,
         mask_targets, centerness_targets, positive_mask,
         negative_mask) = targets
        # import pdb
        # pdb.set_trace()
        # calculate centerness loss
        centerness_weights = (positive_mask +
                              negative_mask).unsqueeze(-1).repeat(
                                  1, 1, self.num_classes)
        centerness_loss = self.objectness_loss(
            bbox_preds['obj_scores'].transpose(2, 1),
            centerness_targets,
            weight=centerness_weights / (centerness_weights.sum() + 1e-6))

        box_loss_weights = positive_mask / (positive_mask.sum() + 1e-6)
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
        batch_size, proposal_num = dir_class_targets.shape[:2]
        heading_label_one_hot = dir_class_targets.new_zeros(
            (batch_size, proposal_num, self.num_dir_bins))
        heading_label_one_hot.scatter_(2, dir_class_targets.unsqueeze(-1), 1)
        assert dir_class_targets.max() < self.num_dir_bins
        assert dir_class_targets.min() >= 0

        dir_res_norm = torch.sum(
            bbox_preds['dir_res_norm'] * heading_label_one_hot, -1)
        dir_res_loss = self.dir_res_loss(
            dir_res_norm, dir_res_targets, weight=box_loss_weights)

        # calculate size residual loss
        size_loss = self.size_res_loss(
            bbox_preds['size'],
            size_res_targets,
            weight=box_loss_weights.unsqueeze(-1))

        losses = dict(
            centerness_loss=centerness_loss,
            center_loss=center_loss,
            dir_class_loss=dir_class_loss,
            dir_res_loss=dir_res_loss,
            size_res_loss=size_loss)
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
        valid_gt_masks = list()
        gt_num = list()
        for index in range(len(gt_labels_3d)):
            if len(gt_labels_3d[index]) == 0:
                fake_box = gt_bboxes_3d[index].tensor.new_zeros(
                    1, gt_bboxes_3d[index].tensor.shape[-1])
                gt_bboxes_3d[index] = gt_bboxes_3d[index].new_box(fake_box)
                gt_labels_3d[index] = gt_labels_3d[index].new_zeros(1)
                valid_gt_masks.append(gt_labels_3d[index].new_zeros(1))
                gt_num.append(1)
            else:
                valid_gt_masks.append(gt_labels_3d[index].new_ones(
                    gt_labels_3d[index].shape))
                gt_num.append(gt_labels_3d[index].shape[0])
        max_gt_num = max(gt_num)

        if pts_semantic_mask is None:
            pts_semantic_mask = [None for i in range(len(gt_labels_3d))]
            pts_instance_mask = [None for i in range(len(gt_labels_3d))]

        aggregated_points = [
            bbox_preds['aggregated_points'][i].detach()
            for i in range(len(gt_labels_3d))
        ]

        (center_targets, size_res_targets, dir_class_targets, dir_res_targets,
         mask_targets, centerness_targets, positive_mask,
         negative_mask) = multi_apply(self.get_targets_single, points,
                                      gt_bboxes_3d, gt_labels_3d,
                                      pts_semantic_mask, pts_instance_mask,
                                      aggregated_points)

        # pad targets as original code of votenet.
        for index in range(len(gt_labels_3d)):
            pad_num = max_gt_num - gt_labels_3d[index].shape[0]
            # center_targets[index] = F.pad(center_targets[index],
            #                               (0, 0, 0, pad_num))
            valid_gt_masks[index] = F.pad(valid_gt_masks[index], (0, pad_num))

        center_targets = torch.stack(center_targets)
        valid_gt_masks = torch.stack(valid_gt_masks)

        positive_mask = torch.stack(positive_mask)
        negative_mask = torch.stack(negative_mask)
        # objectness_weights /= (torch.sum(objectness_weights) + 1e-6)
        # box_loss_weights = objectness_targets.float() / (
        #     torch.sum(objectness_targets).float() + 1e-6)
        # valid_gt_weights = valid_gt_masks.float() / (
        #     torch.sum(valid_gt_masks.float()) + 1e-6)
        dir_class_targets = torch.stack(dir_class_targets)
        dir_res_targets = torch.stack(dir_res_targets)
        size_res_targets = torch.stack(size_res_targets)
        mask_targets = torch.stack(mask_targets)
        centerness_targets = torch.stack(centerness_targets)

        return (center_targets, size_res_targets, dir_class_targets,
                dir_res_targets, mask_targets, centerness_targets,
                positive_mask, negative_mask)

    def get_targets_single(self,
                           points,
                           gt_bboxes_3d,
                           gt_labels_3d,
                           pts_semantic_mask=None,
                           pts_instance_mask=None,
                           aggregated_points=None):
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
                vote aggregation layer.

        Returns:
            tuple[torch.Tensor]: Targets of vote head.
        """
        assert self.bbox_coder.with_rot or pts_semantic_mask is not None

        gt_bboxes_3d = gt_bboxes_3d.to(points.device)
        # import pdb
        # pdb.set_trace()
        valid_gt = gt_labels_3d != -1
        gt_bboxes_3d = gt_bboxes_3d[valid_gt]
        gt_labels_3d = gt_labels_3d[valid_gt]

        (center_targets, size_targets, dir_class_targets,
         dir_res_targets) = self.bbox_coder.encode(gt_bboxes_3d, gt_labels_3d)

        num_gt = gt_bboxes_3d.tensor.shape[0]
        if isinstance(gt_bboxes_3d, LiDARInstance3DBoxes):
            assignment = gt_bboxes_3d.points_in_boxes(aggregated_points).long()
            points_mask = assignment.new_zeros(
                [assignment.shape[0], num_gt + 1])
            assignment[assignment == -1] = num_gt
            assert assignment.max() < num_gt + 1
            assert assignment.min() >= 0
            points_mask.scatter_(1, assignment.unsqueeze(1), 1)
            points_mask = points_mask[:, :-1]
            assignment[assignment == num_gt] = num_gt - 1
        elif isinstance(gt_bboxes_3d, DepthInstance3DBoxes):
            points_mask = gt_bboxes_3d.points_in_boxes(aggregated_points)
            assignment = points_mask.argmax(dim=-1)
        else:
            raise NotImplementedError('Unsupported bbox type!')

        center_targets = center_targets[assignment]
        size_res_targets = size_targets[assignment]
        mask_targets = gt_labels_3d[assignment]
        dir_class_targets = dir_class_targets[assignment]
        dir_res_targets = dir_res_targets[assignment]

        dist = torch.norm(aggregated_points - center_targets, dim=1)
        dist_mask = dist < 10
        positive_mask = (points_mask.max(1)[0] > 0) * dist_mask
        negative_mask = (points_mask.max(1)[0] == 0)

        canonical_xyz = aggregated_points - center_targets
        canonical_xyz = rotation_3d_in_axis(
            canonical_xyz.unsqueeze(0).transpose(0, 1),
            gt_bboxes_3d.yaw[assignment], 2).squeeze(1)
        distance_front = size_res_targets[:, 0] - canonical_xyz[:, 0]
        distance_back = size_res_targets[:, 0] + canonical_xyz[:, 0]
        distance_left = size_res_targets[:, 1] + canonical_xyz[:, 1]
        distance_right = size_res_targets[:, 1] - canonical_xyz[:, 1]
        distance_bottom = canonical_xyz[:, 2]
        distance_top = size_res_targets[:, 2] * 2 - canonical_xyz[:, 2]
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
        assert mask_targets.max() < self.num_classes
        if mask_targets.min() < 0:
            import pdb
            pdb.set_trace()
        one_hot_centerness_targets.scatter_(1, mask_targets.unsqueeze(-1), 1)
        centerness_targets = centerness_targets.unsqueeze(
            1) * one_hot_centerness_targets

        return (center_targets, size_res_targets, dir_class_targets,
                dir_res_targets, mask_targets, centerness_targets,
                positive_mask, negative_mask)

    def get_bboxes(self, points, bbox_preds, input_metas, rescale=False):
        """Generate bboxes from vote head predictions.

        Args:
            points (torch.Tensor): Input points.
            bbox_preds (dict): Predictions from vote head.
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
                bbox_selected,
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
            bbox,
            box_dim=bbox.shape[-1],
            with_yaw=self.bbox_coder.with_rot,
            origin=(0.5, 0.5, 0))

        if isinstance(bbox, LiDARInstance3DBoxes):
            box_idx = bbox.points_in_boxes(points)
            box_indices = box_idx.new_zeros([num_bbox + 1])
            box_idx[box_idx == -1] = num_bbox
            box_indices.scatter_add_(0, box_idx.long(),
                                     box_idx.new_ones(box_idx.shape))
            box_indices = box_indices[:-1]
            nonempty_box_mask = box_indices > 5
        elif isinstance(bbox, DepthInstance3DBoxes):
            box_indices = bbox.points_in_boxes(points)
            nonempty_box_mask = box_indices.T.sum(1) > 5
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
