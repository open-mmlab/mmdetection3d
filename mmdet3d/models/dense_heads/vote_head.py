import numpy as np
import torch
from mmcv.runner import force_fp32
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.core.post_processing import aligned_3d_nms
from mmdet3d.models.builder import build_loss
from mmdet3d.models.losses import chamfer_distance
from mmdet3d.models.model_utils import VoteModule
from mmdet3d.ops import build_sa_module, furthest_point_sample
from mmdet.core import build_bbox_coder, multi_apply
from mmdet.models import HEADS
from .base_conv_bbox_head import BaseConvBboxHead


@HEADS.register_module()
class VoteHead(nn.Module):
    r"""Bbox head of `Votenet <https://arxiv.org/abs/1904.09664>`_.

    Args:
        num_classes (int): The number of class.
        bbox_coder (:obj:`BaseBBoxCoder`): Bbox coder for encoding and
            decoding boxes.
        train_cfg (dict): Config for training.
        test_cfg (dict): Config for testing.
        vote_module_cfg (dict): Config of VoteModule for point-wise votes.
        vote_aggregation_cfg (dict): Config of vote aggregation layer.
        pred_layer_cfg (dict): Config of classfication and regression
            prediction layers.
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
                 train_cfg=None,
                 test_cfg=None,
                 vote_module_cfg=None,
                 vote_aggregation_cfg=None,
                 pred_layer_cfg=None,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 objectness_loss=None,
                 center_loss=None,
                 dir_class_loss=None,
                 dir_res_loss=None,
                 size_class_loss=None,
                 size_res_loss=None,
                 semantic_loss=None):
        super(VoteHead, self).__init__()
        self.num_classes = num_classes
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.gt_per_seed = vote_module_cfg['gt_per_seed']
        self.num_proposal = vote_aggregation_cfg['num_point']

        self.objectness_loss = build_loss(objectness_loss)
        self.center_loss = build_loss(center_loss)
        self.dir_res_loss = build_loss(dir_res_loss)
        self.dir_class_loss = build_loss(dir_class_loss)
        self.size_res_loss = build_loss(size_res_loss)
        if size_class_loss is not None:
            self.size_class_loss = build_loss(size_class_loss)
        if semantic_loss is not None:
            self.semantic_loss = build_loss(semantic_loss)

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.num_sizes = self.bbox_coder.num_sizes
        self.num_dir_bins = self.bbox_coder.num_dir_bins

        self.vote_module = VoteModule(**vote_module_cfg)
        self.vote_aggregation = build_sa_module(vote_aggregation_cfg)
        self.fp16_enabled = False

        # Bbox classification and regression
        self.conv_pred = BaseConvBboxHead(
            **pred_layer_cfg,
            num_cls_out_channels=self._get_cls_out_channels(),
            num_reg_out_channels=self._get_reg_out_channels())

    def init_weights(self):
        """Initialize weights of VoteHead."""
        pass

    def _get_cls_out_channels(self):
        """Return the channel number of classification outputs."""
        # Class numbers (k) + objectness (2)
        return self.num_classes + 2

    def _get_reg_out_channels(self):
        """Return the channel number of regression outputs."""
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_dir_bins*2),
        # size class+residual(num_sizes*4)
        return 3 + self.num_dir_bins * 2 + self.num_sizes * 4

    def _extract_input(self, feat_dict):
        """Extract inputs from features dictionary.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            torch.Tensor: Coordinates of input points.
            torch.Tensor: Features of input points.
            torch.Tensor: Indices of input points.
        """
        seed_points = feat_dict['fp_xyz'][-1]
        seed_features = feat_dict['fp_features'][-1]
        seed_indices = feat_dict['fp_indices'][-1]

        return seed_points, seed_features, seed_indices

    def forward(self, feat_dict, sample_mod):
        """Forward pass.

        Note:
            The forward of VoteHead is devided into 4 steps:

                1. Generate vote_points from seed_points.
                2. Aggregate vote_points.
                3. Predict bbox and score.
                4. Decode predictions.

        Args:
            feat_dict (dict): Feature dict from backbone.
            sample_mod (str): Sample mode for vote aggregation layer.
                valid modes are "vote", "seed", "random" and "spec".

        Returns:
            dict: Predictions of vote head.
        """
        assert sample_mod in ['vote', 'seed', 'random', 'spec']

        seed_points, seed_features, seed_indices = self._extract_input(
            feat_dict)

        # 1. generate vote_points from seed_points
        vote_points, vote_features, vote_offset = self.vote_module(
            seed_points, seed_features)
        results = dict(
            seed_points=seed_points,
            seed_indices=seed_indices,
            vote_points=vote_points,
            vote_features=vote_features,
            vote_offset=vote_offset)

        # 2. aggregate vote_points
        if sample_mod == 'vote':
            # use fps in vote_aggregation
            aggregation_inputs = dict(
                points_xyz=vote_points, features=vote_features)
        elif sample_mod == 'seed':
            # FPS on seed and choose the votes corresponding to the seeds
            sample_indices = furthest_point_sample(seed_points,
                                                   self.num_proposal)
            aggregation_inputs = dict(
                points_xyz=vote_points,
                features=vote_features,
                indices=sample_indices)
        elif sample_mod == 'random':
            # Random sampling from the votes
            batch_size, num_seed = seed_points.shape[:2]
            sample_indices = seed_points.new_tensor(
                torch.randint(0, num_seed, (batch_size, self.num_proposal)),
                dtype=torch.int32)
            aggregation_inputs = dict(
                points_xyz=vote_points,
                features=vote_features,
                indices=sample_indices)
        elif sample_mod == 'spec':
            # Specify the new center in vote_aggregation
            aggregation_inputs = dict(
                points_xyz=seed_points,
                features=seed_features,
                target_xyz=vote_points)
        else:
            raise NotImplementedError(
                f'Sample mode {sample_mod} is not supported!')

        vote_aggregation_ret = self.vote_aggregation(**aggregation_inputs)
        aggregated_points, features, aggregated_indices = vote_aggregation_ret

        results['aggregated_points'] = aggregated_points
        results['aggregated_features'] = features
        results['aggregated_indices'] = aggregated_indices

        # 3. predict bbox and score
        cls_predictions, reg_predictions = self.conv_pred(features)

        # 4. decode predictions
        decode_res = self.bbox_coder.split_pred(cls_predictions,
                                                reg_predictions,
                                                aggregated_points)

        results.update(decode_res)

        return results

    @force_fp32(apply_to=('bbox_preds', ))
    def loss(self,
             bbox_preds,
             points,
             gt_bboxes_3d,
             gt_labels_3d,
             pts_semantic_mask=None,
             pts_instance_mask=None,
             img_metas=None,
             gt_bboxes_ignore=None,
             ret_target=False):
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
            ret_target (Bool): Return targets or not.

        Returns:
            dict: Losses of Votenet.
        """
        targets = self.get_targets(points, gt_bboxes_3d, gt_labels_3d,
                                   pts_semantic_mask, pts_instance_mask,
                                   bbox_preds)
        (vote_targets, vote_target_masks, size_class_targets, size_res_targets,
         dir_class_targets, dir_res_targets, center_targets, mask_targets,
         valid_gt_masks, objectness_targets, objectness_weights,
         box_loss_weights, valid_gt_weights) = targets

        # calculate vote loss
        vote_loss = self.vote_module.get_loss(bbox_preds['seed_points'],
                                              bbox_preds['vote_points'],
                                              bbox_preds['seed_indices'],
                                              vote_target_masks, vote_targets)

        # calculate objectness loss
        objectness_loss = self.objectness_loss(
            bbox_preds['obj_scores'].transpose(2, 1),
            objectness_targets,
            weight=objectness_weights)

        # calculate center loss
        source2target_loss, target2source_loss = self.center_loss(
            bbox_preds['center'],
            center_targets,
            src_weight=box_loss_weights,
            dst_weight=valid_gt_weights)
        center_loss = source2target_loss + target2source_loss

        # calculate direction class loss
        dir_class_loss = self.dir_class_loss(
            bbox_preds['dir_class'].transpose(2, 1),
            dir_class_targets,
            weight=box_loss_weights)

        # calculate direction residual loss
        batch_size, proposal_num = size_class_targets.shape[:2]
        heading_label_one_hot = vote_targets.new_zeros(
            (batch_size, proposal_num, self.num_dir_bins))
        heading_label_one_hot.scatter_(2, dir_class_targets.unsqueeze(-1), 1)
        dir_res_norm = torch.sum(
            bbox_preds['dir_res_norm'] * heading_label_one_hot, -1)
        dir_res_loss = self.dir_res_loss(
            dir_res_norm, dir_res_targets, weight=box_loss_weights)

        # calculate size class loss
        size_class_loss = self.size_class_loss(
            bbox_preds['size_class'].transpose(2, 1),
            size_class_targets,
            weight=box_loss_weights)

        # calculate size residual loss
        one_hot_size_targets = vote_targets.new_zeros(
            (batch_size, proposal_num, self.num_sizes))
        one_hot_size_targets.scatter_(2, size_class_targets.unsqueeze(-1), 1)
        one_hot_size_targets_expand = one_hot_size_targets.unsqueeze(
            -1).repeat(1, 1, 1, 3).contiguous()
        size_residual_norm = torch.sum(
            bbox_preds['size_res_norm'] * one_hot_size_targets_expand, 2)
        box_loss_weights_expand = box_loss_weights.unsqueeze(-1).repeat(
            1, 1, 3)
        size_res_loss = self.size_res_loss(
            size_residual_norm,
            size_res_targets,
            weight=box_loss_weights_expand)

        # calculate semantic loss
        semantic_loss = self.semantic_loss(
            bbox_preds['sem_scores'].transpose(2, 1),
            mask_targets,
            weight=box_loss_weights)

        losses = dict(
            vote_loss=vote_loss,
            objectness_loss=objectness_loss,
            semantic_loss=semantic_loss,
            center_loss=center_loss,
            dir_class_loss=dir_class_loss,
            dir_res_loss=dir_res_loss,
            size_class_loss=size_class_loss,
            size_res_loss=size_res_loss)

        if ret_target:
            losses['targets'] = targets

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
            bbox_preds['aggregated_points'][i]
            for i in range(len(gt_labels_3d))
        ]

        (vote_targets, vote_target_masks, size_class_targets, size_res_targets,
         dir_class_targets, dir_res_targets, center_targets, mask_targets,
         objectness_targets, objectness_masks) = multi_apply(
             self.get_targets_single, points, gt_bboxes_3d, gt_labels_3d,
             pts_semantic_mask, pts_instance_mask, aggregated_points)

        # pad targets as original code of votenet.
        for index in range(len(gt_labels_3d)):
            pad_num = max_gt_num - gt_labels_3d[index].shape[0]
            center_targets[index] = F.pad(center_targets[index],
                                          (0, 0, 0, pad_num))
            valid_gt_masks[index] = F.pad(valid_gt_masks[index], (0, pad_num))

        vote_targets = torch.stack(vote_targets)
        vote_target_masks = torch.stack(vote_target_masks)
        center_targets = torch.stack(center_targets)
        valid_gt_masks = torch.stack(valid_gt_masks)

        objectness_targets = torch.stack(objectness_targets)
        objectness_weights = torch.stack(objectness_masks)
        objectness_weights /= (torch.sum(objectness_weights) + 1e-6)
        box_loss_weights = objectness_targets.float() / (
            torch.sum(objectness_targets).float() + 1e-6)
        valid_gt_weights = valid_gt_masks.float() / (
            torch.sum(valid_gt_masks.float()) + 1e-6)
        dir_class_targets = torch.stack(dir_class_targets)
        dir_res_targets = torch.stack(dir_res_targets)
        size_class_targets = torch.stack(size_class_targets)
        size_res_targets = torch.stack(size_res_targets)
        mask_targets = torch.stack(mask_targets)

        return (vote_targets, vote_target_masks, size_class_targets,
                size_res_targets, dir_class_targets, dir_res_targets,
                center_targets, mask_targets, valid_gt_masks,
                objectness_targets, objectness_weights, box_loss_weights,
                valid_gt_weights)

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

        # generate votes target
        num_points = points.shape[0]
        if self.bbox_coder.with_rot:
            vote_targets = points.new_zeros([num_points, 3 * self.gt_per_seed])
            vote_target_masks = points.new_zeros([num_points],
                                                 dtype=torch.long)
            vote_target_idx = points.new_zeros([num_points], dtype=torch.long)
            box_indices_all = gt_bboxes_3d.points_in_boxes(points)
            for i in range(gt_labels_3d.shape[0]):
                box_indices = box_indices_all[:, i]
                indices = torch.nonzero(
                    box_indices, as_tuple=False).squeeze(-1)
                selected_points = points[indices]
                vote_target_masks[indices] = 1
                vote_targets_tmp = vote_targets[indices]
                votes = gt_bboxes_3d.gravity_center[i].unsqueeze(
                    0) - selected_points[:, :3]

                for j in range(self.gt_per_seed):
                    column_indices = torch.nonzero(
                        vote_target_idx[indices] == j,
                        as_tuple=False).squeeze(-1)
                    vote_targets_tmp[column_indices,
                                     int(j * 3):int(j * 3 +
                                                    3)] = votes[column_indices]
                    if j == 0:
                        vote_targets_tmp[column_indices] = votes[
                            column_indices].repeat(1, self.gt_per_seed)

                vote_targets[indices] = vote_targets_tmp
                vote_target_idx[indices] = torch.clamp(
                    vote_target_idx[indices] + 1, max=2)
        elif pts_semantic_mask is not None:
            vote_targets = points.new_zeros([num_points, 3])
            vote_target_masks = points.new_zeros([num_points],
                                                 dtype=torch.long)

            for i in torch.unique(pts_instance_mask):
                indices = torch.nonzero(
                    pts_instance_mask == i, as_tuple=False).squeeze(-1)
                if pts_semantic_mask[indices[0]] < self.num_classes:
                    selected_points = points[indices, :3]
                    center = 0.5 * (
                        selected_points.min(0)[0] + selected_points.max(0)[0])
                    vote_targets[indices, :] = center - selected_points
                    vote_target_masks[indices] = 1
            vote_targets = vote_targets.repeat((1, self.gt_per_seed))
        else:
            raise NotImplementedError

        (center_targets, size_class_targets, size_res_targets,
         dir_class_targets,
         dir_res_targets) = self.bbox_coder.encode(gt_bboxes_3d, gt_labels_3d)

        proposal_num = aggregated_points.shape[0]
        distance1, _, assignment, _ = chamfer_distance(
            aggregated_points.unsqueeze(0),
            center_targets.unsqueeze(0),
            reduction='none')
        assignment = assignment.squeeze(0)
        euclidean_distance1 = torch.sqrt(distance1.squeeze(0) + 1e-6)

        objectness_targets = points.new_zeros((proposal_num), dtype=torch.long)
        objectness_targets[
            euclidean_distance1 < self.train_cfg['pos_distance_thr']] = 1

        objectness_masks = points.new_zeros((proposal_num))
        objectness_masks[
            euclidean_distance1 < self.train_cfg['pos_distance_thr']] = 1.0
        objectness_masks[
            euclidean_distance1 > self.train_cfg['neg_distance_thr']] = 1.0

        dir_class_targets = dir_class_targets[assignment]
        dir_res_targets = dir_res_targets[assignment]
        dir_res_targets /= (np.pi / self.num_dir_bins)
        size_class_targets = size_class_targets[assignment]
        size_res_targets = size_res_targets[assignment]

        one_hot_size_targets = gt_bboxes_3d.tensor.new_zeros(
            (proposal_num, self.num_sizes))
        one_hot_size_targets.scatter_(1, size_class_targets.unsqueeze(-1), 1)
        one_hot_size_targets = one_hot_size_targets.unsqueeze(-1).repeat(
            1, 1, 3)
        mean_sizes = size_res_targets.new_tensor(
            self.bbox_coder.mean_sizes).unsqueeze(0)
        pos_mean_sizes = torch.sum(one_hot_size_targets * mean_sizes, 1)
        size_res_targets /= pos_mean_sizes

        mask_targets = gt_labels_3d[assignment]

        return (vote_targets, vote_target_masks, size_class_targets,
                size_res_targets,
                dir_class_targets, dir_res_targets, center_targets,
                mask_targets.long(), objectness_targets, objectness_masks)

    def get_bboxes(self,
                   points,
                   bbox_preds,
                   input_metas,
                   rescale=False,
                   use_nms=True):
        """Generate bboxes from vote head predictions.

        Args:
            points (torch.Tensor): Input points.
            bbox_preds (dict): Predictions from vote head.
            input_metas (list[dict]): Point cloud and image's meta info.
            rescale (bool): Whether to rescale bboxes.
            use_nms (bool): Whether to apply NMS, skip nms postprocessing
                while using vote head in rpn stage.

        Returns:
            list[tuple[torch.Tensor]]: Bounding boxes, scores and labels.
        """
        # decode boxes
        obj_scores = F.softmax(bbox_preds['obj_scores'], dim=-1)[..., -1]
        sem_scores = F.softmax(bbox_preds['sem_scores'], dim=-1)
        bbox3d = self.bbox_coder.decode(bbox_preds)

        if use_nms:
            batch_size = bbox3d.shape[0]
            results = list()
            for b in range(batch_size):
                bbox_selected, score_selected, labels = \
                    self.multiclass_nms_single(obj_scores[b], sem_scores[b],
                                               bbox3d[b], points[b, ..., :3],
                                               input_metas[b])
                bbox = input_metas[b]['box_type_3d'](
                    bbox_selected,
                    box_dim=bbox_selected.shape[-1],
                    with_yaw=self.bbox_coder.with_rot)
                results.append((bbox, score_selected, labels))

            return results
        else:
            return bbox3d

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
        bbox = input_meta['box_type_3d'](
            bbox,
            box_dim=bbox.shape[-1],
            with_yaw=self.bbox_coder.with_rot,
            origin=(0.5, 0.5, 0.5))
        box_indices = bbox.points_in_boxes(points)

        corner3d = bbox.corners
        minmax_box3d = corner3d.new(torch.Size((corner3d.shape[0], 6)))
        minmax_box3d[:, :3] = torch.min(corner3d, dim=1)[0]
        minmax_box3d[:, 3:] = torch.max(corner3d, dim=1)[0]

        nonempty_box_mask = box_indices.T.sum(1) > 5

        bbox_classes = torch.argmax(sem_scores, -1)
        nms_selected = aligned_3d_nms(minmax_box3d[nonempty_box_mask],
                                      obj_scores[nonempty_box_mask],
                                      bbox_classes[nonempty_box_mask],
                                      self.test_cfg.nms_thr)

        # filter empty boxes and boxes with low score
        scores_mask = (obj_scores > self.test_cfg.score_thr)
        nonempty_box_inds = torch.nonzero(
            nonempty_box_mask, as_tuple=False).flatten()
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
