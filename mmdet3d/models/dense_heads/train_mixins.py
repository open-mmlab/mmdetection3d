# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmdet.models.utils import images_to_levels, multi_apply
from mmengine.structures import InstanceData

from mmdet3d.structures import limit_period


class AnchorTrainMixin(object):
    """Mixin class for target assigning of dense heads."""

    def anchor_target_3d(self,
                         anchor_list,
                         batch_gt_instances_3d,
                         batch_input_metas,
                         batch_gt_instances_ignore=None,
                         label_channels=1,
                         num_classes=1,
                         sampling=True):
        """Compute regression and classification targets for anchors.

        Args:
            anchor_list (list[list]): Multi level anchors of each image.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Ground truth
                bboxes of each image.
            batch_input_metas (list[dict]): Meta info of each image.
            batch_gt_instances_ignore (list): Ignore list of gt bboxes.
            label_channels (int): The channel of labels.
            num_classes (int): The number of classes.
            sampling (bool): Whether to sample anchors.

        Returns:
            tuple (list, list, list, list, list, list, int, int):
                Anchor targets, including labels, label weights,
                bbox targets, bbox weights, direction targets,
                direction weights, number of positive anchors and
                number of negative anchors.
        """
        num_inputs = len(batch_input_metas)
        assert len(anchor_list) == num_inputs

        if isinstance(anchor_list[0][0], list):
            # sizes of anchors are different
            # anchor number of a single level
            num_level_anchors = [
                sum([anchor.size(0) for anchor in anchors])
                for anchors in anchor_list[0]
            ]
            for i in range(num_inputs):
                anchor_list[i] = anchor_list[i][0]
        else:
            # anchor number of multi levels
            num_level_anchors = [
                anchors.view(-1, self.box_code_size).size(0)
                for anchors in anchor_list[0]
            ]
            # concat all level anchors and flags to a single tensor
            for i in range(num_inputs):
                anchor_list[i] = torch.cat(anchor_list[i])

        # compute targets for each image
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None for _ in range(num_inputs)]

        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         all_dir_targets, all_dir_weights, pos_inds_list,
         neg_inds_list) = multi_apply(
             self.anchor_target_3d_single,
             anchor_list,
             batch_gt_instances_3d,
             batch_gt_instances_ignore,
             batch_input_metas,
             label_channels=label_channels,
             num_classes=num_classes,
             sampling=sampling)

        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        dir_targets_list = images_to_levels(all_dir_targets, num_level_anchors)
        dir_weights_list = images_to_levels(all_dir_weights, num_level_anchors)
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, dir_targets_list, dir_weights_list,
                num_total_pos, num_total_neg)

    def anchor_target_3d_single(self,
                                anchors,
                                gt_instance_3d,
                                gt_instance_ignore,
                                input_meta,
                                label_channels=1,
                                num_classes=1,
                                sampling=True):
        """Compute targets of anchors in single batch.

        Args:
            anchors (torch.Tensor): Concatenated multi-level anchor.
            gt_instance_3d (:obj:`InstanceData`): Gt bboxes.
            gt_instance_ignore (:obj:`InstanceData`): Ignored gt bboxes.
            input_meta (dict): Meta info of each image.
            label_channels (int): The channel of labels.
            num_classes (int): The number of classes.
            sampling (bool): Whether to sample anchors.

        Returns:
            tuple[torch.Tensor]: Anchor targets.
        """
        if isinstance(self.bbox_assigner,
                      list) and (not isinstance(anchors, list)):
            feat_size = anchors.size(0) * anchors.size(1) * anchors.size(2)
            rot_angles = anchors.size(-2)
            assert len(self.bbox_assigner) == anchors.size(-3)
            (total_labels, total_label_weights, total_bbox_targets,
             total_bbox_weights, total_dir_targets, total_dir_weights,
             total_pos_inds, total_neg_inds) = [], [], [], [], [], [], [], []
            current_anchor_num = 0
            for i, assigner in enumerate(self.bbox_assigner):
                current_anchors = anchors[..., i, :, :].reshape(
                    -1, self.box_code_size)
                current_anchor_num += current_anchors.size(0)
                if self.assign_per_class:
                    gt_per_cls = (gt_instance_3d.labels_3d == i)
                    gt_per_cls_instance = InstanceData()
                    gt_per_cls_instance.labels_3d = gt_instance_3d.labels_3d[
                        gt_per_cls]
                    gt_per_cls_instance.bboxes_3d = gt_instance_3d.bboxes_3d[
                        gt_per_cls, :]
                    anchor_targets = self.anchor_target_single_assigner(
                        assigner, current_anchors, gt_per_cls_instance,
                        gt_instance_ignore, input_meta, num_classes, sampling)
                else:
                    anchor_targets = self.anchor_target_single_assigner(
                        assigner, current_anchors, gt_instance_3d,
                        gt_instance_ignore, input_meta, num_classes, sampling)

                (labels, label_weights, bbox_targets, bbox_weights,
                 dir_targets, dir_weights, pos_inds, neg_inds) = anchor_targets
                total_labels.append(labels.reshape(feat_size, 1, rot_angles))
                total_label_weights.append(
                    label_weights.reshape(feat_size, 1, rot_angles))
                total_bbox_targets.append(
                    bbox_targets.reshape(feat_size, 1, rot_angles,
                                         anchors.size(-1)))
                total_bbox_weights.append(
                    bbox_weights.reshape(feat_size, 1, rot_angles,
                                         anchors.size(-1)))
                total_dir_targets.append(
                    dir_targets.reshape(feat_size, 1, rot_angles))
                total_dir_weights.append(
                    dir_weights.reshape(feat_size, 1, rot_angles))
                total_pos_inds.append(pos_inds)
                total_neg_inds.append(neg_inds)

            total_labels = torch.cat(total_labels, dim=-2).reshape(-1)
            total_label_weights = torch.cat(
                total_label_weights, dim=-2).reshape(-1)
            total_bbox_targets = torch.cat(
                total_bbox_targets, dim=-3).reshape(-1, anchors.size(-1))
            total_bbox_weights = torch.cat(
                total_bbox_weights, dim=-3).reshape(-1, anchors.size(-1))
            total_dir_targets = torch.cat(
                total_dir_targets, dim=-2).reshape(-1)
            total_dir_weights = torch.cat(
                total_dir_weights, dim=-2).reshape(-1)
            total_pos_inds = torch.cat(total_pos_inds, dim=0).reshape(-1)
            total_neg_inds = torch.cat(total_neg_inds, dim=0).reshape(-1)
            return (total_labels, total_label_weights, total_bbox_targets,
                    total_bbox_weights, total_dir_targets, total_dir_weights,
                    total_pos_inds, total_neg_inds)
        elif isinstance(self.bbox_assigner, list) and isinstance(
                anchors, list):
            # class-aware anchors with different feature map sizes
            assert len(self.bbox_assigner) == len(anchors), \
                'The number of bbox assigners and anchors should be the same.'
            (total_labels, total_label_weights, total_bbox_targets,
             total_bbox_weights, total_dir_targets, total_dir_weights,
             total_pos_inds, total_neg_inds) = [], [], [], [], [], [], [], []
            current_anchor_num = 0
            for i, assigner in enumerate(self.bbox_assigner):
                current_anchors = anchors[i]
                current_anchor_num += current_anchors.size(0)
                if self.assign_per_class:
                    gt_per_cls = (gt_instance_3d.labels_3d == i)
                    gt_per_cls_instance = InstanceData()
                    gt_per_cls_instance.labels_3d = gt_instance_3d.labels_3d[
                        gt_per_cls]
                    gt_per_cls_instance.bboxes_3d = gt_instance_3d.bboxes_3d[
                        gt_per_cls, :]
                    anchor_targets = self.anchor_target_single_assigner(
                        assigner, current_anchors, gt_per_cls_instance,
                        gt_instance_ignore, input_meta, num_classes, sampling)
                else:
                    anchor_targets = self.anchor_target_single_assigner(
                        assigner, current_anchors, gt_instance_3d,
                        gt_instance_ignore, input_meta, num_classes, sampling)

                (labels, label_weights, bbox_targets, bbox_weights,
                 dir_targets, dir_weights, pos_inds, neg_inds) = anchor_targets
                total_labels.append(labels)
                total_label_weights.append(label_weights)
                total_bbox_targets.append(
                    bbox_targets.reshape(-1, anchors[i].size(-1)))
                total_bbox_weights.append(
                    bbox_weights.reshape(-1, anchors[i].size(-1)))
                total_dir_targets.append(dir_targets)
                total_dir_weights.append(dir_weights)
                total_pos_inds.append(pos_inds)
                total_neg_inds.append(neg_inds)

            total_labels = torch.cat(total_labels, dim=0)
            total_label_weights = torch.cat(total_label_weights, dim=0)
            total_bbox_targets = torch.cat(total_bbox_targets, dim=0)
            total_bbox_weights = torch.cat(total_bbox_weights, dim=0)
            total_dir_targets = torch.cat(total_dir_targets, dim=0)
            total_dir_weights = torch.cat(total_dir_weights, dim=0)
            total_pos_inds = torch.cat(total_pos_inds, dim=0)
            total_neg_inds = torch.cat(total_neg_inds, dim=0)
            return (total_labels, total_label_weights, total_bbox_targets,
                    total_bbox_weights, total_dir_targets, total_dir_weights,
                    total_pos_inds, total_neg_inds)
        else:
            return self.anchor_target_single_assigner(self.bbox_assigner,
                                                      anchors, gt_instance_3d,
                                                      gt_instance_ignore,
                                                      input_meta, num_classes,
                                                      sampling)

    def anchor_target_single_assigner(self,
                                      bbox_assigner,
                                      anchors,
                                      gt_instance_3d,
                                      gt_instance_ignore,
                                      input_meta,
                                      num_classes=1,
                                      sampling=True):
        """Assign anchors and encode positive anchors.

        Args:
            bbox_assigner (BaseAssigner): assign positive and negative boxes.
            anchors (torch.Tensor): Concatenated multi-level anchor.
            gt_instance_3d (:obj:`InstanceData`): Gt bboxes.
            gt_instance_ignore (torch.Tensor): Ignored gt bboxes.
            input_meta (dict): Meta info of each image.
            num_classes (int): The number of classes.
            sampling (bool): Whether to sample anchors.

        Returns:
            tuple[torch.Tensor]: Anchor targets.
        """
        anchors = anchors.reshape(-1, anchors.size(-1))
        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        dir_targets = anchors.new_zeros((anchors.shape[0]), dtype=torch.long)
        dir_weights = anchors.new_zeros((anchors.shape[0]), dtype=torch.float)
        labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        if len(gt_instance_3d.bboxes_3d) > 0:
            if not isinstance(gt_instance_3d.bboxes_3d, torch.Tensor):
                gt_instance_3d.bboxes_3d = gt_instance_3d.bboxes_3d.tensor.to(
                    anchors.device)
            pred_instance_3d = InstanceData(priors=anchors)
            assign_result = bbox_assigner.assign(pred_instance_3d,
                                                 gt_instance_3d,
                                                 gt_instance_ignore)
            sampling_result = self.bbox_sampler.sample(assign_result,
                                                       pred_instance_3d,
                                                       gt_instance_3d)
            pos_inds = sampling_result.pos_inds
            neg_inds = sampling_result.neg_inds
        else:
            pos_inds = torch.nonzero(
                anchors.new_zeros((anchors.shape[0], ), dtype=torch.bool) > 0,
                as_tuple=False).squeeze(-1).unique()
            neg_inds = torch.nonzero(
                anchors.new_zeros((anchors.shape[0], ), dtype=torch.bool) == 0,
                as_tuple=False).squeeze(-1).unique()

        if gt_instance_3d.labels_3d is not None:
            labels += num_classes
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(
                sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            pos_dir_targets = get_direction_target(
                sampling_result.pos_bboxes,
                pos_bbox_targets,
                self.dir_offset,
                self.dir_limit_offset,
                one_hot=False)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            dir_targets[pos_inds] = pos_dir_targets
            dir_weights[pos_inds] = 1.0

            if gt_instance_3d.labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_instance_3d.labels_3d[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        return (labels, label_weights, bbox_targets, bbox_weights, dir_targets,
                dir_weights, pos_inds, neg_inds)


def get_direction_target(anchors,
                         reg_targets,
                         dir_offset=0,
                         dir_limit_offset=0,
                         num_bins=2,
                         one_hot=True):
    """Encode direction to 0 ~ num_bins-1.

    Args:
        anchors (torch.Tensor): Concatenated multi-level anchor.
        reg_targets (torch.Tensor): Bbox regression targets.
        dir_offset (int): Direction offset.
        num_bins (int): Number of bins to divide 2*PI.
        one_hot (bool): Whether to encode as one hot.

    Returns:
        torch.Tensor: Encoded direction targets.
    """
    rot_gt = reg_targets[..., 6] + anchors[..., 6]
    offset_rot = limit_period(rot_gt - dir_offset, dir_limit_offset, 2 * np.pi)
    dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
    dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
    if one_hot:
        dir_targets = torch.zeros(
            *list(dir_cls_targets.shape),
            num_bins,
            dtype=anchors.dtype,
            device=dir_cls_targets.device)
        dir_targets.scatter_(dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
        dir_cls_targets = dir_targets
    return dir_cls_targets
