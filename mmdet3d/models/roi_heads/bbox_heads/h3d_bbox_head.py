import torch
from mmcv.cnn import ConvModule
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.core.bbox import DepthInstance3DBoxes
from mmdet3d.core.post_processing import aligned_3d_nms
from mmdet3d.models.builder import build_loss
from mmdet3d.models.losses import chamfer_distance
from mmdet3d.ops import PointSAModule
from mmdet.core import build_bbox_coder, multi_apply
from mmdet.models import HEADS


@HEADS.register_module()
class H3DBboxHead(nn.Module):
    r"""Bbox head of `H3DNet <https://arxiv.org/abs/2006.05682>`_.

    Args:
        num_classes (int): The number of classes.
        bbox_coder (:obj:`BaseBBoxCoder`): Bbox coder for encoding and
            decoding boxes.
        train_cfg (dict): Config for training.
        test_cfg (dict): Config for testing.
        gt_per_seed (int): Number of ground truth votes generated
            from each seed point.
        num_proposal (int): Number of proposal votes generated.
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
                 train_cfg=None,
                 test_cfg=None,
                 proposal_module_cfg=None,
                 gt_per_seed=1,
                 num_proposal=256,
                 feat_channels=(128, 128),
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 objectness_loss=None,
                 center_loss=None,
                 dir_class_loss=None,
                 dir_res_loss=None,
                 size_class_loss=None,
                 size_res_loss=None,
                 semantic_loss=None):
        super(H3DBboxHead, self).__init__()
        self.num_classes = num_classes
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.gt_per_seed = gt_per_seed
        self.num_proposal = num_proposal

        self.objectness_loss = build_loss(objectness_loss)
        self.center_loss = build_loss(center_loss)
        self.dir_class_loss = build_loss(dir_class_loss)
        self.dir_res_loss = build_loss(dir_res_loss)
        self.size_class_loss = build_loss(size_class_loss)
        self.size_res_loss = build_loss(size_res_loss)
        self.semantic_loss = build_loss(semantic_loss)

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.num_sizes = self.bbox_coder.num_sizes
        self.num_dir_bins = self.bbox_coder.num_dir_bins

        self.refine_module = ProposalRefineModule(
            num_classes=num_classes,
            num_heading_bin=bbox_coder['num_dir_bins'],
            num_size_cluster=bbox_coder['num_sizes'],
            mean_size_arr=bbox_coder['mean_sizes'],
            num_proposal=num_proposal,
            with_angle=bbox_coder['with_rot'],
            **proposal_module_cfg)

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        pass

    def forward(self, feat_dict, sample_mod):
        """Forward pass.

        Args:
            feat_dict (dict): Feature dict from backbone.
            sample_mod (str): Sample mode for vote aggregation layer.
                valid modes are "vote", "seed" and "random".

        Returns:
            dict: Predictions of vote head.
        """
        aggregated_points = feat_dict['aggregated_points']
        refine_predictions, refine_dict = self.refine_module(feat_dict)
        refine_decode_res = self.bbox_coder.split_pred(refine_predictions,
                                                       aggregated_points)
        for key in refine_decode_res.keys():
            refine_dict[key + '_optimized'] = refine_decode_res[key]
        return refine_dict

    def loss(self,
             bbox_preds,
             points,
             gt_bboxes_3d,
             gt_labels_3d,
             pts_semantic_mask=None,
             pts_instance_mask=None,
             img_metas=None,
             rpn_targets=None,
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
            rpn_targets (Tuple) : Targets generated by rpn head.
            gt_bboxes_ignore (None | list[torch.Tensor]): Specify
                which bounding.

        Returns:
            dict: Losses of H3dnet.
        """
        (vote_targets, vote_target_masks, size_class_targets, size_res_targets,
         dir_class_targets, dir_res_targets, center_targets, mask_targets,
         valid_gt_masks, objectness_targets, objectness_weights,
         box_loss_weights, valid_gt_weights) = rpn_targets

        loss_inputs = (bbox_preds, points, gt_bboxes_3d, gt_labels_3d,
                       pts_semantic_mask, pts_instance_mask, img_metas,
                       gt_bboxes_ignore)
        losses = {}

        # calculate refined proposal loss
        refined_proposal_loss = self.get_proposal_stage_loss(
            bbox_preds,
            size_class_targets,
            size_res_targets,
            dir_class_targets,
            dir_res_targets,
            center_targets,
            mask_targets,
            objectness_targets,
            objectness_weights,
            box_loss_weights,
            valid_gt_weights,
            suffix='_optimized')
        for key in refined_proposal_loss.keys():
            losses[key + '_optimized'] = refined_proposal_loss[key]

        bbox3d_optimized = self.bbox_coder.decode(
            bbox_preds, suffix='_optimized')
        refined_loss = self.refine_module.loss(bbox3d_optimized, *loss_inputs)
        losses.update(refined_loss)

        return losses

    def get_bboxes(self,
                   points,
                   bbox_preds,
                   input_metas,
                   rescale=False,
                   suffix=''):
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
        obj_scores = F.softmax(
            bbox_preds['obj_scores' + suffix], dim=-1)[..., -1]

        sem_scores = F.softmax(bbox_preds['sem_scores'], dim=-1)

        prediction_collection = {}
        prediction_collection['center'] = bbox_preds['center' + suffix]
        prediction_collection['dir_class'] = bbox_preds['dir_class']
        prediction_collection['dir_res'] = bbox_preds['dir_res' + suffix]
        prediction_collection['size_class'] = bbox_preds['size_class']
        prediction_collection['size_res'] = bbox_preds['size_res' + suffix]

        bbox3d = self.bbox_coder.decode(prediction_collection)

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

    def get_proposal_stage_loss(self,
                                bbox_preds,
                                size_class_targets,
                                size_res_targets,
                                dir_class_targets,
                                dir_res_targets,
                                center_targets,
                                mask_targets,
                                objectness_targets,
                                objectness_weights,
                                box_loss_weights,
                                valid_gt_weights,
                                suffix=''):
        """Compute loss for the aggregation module.

        Args:
            bbox_preds (dict): Predictions from forward of vote head.
            size_class_targets (torch.Tensor): Ground truth \
                size class of each prediction bounding box.
            size_res_targets (torch.Tensor): Ground truth \
                size residual of each prediction bounding box.
            dir_class_targets (torch.Tensor): Ground truth \
                direction class of each prediction bounding box.
            dir_res_targets (torch.Tensor): Ground truth \
                direction residual of each prediction bounding box.
            center_targets (torch.Tensor): Ground truth center \
                of each prediction bounding box.
            mask_targets (torch.Tensor): Validation of each \
                prediction bounding box.
            objectness_targets (torch.Tensor): Ground truth \
                objectness label of each prediction bounding box.
            objectness_weights (torch.Tensor): Weights of objectness \
                loss for each prediction bounding box.
            box_loss_weights (torch.Tensor): Weights of regression \
                loss for each prediction bounding box.
            valid_gt_weights (torch.Tensor): Validation of each \
                ground truth bounding box.

        Returns:
            dict: Losses of aggregation module.
        """
        # calculate objectness loss
        objectness_loss = self.objectness_loss(
            bbox_preds['obj_scores' + suffix].transpose(2, 1),
            objectness_targets,
            weight=objectness_weights)

        # calculate center loss
        source2target_loss, target2source_loss = self.center_loss(
            bbox_preds['center' + suffix],
            center_targets,
            src_weight=box_loss_weights,
            dst_weight=valid_gt_weights)
        center_loss = source2target_loss + target2source_loss

        # calculate direction class loss
        dir_class_loss = self.dir_class_loss(
            bbox_preds['dir_class' + suffix].transpose(2, 1),
            dir_class_targets,
            weight=box_loss_weights)

        # calculate direction residual loss
        batch_size, proposal_num = size_class_targets.shape[:2]
        heading_label_one_hot = dir_class_targets.new_zeros(
            (batch_size, proposal_num, self.num_dir_bins))
        heading_label_one_hot.scatter_(2, dir_class_targets.unsqueeze(-1), 1)
        dir_res_norm = torch.sum(
            bbox_preds['dir_res_norm' + suffix] * heading_label_one_hot, -1)
        dir_res_loss = self.dir_res_loss(
            dir_res_norm, dir_res_targets, weight=box_loss_weights)

        # calculate size class loss
        size_class_loss = self.size_class_loss(
            bbox_preds['size_class' + suffix].transpose(2, 1),
            size_class_targets,
            weight=box_loss_weights)

        # calculate size residual loss
        one_hot_size_targets = box_loss_weights.new_zeros(
            (batch_size, proposal_num, self.num_sizes))
        one_hot_size_targets.scatter_(2, size_class_targets.unsqueeze(-1), 1)
        one_hot_size_targets_expand = one_hot_size_targets.unsqueeze(
            -1).repeat(1, 1, 1, 3)
        size_residual_norm = torch.sum(
            bbox_preds['size_res_norm' + suffix] * one_hot_size_targets_expand,
            2)
        box_loss_weights_expand = box_loss_weights.unsqueeze(-1).repeat(
            1, 1, 3)
        size_res_loss = self.size_res_loss(
            size_residual_norm,
            size_res_targets,
            weight=box_loss_weights_expand)

        # calculate semantic loss
        semantic_loss = self.semantic_loss(
            bbox_preds['sem_scores' + suffix].transpose(2, 1),
            mask_targets,
            weight=box_loss_weights)

        losses = dict(
            objectness_loss=objectness_loss,
            semantic_loss=semantic_loss,
            center_loss=center_loss,
            dir_class_loss=dir_class_loss,
            dir_res_loss=dir_res_loss,
            size_class_loss=size_class_loss,
            size_res_loss=size_res_loss)

        return losses


class ProposalRefineModule(nn.Module):
    r"""Bbox head of `H3dnet <https://arxiv.org/abs/2006.05682>`_.

    Args:
        num_classes (int): The number of classes.
        num_heading_bin (int): Number of bins to encode direction angle.
        num_size_cluster (int): Number of size clusters.
        mean_size_arr (list[list[int]]): Mean size of bboxes in each class.
        num_proposal (int): The number of proposal.
        suface_matching_cfg (dict): Config for suface primitive matching.
        line_matching_cfg (dict): Config for line primitive matching.
        primitive_refine_channels (tuple[int]): Convolution channels of
            prediction layer.
        primitive_feat_refine_streams (int): The number of mlps to
            refine primitive feature.
        upper_thresh (float): Threshold for line matching.
        surface_thresh (float): Threshold for suface matching.
        line_thresh (float): Threshold for line matching.
        with_angle (bool): Whether the bbox is with rotation.
        train_cfg (dict): Config for training.
        cues_objectness_loss (dict): Config of cues objectness loss.
        cues_semantic_loss (dict): Config of cues semantic loss.
        proposal_objectness_loss (dict): Config of proposal objectness
            loss.
        primitive_center_loss (dict): Config of primitive center regression
            loss.
        conv_cfg (dict): Config of convolution in prediction layer.
        norm_cfg (dict): Config of BN in prediction layer.
    """

    def __init__(self,
                 num_classes,
                 num_heading_bin,
                 num_size_cluster,
                 mean_size_arr,
                 num_proposal,
                 suface_matching_cfg,
                 line_matching_cfg,
                 primitive_refine_channels,
                 primitive_feat_refine_streams=2,
                 upper_thresh=100.0,
                 surface_thresh=0.5,
                 line_thresh=0.5,
                 with_angle=False,
                 train_cfg=None,
                 cues_objectness_loss=None,
                 cues_semantic_loss=None,
                 proposal_objectness_loss=None,
                 primitive_center_loss=None,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d')):
        super().__init__()

        self.num_classes = num_classes
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.with_angle = with_angle
        self.upper_thresh = upper_thresh
        self.surface_thresh = surface_thresh
        self.line_thresh = line_thresh
        self.train_cfg = train_cfg

        self.cues_objectness_loss = build_loss(cues_objectness_loss)
        self.cues_semantic_loss = build_loss(cues_semantic_loss)
        self.proposal_objectness_loss = build_loss(proposal_objectness_loss)
        self.primitive_center_loss = build_loss(primitive_center_loss)

        assert suface_matching_cfg['mlp_channels'][-1] == \
            line_matching_cfg['mlp_channels'][-1]
        # surface center matching
        self.match_surface_center = PointSAModule(**suface_matching_cfg)
        # line center matching
        self.match_line_center = PointSAModule(**line_matching_cfg)

        # Compute the matching scores
        matching_feat_dims = suface_matching_cfg['mlp_channels'][-1]
        self.matching_conv = ConvModule(
            matching_feat_dims,
            matching_feat_dims,
            1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=True,
            inplace=True)
        self.matching_pred = torch.nn.Conv1d(matching_feat_dims, 2, 1)

        # Compute the semantic matching scores
        self.matching_sem_conv = ConvModule(
            matching_feat_dims,
            matching_feat_dims,
            1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=True,
            inplace=True)
        self.matching_sem_pred = torch.nn.Conv1d(matching_feat_dims, 2, 1)

        # Surface feature aggregation
        self.surface_feat_aggregation = list()
        for k in range(primitive_feat_refine_streams):
            self.surface_feat_aggregation.append(
                ConvModule(
                    matching_feat_dims,
                    matching_feat_dims,
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    bias=True,
                    inplace=True))
        self.surface_feat_aggregation = nn.Sequential(
            *self.surface_feat_aggregation)

        # Line feature aggregation
        self.line_feat_aggregation = list()
        for k in range(primitive_feat_refine_streams):
            self.line_feat_aggregation.append(
                ConvModule(
                    matching_feat_dims,
                    matching_feat_dims,
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    bias=True,
                    inplace=True))
        self.line_feat_aggregation = nn.Sequential(*self.line_feat_aggregation)

        prev_channel = 18 * matching_feat_dims
        self.proposal_pred = nn.ModuleList()
        for k in range(len(primitive_refine_channels)):
            self.proposal_pred.append(
                ConvModule(
                    prev_channel,
                    primitive_refine_channels[k],
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    bias=True,
                    inplace=False))
            prev_channel = primitive_refine_channels[k]
        # Final object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class +
        # residual(num_size_cluster*4)
        conv_out_channel = (2 + 3 + num_heading_bin * 2 +
                            num_size_cluster * 4 + self.num_classes)
        self.proposal_pred.append(nn.Conv1d(prev_channel, conv_out_channel, 1))

    def forward(self, feat_dict):
        """Forward pass.

        Args:
            feat_dict (dict): Feature dict from backbone and primitive head.

        Returns:
            dict: Predictions of refined bbox.
        """
        ret_dict = {}
        z_feature = feat_dict['aggregated_features_z']
        xy_feature = feat_dict['aggregated_features_xy']
        line_feature = feat_dict['aggregated_features_line']
        original_feature = feat_dict['aggregated_features']

        batch_size = original_feature.shape[0]
        object_proposal = original_feature.shape[2]

        # Extract surface points and features here
        z_center = feat_dict['pred_z_center']
        xy_center = feat_dict['pred_xy_center']
        z_sem = feat_dict['sem_cls_scores_z']
        xy_sem = feat_dict['sem_cls_scores_xy']

        surface_center_pred = torch.cat((z_center, xy_center), dim=1)
        ret_dict['surface_center_pred'] = surface_center_pred
        ret_dict['surface_sem_pred'] = torch.cat((z_sem, xy_sem), dim=1)
        surface_center_feature_pred = torch.cat((z_feature, xy_feature), dim=2)
        surface_center_feature_pred = torch.cat(
            (surface_center_feature_pred.new_zeros(
                (batch_size, 6, surface_center_feature_pred.shape[2])),
             surface_center_feature_pred),
            dim=1)

        # Extract line points and features
        line_center = feat_dict['pred_line_center']

        # Extract the surface and line centers of rpn proposals
        rpn_proposals = feat_dict['proposal_list'].float()
        rpn_proposals_bbox = DepthInstance3DBoxes(
            rpn_proposals.reshape(-1, 7).clone(),
            box_dim=rpn_proposals.shape[-1],
            with_yaw=self.with_angle,
            origin=(0.5, 0.5, 0.5))

        obj_surface_center, obj_line_center = \
            rpn_proposals_bbox.get_surface_line_center()
        obj_surface_center = obj_surface_center.reshape(
            batch_size, object_proposal * 6, 3)
        obj_line_center = obj_line_center.reshape(batch_size,
                                                  object_proposal * 12, 3)

        ret_dict['surface_center_object'] = obj_surface_center
        ret_dict['line_center_object'] = obj_line_center

        surface_xyz, surface_features, _ = self.match_surface_center(
            surface_center_pred,
            surface_center_feature_pred,
            target_xyz=obj_surface_center)
        line_feature = torch.cat((line_feature.new_zeros(
            (batch_size, 12, line_feature.shape[2])), line_feature),
                                 dim=1)

        line_xyz, line_features, _ = self.match_line_center(
            line_center, line_feature, target_xyz=obj_line_center)

        combine_features = torch.cat((surface_features, line_features), dim=2)

        match_features = self.matching_conv(combine_features)
        match_score = self.matching_pred(match_features)
        ret_dict['match_scores'] = match_score.transpose(2, 1)

        match_features_sem = self.matching_sem_conv(combine_features)
        match_score_sem = self.matching_sem_pred(match_features_sem)
        ret_dict['match_scores_sem'] = match_score_sem.transpose(2, 1)

        surface_features = self.surface_feat_aggregation(surface_features)

        line_features = self.line_feat_aggregation(line_features)

        surface_features = surface_features.view(batch_size, -1, 6,
                                                 object_proposal)
        line_features = line_features.view(batch_size, -1, 12, object_proposal)

        # Combine all surface and line features
        surface_pool_feature = surface_features.view(batch_size, -1,
                                                     object_proposal)
        line_pool_feature = line_features.view(batch_size, -1, object_proposal)

        combine_feature = torch.cat((surface_pool_feature, line_pool_feature),
                                    dim=1)

        predictions = self.proposal_pred[0](combine_feature)
        predictions += original_feature
        for conv_module in self.proposal_pred[1:]:
            predictions = conv_module(predictions)

        return predictions, ret_dict

    def loss(self,
             bbox_preds,
             preds_dict,
             points,
             gt_bboxes_3d,
             gt_labels_3d,
             pts_semantic_mask=None,
             pts_instance_mask=None,
             img_metas=None,
             gt_bboxes_ignore=None):
        """Compute loss.

        Args:
            bbox_preds (torch.Tensor): Predictions of the bounding boxes.
            preds_dict (dict): Predictions from forward of vote head.
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
                                   preds_dict)

        (cues_objectness_label, cues_sem_label, proposal_objectness_label,
         cues_mask, cues_match_mask, proposal_objectness_mask,
         cues_matching_label, obj_surface_line_center) = targets

        # match scores for each geometric primitive
        objectness_scores = preds_dict['match_scores']
        # match scores for the semantics of primitives
        objectness_scores_sem = preds_dict['match_scores_sem']

        primitive_objectness_loss = self.cues_objectness_loss(
            objectness_scores.transpose(2, 1),
            cues_objectness_label,
            weight=cues_mask,
            avg_factor=cues_mask.sum() + 1e-6)

        primitive_sem_loss = self.cues_semantic_loss(
            objectness_scores_sem.transpose(2, 1),
            cues_sem_label,
            weight=cues_mask,
            avg_factor=cues_mask.sum() + 1e-6)

        objectness_scores = preds_dict['obj_scores_optimized']
        objectness_loss_refine = self.proposal_objectness_loss(
            objectness_scores.transpose(2, 1), proposal_objectness_label)
        primitive_matching_loss = torch.sum(
            objectness_loss_refine *
            cues_match_mask) / (torch.sum(cues_match_mask) + 1e-6) * 0.5
        primitive_sem_matching_loss = torch.sum(
            objectness_loss_refine * proposal_objectness_mask) / (
                torch.sum(proposal_objectness_mask) + 1e-6) * 0.5

        # Get the object surface center here
        batch_size, object_proposal = bbox_preds.shape[:2]
        refined_bbox = DepthInstance3DBoxes(
            bbox_preds.reshape(-1, 7).clone(),
            box_dim=bbox_preds.shape[-1],
            with_yaw=self.with_angle,
            origin=(0.5, 0.5, 0.5))

        pred_obj_surface_center, pred_obj_line_center = \
            refined_bbox.get_surface_line_center()
        pred_obj_surface_center = pred_obj_surface_center.reshape(
            batch_size, object_proposal * 6, 3)
        pred_obj_line_center = pred_obj_line_center.reshape(
            batch_size, object_proposal * 12, 3)
        pred_surface_line_center = torch.cat(
            (pred_obj_surface_center, pred_obj_line_center), 1)

        square_dist = self.primitive_center_loss(pred_surface_line_center,
                                                 obj_surface_line_center)

        match_dist = torch.sqrt(torch.sum(square_dist, dim=-1) + 1e-6)
        primitive_centroid_reg_loss = torch.sum(
            match_dist * cues_matching_label) / (
                torch.sum(cues_matching_label) + 1e-6)

        losses = dict(
            primitive_objectness_loss=primitive_objectness_loss,
            primitive_sem_loss=primitive_sem_loss,
            primitive_matching_loss=primitive_matching_loss,
            primitive_sem_matching_loss=primitive_sem_matching_loss,
            primitive_centroid_reg_loss=primitive_centroid_reg_loss)
        return losses

    def get_targets(self,
                    points,
                    gt_bboxes_3d,
                    gt_labels_3d,
                    pts_semantic_mask=None,
                    pts_instance_mask=None,
                    bbox_preds=None):
        """Generate targets of proposal module.

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
            tuple[torch.Tensor]: Targets of proposal module.
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

        if pts_semantic_mask is None:
            pts_semantic_mask = [None for i in range(len(gt_labels_3d))]
            pts_instance_mask = [None for i in range(len(gt_labels_3d))]

        aggregated_points = [
            bbox_preds['aggregated_points'][i]
            for i in range(len(gt_labels_3d))
        ]

        surface_center_pred = [
            bbox_preds['surface_center_pred'][i]
            for i in range(len(gt_labels_3d))
        ]

        line_center_pred = [
            bbox_preds['pred_line_center'][i]
            for i in range(len(gt_labels_3d))
        ]

        surface_center_object = [
            bbox_preds['surface_center_object'][i]
            for i in range(len(gt_labels_3d))
        ]

        line_center_object = [
            bbox_preds['line_center_object'][i]
            for i in range(len(gt_labels_3d))
        ]

        surface_sem_pred = [
            bbox_preds['surface_sem_pred'][i]
            for i in range(len(gt_labels_3d))
        ]

        line_sem_pred = [
            bbox_preds['sem_cls_scores_line'][i]
            for i in range(len(gt_labels_3d))
        ]

        (cues_objectness_label, cues_sem_label, proposal_objectness_label,
         cues_mask, cues_match_mask, proposal_objectness_mask,
         cues_matching_label, obj_surface_line_center) = multi_apply(
             self.get_targets_single, points, gt_bboxes_3d, gt_labels_3d,
             pts_semantic_mask, pts_instance_mask, aggregated_points,
             surface_center_pred, line_center_pred, surface_center_object,
             line_center_object, surface_sem_pred, line_sem_pred)

        cues_objectness_label = torch.stack(cues_objectness_label)
        cues_sem_label = torch.stack(cues_sem_label)
        proposal_objectness_label = torch.stack(proposal_objectness_label)
        cues_mask = torch.stack(cues_mask)
        cues_match_mask = torch.stack(cues_match_mask)
        proposal_objectness_mask = torch.stack(proposal_objectness_mask)
        cues_matching_label = torch.stack(cues_matching_label)
        obj_surface_line_center = torch.stack(obj_surface_line_center)

        return (cues_objectness_label, cues_sem_label,
                proposal_objectness_label, cues_mask, cues_match_mask,
                proposal_objectness_mask, cues_matching_label,
                obj_surface_line_center)

    def get_targets_single(self,
                           points,
                           gt_bboxes_3d,
                           gt_labels_3d,
                           pts_semantic_mask=None,
                           pts_instance_mask=None,
                           aggregated_points=None,
                           pred_surface_center=None,
                           pred_line_center=None,
                           pred_obj_surface_center=None,
                           pred_obj_line_center=None,
                           pred_surface_sem=None,
                           pred_line_sem=None):
        """Generate targets for primitive cues for single batch.

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
            pred_surface_center (torch.Tensor): Prediction of surface center.
            pred_line_center (torch.Tensor): Prediction of line center.
            pred_obj_surface_center (torch.Tensor): Objectness prediction \
                of surface center.
            pred_obj_line_center (torch.Tensor): Objectness prediction of \
                line center.
            pred_surface_sem (torch.Tensor): Semantic prediction of \
                surface center.
            pred_line_sem (torch.Tensor): Semantic prediction of line center.
        Returns:
            tuple[torch.Tensor]: Targets for primitive cues.
        """
        device = points.device
        gt_bboxes_3d = gt_bboxes_3d.to(device)
        num_proposals = aggregated_points.shape[0]
        gt_center = gt_bboxes_3d.gravity_center

        dist1, dist2, ind1, _ = chamfer_distance(
            aggregated_points.unsqueeze(0),
            gt_center.unsqueeze(0),
            reduction='none')
        # Set assignment
        object_assignment = ind1.squeeze(0)

        # Generate objectness label and mask
        # objectness_label: 1 if pred object center is within
        # self.train_cfg['near_threshold'] of any GT object
        # objectness_mask: 0 if pred object center is in gray
        # zone (DONOTCARE), 1 otherwise
        euclidean_dist1 = torch.sqrt(dist1.squeeze(0) + 1e-6)
        proposal_objectness_label = euclidean_dist1.new_zeros(
            num_proposals, dtype=torch.long)
        proposal_objectness_mask = euclidean_dist1.new_zeros(num_proposals)

        gt_sem = gt_labels_3d[object_assignment]

        obj_surface_center, obj_line_center = \
            gt_bboxes_3d.get_surface_line_center()
        obj_surface_center = obj_surface_center.reshape(
            6, -1, 3)[:, object_assignment].reshape(1, -1, 3)
        obj_line_center = obj_line_center.reshape(
            12, -1, 3)[:, object_assignment].reshape(1, -1, 3)

        surface_sem = torch.argmax(pred_surface_sem, dim=1).float()
        line_sem = torch.argmax(pred_line_sem, dim=1).float()

        dist_surface, _, surface_ind, _ = chamfer_distance(
            obj_surface_center,
            pred_surface_center.unsqueeze(0),
            reduction='none')
        dist_line, _, line_ind, _ = chamfer_distance(
            obj_line_center, pred_line_center.unsqueeze(0), reduction='none')

        surface_sel = pred_surface_center[surface_ind.squeeze(0)]
        line_sel = pred_line_center[line_ind.squeeze(0)]
        surface_sel_sem = surface_sem[surface_ind.squeeze(0)]
        line_sel_sem = line_sem[line_ind.squeeze(0)]

        surface_sel_sem_gt = gt_sem.repeat(6).float()
        line_sel_sem_gt = gt_sem.repeat(12).float()

        euclidean_dist_surface = torch.sqrt(dist_surface.squeeze(0) + 1e-6)
        euclidean_dist_line = torch.sqrt(dist_line.squeeze(0) + 1e-6)
        objectness_label_surface = euclidean_dist_line.new_zeros(
            num_proposals * 6, dtype=torch.long)
        objectness_mask_surface = euclidean_dist_line.new_zeros(num_proposals *
                                                                6)
        objectness_label_line = euclidean_dist_line.new_zeros(
            num_proposals * 12, dtype=torch.long)
        objectness_mask_line = euclidean_dist_line.new_zeros(num_proposals *
                                                             12)
        objectness_label_surface_sem = euclidean_dist_line.new_zeros(
            num_proposals * 6, dtype=torch.long)
        objectness_label_line_sem = euclidean_dist_line.new_zeros(
            num_proposals * 12, dtype=torch.long)

        euclidean_dist_obj_surface = torch.sqrt(
            torch.sum((pred_obj_surface_center - surface_sel)**2, dim=-1) +
            1e-6)
        euclidean_dist_obj_line = torch.sqrt(
            torch.sum((pred_obj_line_center - line_sel)**2, dim=-1) + 1e-6)

        # Objectness score just with centers
        proposal_objectness_label[
            euclidean_dist1 < self.train_cfg['near_threshold']] = 1
        proposal_objectness_mask[
            euclidean_dist1 < self.train_cfg['near_threshold']] = 1
        proposal_objectness_mask[
            euclidean_dist1 > self.train_cfg['far_threshold']] = 1

        objectness_label_surface[
            (euclidean_dist_obj_surface <
             self.train_cfg['label_surface_threshold']) *
            (euclidean_dist_surface <
             self.train_cfg['mask_surface_threshold'])] = 1
        objectness_label_surface_sem[
            (euclidean_dist_obj_surface <
             self.train_cfg['label_surface_threshold']) *
            (euclidean_dist_surface < self.train_cfg['mask_surface_threshold'])
            * (surface_sel_sem == surface_sel_sem_gt)] = 1

        objectness_label_line[
            (euclidean_dist_obj_line < self.train_cfg['label_line_threshold'])
            *
            (euclidean_dist_line < self.train_cfg['mask_line_threshold'])] = 1
        objectness_label_line_sem[
            (euclidean_dist_obj_line < self.train_cfg['label_line_threshold'])
            * (euclidean_dist_line < self.train_cfg['mask_line_threshold']) *
            (line_sel_sem == line_sel_sem_gt)] = 1

        objectness_label_surface_obj = proposal_objectness_label.repeat(6)
        objectness_mask_surface_obj = proposal_objectness_mask.repeat(6)
        objectness_label_line_obj = proposal_objectness_label.repeat(12)
        objectness_mask_line_obj = proposal_objectness_mask.repeat(12)

        objectness_mask_surface = objectness_mask_surface_obj
        objectness_mask_line = objectness_mask_line_obj

        cues_objectness_label = torch.cat(
            (objectness_label_surface, objectness_label_line), 0)
        cues_sem_label = torch.cat(
            (objectness_label_surface_sem, objectness_label_line_sem), 0)
        cues_mask = torch.cat((objectness_mask_surface, objectness_mask_line),
                              0)

        objectness_label_surface *= objectness_label_surface_obj
        objectness_label_line *= objectness_label_line_obj
        cues_matching_label = torch.cat(
            (objectness_label_surface, objectness_label_line), 0)

        objectness_label_surface_sem *= objectness_label_surface_obj
        objectness_label_line_sem *= objectness_label_line_obj

        cues_match_mask = (torch.sum(
            cues_objectness_label.view(18, num_proposals), dim=0) >=
                           1).float()

        obj_surface_line_center = torch.cat(
            (obj_surface_center, obj_line_center), 1).squeeze(0)

        return (cues_objectness_label, cues_sem_label,
                proposal_objectness_label, cues_mask, cues_match_mask,
                proposal_objectness_mask, cues_matching_label,
                obj_surface_line_center)
