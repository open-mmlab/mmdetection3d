# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.cnn import Scale, bias_init_with_prob, normal_init
from mmcv.runner import force_fp32
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.core import box3d_multiclass_nms, xywhr2xyxyr
from mmdet3d.core.bbox import points_cam2img, points_img2cam
from mmdet.core import distance2bbox, multi_apply
from ..builder import HEADS, build_loss
from .fcos_mono3d_head import FCOSMono3DHead


@HEADS.register_module()
class PGDHead(FCOSMono3DHead):
    r"""Anchor-free head used in `PGD <https://arxiv.org/abs/2107.14160>`_.

    Args:
        use_depth_classifer (bool, optional): Whether to use depth classifier.
            Defaults to True.
        use_only_reg_proj (bool, optional): Whether to use only direct
            regressed depth in the re-projection (to make the network easier
            to learn). Defaults to False.
        weight_dim (int, optional): Dimension of the location-aware weight
            map. Defaults to -1.
        weight_branch (tuple[tuple[int]], optional): Feature map channels of
            the convolutional branch for weight map. Defaults to ((256, ), ).
        depth_branch (tuple[int], optional): Feature map channels of the
            branch for probabilistic depth estimation. Defaults to (64, ),
        depth_range (tuple[float], optional): Range of depth estimation.
            Defaults to (0, 70),
        depth_unit (int, optional): Unit of depth range division. Defaults to
            10.
        division (str, optional): Depth division method. Options include
            'uniform', 'linear', 'log', 'loguniform'. Defaults to 'uniform'.
        depth_bins (int, optional): Discrete bins of depth division. Defaults
            to 8.
        loss_depth (dict, optional): Depth loss. Defaults to dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0).
        loss_bbox2d (dict, optional): Loss for 2D box estimation. Defaults to
            dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0).
        loss_consistency (dict, optional): Consistency loss. Defaults to
            dict(type='GIoULoss', loss_weight=1.0),
        pred_velo (bool, optional): Whether to predict velocity. Defaults to
            False.
        pred_bbox2d (bool, optional): Whether to predict 2D bounding boxes.
            Defaults to True.
        pred_keypoints (bool, optional): Whether to predict keypoints.
            Defaults to False,
        bbox_coder (dict, optional): Bounding box coder. Defaults to
            dict(type='PGDBBoxCoder', base_depths=((28.01, 16.32), ),
            base_dims=((0.8, 1.73, 0.6), (1.76, 1.73, 0.6), (3.9, 1.56, 1.6)),
            code_size=7).
    """

    def __init__(self,
                 use_depth_classifier=True,
                 use_onlyreg_proj=False,
                 weight_dim=-1,
                 weight_branch=((256, ), ),
                 depth_branch=(64, ),
                 depth_range=(0, 70),
                 depth_unit=10,
                 division='uniform',
                 depth_bins=8,
                 loss_depth=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_bbox2d=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_consistency=dict(type='GIoULoss', loss_weight=1.0),
                 pred_bbox2d=True,
                 pred_keypoints=False,
                 bbox_coder=dict(
                     type='PGDBBoxCoder',
                     base_depths=((28.01, 16.32), ),
                     base_dims=((0.8, 1.73, 0.6), (1.76, 1.73, 0.6),
                                (3.9, 1.56, 1.6)),
                     code_size=7),
                 **kwargs):
        self.use_depth_classifier = use_depth_classifier
        self.use_onlyreg_proj = use_onlyreg_proj
        self.depth_branch = depth_branch
        self.pred_keypoints = pred_keypoints
        self.weight_dim = weight_dim
        self.weight_branch = weight_branch
        self.weight_out_channels = []
        for weight_branch_channels in weight_branch:
            if len(weight_branch_channels) > 0:
                self.weight_out_channels.append(weight_branch_channels[-1])
            else:
                self.weight_out_channels.append(-1)
        self.depth_range = depth_range
        self.depth_unit = depth_unit
        self.division = division
        if self.division == 'uniform':
            self.num_depth_cls = int(
                (depth_range[1] - depth_range[0]) / depth_unit) + 1
            if self.num_depth_cls != depth_bins:
                print('Warning: The number of bins computed from ' +
                      'depth_unit is different from given parameter! ' +
                      'Depth_unit will be considered with priority in ' +
                      'Uniform Division.')
        else:
            self.num_depth_cls = depth_bins
        super().__init__(
            pred_bbox2d=pred_bbox2d, bbox_coder=bbox_coder, **kwargs)
        self.loss_depth = build_loss(loss_depth)
        if self.pred_bbox2d:
            self.loss_bbox2d = build_loss(loss_bbox2d)
            self.loss_consistency = build_loss(loss_consistency)
        if self.pred_keypoints:
            self.kpts_start = 9 if self.pred_velo else 7

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        if self.pred_bbox2d:
            self.scale_dim += 1
        if self.pred_keypoints:
            self.scale_dim += 1
        self.scales = nn.ModuleList([
            nn.ModuleList([Scale(1.0) for _ in range(self.scale_dim)])
            for _ in self.strides
        ])

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        super()._init_predictor()

        if self.use_depth_classifier:
            self.conv_depth_cls_prev = self._init_branch(
                conv_channels=self.depth_branch,
                conv_strides=(1, ) * len(self.depth_branch))
            self.conv_depth_cls = nn.Conv2d(self.depth_branch[-1],
                                            self.num_depth_cls, 1)
            # Data-agnostic single param lambda for local depth fusion
            self.fuse_lambda = nn.Parameter(torch.tensor(10e-5))

        if self.weight_dim != -1:
            self.conv_weight_prevs = nn.ModuleList()
            self.conv_weights = nn.ModuleList()
            for i in range(self.weight_dim):
                weight_branch_channels = self.weight_branch[i]
                weight_out_channel = self.weight_out_channels[i]
                if len(weight_branch_channels) > 0:
                    self.conv_weight_prevs.append(
                        self._init_branch(
                            conv_channels=weight_branch_channels,
                            conv_strides=(1, ) * len(weight_branch_channels)))
                    self.conv_weights.append(
                        nn.Conv2d(weight_out_channel, 1, 1))
                else:
                    self.conv_weight_prevs.append(None)
                    self.conv_weights.append(
                        nn.Conv2d(self.feat_channels, 1, 1))

    def init_weights(self):
        """Initialize weights of the head.

        We currently still use the customized defined init_weights because the
        default init of DCN triggered by the init_cfg will init
        conv_offset.weight, which mistakenly affects the training stability.
        """
        super().init_weights()

        bias_cls = bias_init_with_prob(0.01)
        if self.use_depth_classifier:
            for m in self.conv_depth_cls_prev:
                if isinstance(m.conv, nn.Conv2d):
                    normal_init(m.conv, std=0.01)
            normal_init(self.conv_depth_cls, std=0.01, bias=bias_cls)

        if self.weight_dim != -1:
            for conv_weight_prev in self.conv_weight_prevs:
                if conv_weight_prev is None:
                    continue
                for m in conv_weight_prev:
                    if isinstance(m.conv, nn.Conv2d):
                        normal_init(m.conv, std=0.01)
            for conv_weight in self.conv_weights:
                normal_init(conv_weight, std=0.01)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level,
                    each is a 4D-tensor, the channel number is
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * bbox_code_size.
                dir_cls_preds (list[Tensor]): Box scores for direction class
                    predictions on each scale level, each is a 4D-tensor,
                    the channel number is num_points * 2. (bin = 2).
                weight (list[Tensor]): Location-aware weight maps on each
                    scale level, each is a 4D-tensor, the channel number is
                    num_points * 1.
                depth_cls_preds (list[Tensor]): Box scores for depth class
                    predictions on each scale level, each is a 4D-tensor,
                    the channel number is num_points * self.num_depth_cls.
                attr_preds (list[Tensor]): Attribute scores for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * num_attrs.
                centernesses (list[Tensor]): Centerness for each scale level,
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides)

    def forward_single(self, x, scale, stride):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox and direction class
                predictions, depth class predictions, location-aware weights,
                attribute and centerness predictions of input feature maps.
        """
        cls_score, bbox_pred, dir_cls_pred, attr_pred, centerness, cls_feat, \
            reg_feat = super().forward_single(x, scale, stride)

        max_regress_range = stride * self.regress_ranges[0][1] / \
            self.strides[0]
        bbox_pred = self.bbox_coder.decode_2d(bbox_pred, scale, stride,
                                              max_regress_range, self.training,
                                              self.pred_keypoints,
                                              self.pred_bbox2d)

        depth_cls_pred = None
        if self.use_depth_classifier:
            clone_reg_feat = reg_feat.clone()
            for conv_depth_cls_prev_layer in self.conv_depth_cls_prev:
                clone_reg_feat = conv_depth_cls_prev_layer(clone_reg_feat)
            depth_cls_pred = self.conv_depth_cls(clone_reg_feat)

        weight = None
        if self.weight_dim != -1:
            weight = []
            for i in range(self.weight_dim):
                clone_reg_feat = reg_feat.clone()
                if len(self.weight_branch[i]) > 0:
                    for conv_weight_prev_layer in self.conv_weight_prevs[i]:
                        clone_reg_feat = conv_weight_prev_layer(clone_reg_feat)
                weight.append(self.conv_weights[i](clone_reg_feat))
            weight = torch.cat(weight, dim=1)

        return cls_score, bbox_pred, dir_cls_pred, depth_cls_pred, weight, \
            attr_pred, centerness

    def get_proj_bbox2d(self,
                        bbox_preds,
                        pos_dir_cls_preds,
                        labels_3d,
                        bbox_targets_3d,
                        pos_points,
                        pos_inds,
                        img_metas,
                        pos_depth_cls_preds=None,
                        pos_weights=None,
                        pos_cls_scores=None,
                        with_kpts=False):
        """Decode box predictions and get projected 2D attributes.

        Args:
            bbox_preds (list[Tensor]): Box predictions for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * bbox_code_size.
            pos_dir_cls_preds (Tensor): Box scores for direction class
                predictions of positive boxes on all the scale levels in shape
                (num_pos_points, 2).
            labels_3d (list[Tensor]): 3D box category labels for each scale
                level, each is a 4D-tensor.
            bbox_targets_3d (list[Tensor]): 3D box targets for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * bbox_code_size.
            pos_points (Tensor): Foreground points.
            pos_inds (Tensor): Index of foreground points from flattened
                tensors.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            pos_depth_cls_preds (Tensor, optional): Probabilistic depth map of
                positive boxes on all the scale levels in shape
                (num_pos_points, self.num_depth_cls). Defaults to None.
            pos_weights (Tensor, optional): Location-aware weights of positive
                boxes in shape (num_pos_points, self.weight_dim). Defaults to
                None.
            pos_cls_scores (Tensor, optional): Classification scores of
                positive boxes in shape (num_pos_points, self.num_classes).
                Defaults to None.
            with_kpts (bool, optional): Whether to output keypoints targets.
                Defaults to False.

        Returns:
            tuple[Tensor]: Exterior 2D boxes from projected 3D boxes,
                predicted 2D boxes and keypoint targets (if necessary).
        """
        views = [np.array(img_meta['cam2img']) for img_meta in img_metas]
        num_imgs = len(img_metas)
        img_idx = []
        for label in labels_3d:
            for idx in range(num_imgs):
                img_idx.append(
                    labels_3d[0].new_ones(int(len(label) / num_imgs)) * idx)
        img_idx = torch.cat(img_idx)
        pos_img_idx = img_idx[pos_inds]

        flatten_strided_bbox_preds = []
        flatten_strided_bbox2d_preds = []
        flatten_bbox_targets_3d = []
        flatten_strides = []

        for stride_idx, bbox_pred in enumerate(bbox_preds):
            flatten_bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(
                -1, sum(self.group_reg_dims))
            flatten_bbox_pred[:, :2] *= self.strides[stride_idx]
            flatten_bbox_pred[:, -4:] *= self.strides[stride_idx]
            flatten_strided_bbox_preds.append(
                flatten_bbox_pred[:, :self.bbox_coder.bbox_code_size])
            flatten_strided_bbox2d_preds.append(flatten_bbox_pred[:, -4:])

            bbox_target_3d = bbox_targets_3d[stride_idx].clone()
            bbox_target_3d[:, :2] *= self.strides[stride_idx]
            bbox_target_3d[:, -4:] *= self.strides[stride_idx]
            flatten_bbox_targets_3d.append(bbox_target_3d)

            flatten_stride = flatten_bbox_pred.new_ones(
                *flatten_bbox_pred.shape[:-1], 1) * self.strides[stride_idx]
            flatten_strides.append(flatten_stride)

        flatten_strided_bbox_preds = torch.cat(flatten_strided_bbox_preds)
        flatten_strided_bbox2d_preds = torch.cat(flatten_strided_bbox2d_preds)
        flatten_bbox_targets_3d = torch.cat(flatten_bbox_targets_3d)
        flatten_strides = torch.cat(flatten_strides)
        pos_strided_bbox_preds = flatten_strided_bbox_preds[pos_inds]
        pos_strided_bbox2d_preds = flatten_strided_bbox2d_preds[pos_inds]
        pos_bbox_targets_3d = flatten_bbox_targets_3d[pos_inds]
        pos_strides = flatten_strides[pos_inds]

        pos_decoded_bbox2d_preds = distance2bbox(pos_points,
                                                 pos_strided_bbox2d_preds)

        pos_strided_bbox_preds[:, :2] = \
            pos_points - pos_strided_bbox_preds[:, :2]
        pos_bbox_targets_3d[:, :2] = \
            pos_points - pos_bbox_targets_3d[:, :2]

        if self.use_depth_classifier and (not self.use_onlyreg_proj):
            pos_prob_depth_preds = self.bbox_coder.decode_prob_depth(
                pos_depth_cls_preds, self.depth_range, self.depth_unit,
                self.division, self.num_depth_cls)
            sig_alpha = torch.sigmoid(self.fuse_lambda)
            pos_strided_bbox_preds[:, 2] = \
                sig_alpha * pos_strided_bbox_preds.clone()[:, 2] + \
                (1 - sig_alpha) * pos_prob_depth_preds

        box_corners_in_image = pos_strided_bbox_preds.new_zeros(
            (*pos_strided_bbox_preds.shape[:-1], 8, 2))
        box_corners_in_image_gt = pos_strided_bbox_preds.new_zeros(
            (*pos_strided_bbox_preds.shape[:-1], 8, 2))

        for idx in range(num_imgs):
            mask = (pos_img_idx == idx)
            if pos_strided_bbox_preds[mask].shape[0] == 0:
                continue
            cam2img = torch.eye(
                4,
                dtype=pos_strided_bbox_preds.dtype,
                device=pos_strided_bbox_preds.device)
            view_shape = views[idx].shape
            cam2img[:view_shape[0], :view_shape[1]] = \
                pos_strided_bbox_preds.new_tensor(views[idx])

            centers2d_preds = pos_strided_bbox_preds.clone()[mask, :2]
            centers2d_targets = pos_bbox_targets_3d.clone()[mask, :2]
            centers3d_targets = points_img2cam(pos_bbox_targets_3d[mask, :3],
                                               views[idx])

            # use predicted depth to re-project the 2.5D centers
            pos_strided_bbox_preds[mask, :3] = points_img2cam(
                pos_strided_bbox_preds[mask, :3], views[idx])
            pos_bbox_targets_3d[mask, :3] = centers3d_targets

            # depth fixed when computing re-project 3D bboxes
            pos_strided_bbox_preds[mask, 2] = \
                pos_bbox_targets_3d.clone()[mask, 2]

            # decode yaws
            if self.use_direction_classifier:
                pos_dir_cls_scores = torch.max(
                    pos_dir_cls_preds[mask], dim=-1)[1]
                pos_strided_bbox_preds[mask] = self.bbox_coder.decode_yaw(
                    pos_strided_bbox_preds[mask], centers2d_preds,
                    pos_dir_cls_scores, self.dir_offset, cam2img)
            pos_bbox_targets_3d[mask, 6] = torch.atan2(
                centers2d_targets[:, 0] - cam2img[0, 2],
                cam2img[0, 0]) + pos_bbox_targets_3d[mask, 6]

            corners = img_metas[0]['box_type_3d'](
                pos_strided_bbox_preds[mask],
                box_dim=self.bbox_coder.bbox_code_size,
                origin=(0.5, 0.5, 0.5)).corners
            box_corners_in_image[mask] = points_cam2img(corners, cam2img)

            corners_gt = img_metas[0]['box_type_3d'](
                pos_bbox_targets_3d[mask, :self.bbox_code_size],
                box_dim=self.bbox_coder.bbox_code_size,
                origin=(0.5, 0.5, 0.5)).corners
            box_corners_in_image_gt[mask] = points_cam2img(corners_gt, cam2img)

        minxy = torch.min(box_corners_in_image, dim=1)[0]
        maxxy = torch.max(box_corners_in_image, dim=1)[0]
        proj_bbox2d_preds = torch.cat([minxy, maxxy], dim=1)

        outputs = (proj_bbox2d_preds, pos_decoded_bbox2d_preds)

        if with_kpts:
            norm_strides = pos_strides * self.regress_ranges[0][1] / \
                self.strides[0]
            kpts_targets = box_corners_in_image_gt - pos_points[..., None, :]
            kpts_targets = kpts_targets.view(
                (*pos_strided_bbox_preds.shape[:-1], 16))
            kpts_targets /= norm_strides

            outputs += (kpts_targets, )

        return outputs

    def get_pos_predictions(self, bbox_preds, dir_cls_preds, depth_cls_preds,
                            weights, attr_preds, centernesses, pos_inds,
                            img_metas):
        """Flatten predictions and get positive ones.

        Args:
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * bbox_code_size.
            dir_cls_preds (list[Tensor]): Box scores for direction class
                predictions on each scale level, each is a 4D-tensor,
                the channel number is num_points * 2. (bin = 2)
            depth_cls_preds (list[Tensor]): Box scores for direction class
                predictions on each scale level, each is a 4D-tensor,
                the channel number is num_points * self.num_depth_cls.
            attr_preds (list[Tensor]): Attribute scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_attrs.
            centernesses (list[Tensor]): Centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            pos_inds (Tensor): Index of foreground points from flattened
                tensors.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            tuple[Tensor]: Box predictions, direction classes, probabilistic
                depth maps, location-aware weight maps, attributes and
                centerness predictions.
        """
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, sum(self.group_reg_dims))
            for bbox_pred in bbox_preds
        ]
        flatten_dir_cls_preds = [
            dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)
            for dir_cls_pred in dir_cls_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_dir_cls_preds = torch.cat(flatten_dir_cls_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_dir_cls_preds = flatten_dir_cls_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]

        pos_depth_cls_preds = None
        if self.use_depth_classifier:
            flatten_depth_cls_preds = [
                depth_cls_pred.permute(0, 2, 3,
                                       1).reshape(-1, self.num_depth_cls)
                for depth_cls_pred in depth_cls_preds
            ]
            flatten_depth_cls_preds = torch.cat(flatten_depth_cls_preds)
            pos_depth_cls_preds = flatten_depth_cls_preds[pos_inds]

        pos_weights = None
        if self.weight_dim != -1:
            flatten_weights = [
                weight.permute(0, 2, 3, 1).reshape(-1, self.weight_dim)
                for weight in weights
            ]
            flatten_weights = torch.cat(flatten_weights)
            pos_weights = flatten_weights[pos_inds]

        pos_attr_preds = None
        if self.pred_attrs:
            flatten_attr_preds = [
                attr_pred.permute(0, 2, 3, 1).reshape(-1, self.num_attrs)
                for attr_pred in attr_preds
            ]
            flatten_attr_preds = torch.cat(flatten_attr_preds)
            pos_attr_preds = flatten_attr_preds[pos_inds]

        return pos_bbox_preds, pos_dir_cls_preds, pos_depth_cls_preds, \
            pos_weights, pos_attr_preds, pos_centerness

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'dir_cls_preds',
                  'depth_cls_preds', 'weights', 'attr_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             dir_cls_preds,
             depth_cls_preds,
             weights,
             attr_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             gt_bboxes_3d,
             gt_labels_3d,
             centers2d,
             depths,
             attr_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * bbox_code_size.
            dir_cls_preds (list[Tensor]): Box scores for direction class
                predictions on each scale level, each is a 4D-tensor,
                the channel number is num_points * 2. (bin = 2)
            depth_cls_preds (list[Tensor]): Box scores for direction class
                predictions on each scale level, each is a 4D-tensor,
                the channel number is num_points * self.num_depth_cls.
            weights (list[Tensor]): Location-aware weights for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * self.weight_dim.
            attr_preds (list[Tensor]): Attribute scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_attrs.
            centernesses (list[Tensor]): Centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_3d (list[Tensor]): 3D boxes ground truth with shape of
                (num_gts, code_size).
            gt_labels_3d (list[Tensor]): same as gt_labels
            centers2d (list[Tensor]): 2D centers on the image with shape of
                (num_gts, 2).
            depths (list[Tensor]): Depth ground truth with shape of
                (num_gts, ).
            attr_labels (list[Tensor]): Attributes indices of each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor]): specify which bounding boxes can
                be ignored when computing the loss. Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(dir_cls_preds) == \
            len(depth_cls_preds) == len(weights) == len(centernesses) == \
            len(attr_preds), 'The length of cls_scores, bbox_preds, ' \
            'dir_cls_preds, depth_cls_preds, weights, centernesses, and' \
            f'attr_preds: {len(cls_scores)}, {len(bbox_preds)}, ' \
            f'{len(dir_cls_preds)}, {len(depth_cls_preds)}, {len(weights)}' \
            f'{len(centernesses)}, {len(attr_preds)} are inconsistent.'
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        labels_3d, bbox_targets_3d, centerness_targets, attr_targets = \
            self.get_targets(
                all_level_points, gt_bboxes, gt_labels, gt_bboxes_3d,
                gt_labels_3d, centers2d, depths, attr_labels)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores and targets
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_labels_3d = torch.cat(labels_3d)
        flatten_bbox_targets_3d = torch.cat(bbox_targets_3d)
        flatten_centerness_targets = torch.cat(centerness_targets)
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])
        if self.pred_attrs:
            flatten_attr_targets = torch.cat(attr_targets)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels_3d >= 0)
                    & (flatten_labels_3d < bg_class_ind)).nonzero().reshape(-1)
        num_pos = len(pos_inds)

        loss_dict = dict()

        loss_dict['loss_cls'] = self.loss_cls(
            flatten_cls_scores,
            flatten_labels_3d,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0

        pos_bbox_preds, pos_dir_cls_preds, pos_depth_cls_preds, pos_weights, \
            pos_attr_preds, pos_centerness = self.get_pos_predictions(
                bbox_preds, dir_cls_preds, depth_cls_preds, weights,
                attr_preds, centernesses, pos_inds, img_metas)

        if num_pos > 0:
            pos_bbox_targets_3d = flatten_bbox_targets_3d[pos_inds]
            pos_centerness_targets = flatten_centerness_targets[pos_inds]
            pos_points = flatten_points[pos_inds]
            if self.pred_attrs:
                pos_attr_targets = flatten_attr_targets[pos_inds]
            if self.use_direction_classifier:
                pos_dir_cls_targets = self.get_direction_target(
                    pos_bbox_targets_3d, self.dir_offset, one_hot=False)

            bbox_weights = pos_centerness_targets.new_ones(
                len(pos_centerness_targets), sum(self.group_reg_dims))
            equal_weights = pos_centerness_targets.new_ones(
                pos_centerness_targets.shape)
            code_weight = self.train_cfg.get('code_weight', None)
            if code_weight:
                assert len(code_weight) == sum(self.group_reg_dims)
                bbox_weights = bbox_weights * bbox_weights.new_tensor(
                    code_weight)

            if self.diff_rad_by_sin:
                pos_bbox_preds, pos_bbox_targets_3d = self.add_sin_difference(
                    pos_bbox_preds, pos_bbox_targets_3d)

            loss_dict['loss_offset'] = self.loss_bbox(
                pos_bbox_preds[:, :2],
                pos_bbox_targets_3d[:, :2],
                weight=bbox_weights[:, :2],
                avg_factor=equal_weights.sum())
            loss_dict['loss_size'] = self.loss_bbox(
                pos_bbox_preds[:, 3:6],
                pos_bbox_targets_3d[:, 3:6],
                weight=bbox_weights[:, 3:6],
                avg_factor=equal_weights.sum())
            loss_dict['loss_rotsin'] = self.loss_bbox(
                pos_bbox_preds[:, 6],
                pos_bbox_targets_3d[:, 6],
                weight=bbox_weights[:, 6],
                avg_factor=equal_weights.sum())
            if self.pred_velo:
                loss_dict['loss_velo'] = self.loss_bbox(
                    pos_bbox_preds[:, 7:9],
                    pos_bbox_targets_3d[:, 7:9],
                    weight=bbox_weights[:, 7:9],
                    avg_factor=equal_weights.sum())

            proj_bbox2d_inputs = (bbox_preds, pos_dir_cls_preds, labels_3d,
                                  bbox_targets_3d, pos_points, pos_inds,
                                  img_metas)

            # direction classification loss
            # TODO: add more check for use_direction_classifier
            if self.use_direction_classifier:
                loss_dict['loss_dir'] = self.loss_dir(
                    pos_dir_cls_preds,
                    pos_dir_cls_targets,
                    equal_weights,
                    avg_factor=equal_weights.sum())

            # init depth loss with the one computed from direct regression
            loss_dict['loss_depth'] = self.loss_bbox(
                pos_bbox_preds[:, 2],
                pos_bbox_targets_3d[:, 2],
                weight=bbox_weights[:, 2],
                avg_factor=equal_weights.sum())
            # depth classification loss
            if self.use_depth_classifier:
                pos_prob_depth_preds = self.bbox_coder.decode_prob_depth(
                    pos_depth_cls_preds, self.depth_range, self.depth_unit,
                    self.division, self.num_depth_cls)
                sig_alpha = torch.sigmoid(self.fuse_lambda)
                if self.weight_dim != -1:
                    loss_fuse_depth = self.loss_depth(
                        sig_alpha * pos_bbox_preds[:, 2] +
                        (1 - sig_alpha) * pos_prob_depth_preds,
                        pos_bbox_targets_3d[:, 2],
                        sigma=pos_weights[:, 0],
                        weight=bbox_weights[:, 2],
                        avg_factor=equal_weights.sum())
                else:
                    loss_fuse_depth = self.loss_depth(
                        sig_alpha * pos_bbox_preds[:, 2] +
                        (1 - sig_alpha) * pos_prob_depth_preds,
                        pos_bbox_targets_3d[:, 2],
                        weight=bbox_weights[:, 2],
                        avg_factor=equal_weights.sum())
                loss_dict['loss_depth'] = loss_fuse_depth

                proj_bbox2d_inputs += (pos_depth_cls_preds, )

            if self.pred_keypoints:
                # use smoothL1 to compute consistency loss for keypoints
                # normalize the offsets with strides
                proj_bbox2d_preds, pos_decoded_bbox2d_preds, kpts_targets = \
                    self.get_proj_bbox2d(*proj_bbox2d_inputs, with_kpts=True)
                loss_dict['loss_kpts'] = self.loss_bbox(
                    pos_bbox_preds[:, self.kpts_start:self.kpts_start + 16],
                    kpts_targets,
                    weight=bbox_weights[:,
                                        self.kpts_start:self.kpts_start + 16],
                    avg_factor=equal_weights.sum())

            if self.pred_bbox2d:
                loss_dict['loss_bbox2d'] = self.loss_bbox2d(
                    pos_bbox_preds[:, -4:],
                    pos_bbox_targets_3d[:, -4:],
                    weight=bbox_weights[:, -4:],
                    avg_factor=equal_weights.sum())
                if not self.pred_keypoints:
                    proj_bbox2d_preds, pos_decoded_bbox2d_preds = \
                        self.get_proj_bbox2d(*proj_bbox2d_inputs)
                loss_dict['loss_consistency'] = self.loss_consistency(
                    proj_bbox2d_preds,
                    pos_decoded_bbox2d_preds,
                    weight=bbox_weights[:, -4:],
                    avg_factor=equal_weights.sum())

            loss_dict['loss_centerness'] = self.loss_centerness(
                pos_centerness, pos_centerness_targets)

            # attribute classification loss
            if self.pred_attrs:
                loss_dict['loss_attr'] = self.loss_attr(
                    pos_attr_preds,
                    pos_attr_targets,
                    pos_centerness_targets,
                    avg_factor=pos_centerness_targets.sum())

        else:
            # need absolute due to possible negative delta x/y
            loss_dict['loss_offset'] = pos_bbox_preds[:, :2].sum()
            loss_dict['loss_size'] = pos_bbox_preds[:, 3:6].sum()
            loss_dict['loss_rotsin'] = pos_bbox_preds[:, 6].sum()
            loss_dict['loss_depth'] = pos_bbox_preds[:, 2].sum()
            if self.pred_velo:
                loss_dict['loss_velo'] = pos_bbox_preds[:, 7:9].sum()
            if self.pred_keypoints:
                loss_dict['loss_kpts'] = pos_bbox_preds[:,
                                                        self.kpts_start:self.
                                                        kpts_start + 16].sum()
            if self.pred_bbox2d:
                loss_dict['loss_bbox2d'] = pos_bbox_preds[:, -4:].sum()
                loss_dict['loss_consistency'] = pos_bbox_preds[:, -4:].sum()
            loss_dict['loss_centerness'] = pos_centerness.sum()
            if self.use_direction_classifier:
                loss_dict['loss_dir'] = pos_dir_cls_preds.sum()
            if self.use_depth_classifier:
                sig_alpha = torch.sigmoid(self.fuse_lambda)
                loss_fuse_depth = \
                    sig_alpha * pos_bbox_preds[:, 2].sum() + \
                    (1 - sig_alpha) * pos_depth_cls_preds.sum()
                if self.weight_dim != -1:
                    loss_fuse_depth *= torch.exp(-pos_weights[:, 0].sum())
                loss_dict['loss_depth'] = loss_fuse_depth
            if self.pred_attrs:
                loss_dict['loss_attr'] = pos_attr_preds.sum()

        return loss_dict

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'dir_cls_preds',
                  'depth_cls_preds', 'weights', 'attr_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   dir_cls_preds,
                   depth_cls_preds,
                   weights,
                   attr_preds,
                   centernesses,
                   img_metas,
                   cfg=None,
                   rescale=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W)
            dir_cls_preds (list[Tensor]): Box scores for direction class
                predictions on each scale level, each is a 4D-tensor,
                the channel number is num_points * 2. (bin = 2)
            depth_cls_preds (list[Tensor]): Box scores for direction class
                predictions on each scale level, each is a 4D-tensor,
                the channel number is num_points * self.num_depth_cls.
            weights (list[Tensor]): Location-aware weights for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * self.weight_dim.
            attr_preds (list[Tensor]): Attribute scores for each scale level
                Has shape (N, num_points * num_attrs, H, W)
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config, optional): Test / postprocessing configuration,
                if None, test_cfg would be used. Defaults to None.
            rescale (bool, optional): If True, return boxes in original image
                space. Defaults to None.

        Returns:
            list[tuple[Tensor]]: Each item in result_list is a tuple, which
                consists of predicted 3D boxes, scores, labels, attributes and
                2D boxes (if necessary).
        """
        assert len(cls_scores) == len(bbox_preds) == len(dir_cls_preds) == \
            len(depth_cls_preds) == len(weights) == len(centernesses) == \
            len(attr_preds), 'The length of cls_scores, bbox_preds, ' \
            'dir_cls_preds, depth_cls_preds, weights, centernesses, and' \
            f'attr_preds: {len(cls_scores)}, {len(bbox_preds)}, ' \
            f'{len(dir_cls_preds)}, {len(depth_cls_preds)}, {len(weights)}' \
            f'{len(centernesses)}, {len(attr_preds)} are inconsistent.'
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            if self.use_direction_classifier:
                dir_cls_pred_list = [
                    dir_cls_preds[i][img_id].detach()
                    for i in range(num_levels)
                ]
            else:
                dir_cls_pred_list = [
                    cls_scores[i][img_id].new_full(
                        [2, *cls_scores[i][img_id].shape[1:]], 0).detach()
                    for i in range(num_levels)
                ]
            if self.use_depth_classifier:
                depth_cls_pred_list = [
                    depth_cls_preds[i][img_id].detach()
                    for i in range(num_levels)
                ]
            else:
                depth_cls_pred_list = [
                    cls_scores[i][img_id].new_full(
                        [self.num_depth_cls, *cls_scores[i][img_id].shape[1:]],
                        0).detach() for i in range(num_levels)
                ]
            if self.weight_dim != -1:
                weight_list = [
                    weights[i][img_id].detach() for i in range(num_levels)
                ]
            else:
                weight_list = [
                    cls_scores[i][img_id].new_full(
                        [1, *cls_scores[i][img_id].shape[1:]], 0).detach()
                    for i in range(num_levels)
                ]
            if self.pred_attrs:
                attr_pred_list = [
                    attr_preds[i][img_id].detach() for i in range(num_levels)
                ]
            else:
                attr_pred_list = [
                    cls_scores[i][img_id].new_full(
                        [self.num_attrs, *cls_scores[i][img_id].shape[1:]],
                        self.attr_background_label).detach()
                    for i in range(num_levels)
                ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            input_meta = img_metas[img_id]
            det_bboxes = self._get_bboxes_single(
                cls_score_list, bbox_pred_list, dir_cls_pred_list,
                depth_cls_pred_list, weight_list, attr_pred_list,
                centerness_pred_list, mlvl_points, input_meta, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           dir_cls_preds,
                           depth_cls_preds,
                           weights,
                           attr_preds,
                           centernesses,
                           mlvl_points,
                           input_meta,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                Has shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * bbox_code_size, H, W).
            dir_cls_preds (list[Tensor]): Box scores for direction class
                predictions on a single scale level with shape
                (num_points * 2, H, W)
            depth_cls_preds (list[Tensor]): Box scores for probabilistic depth
                predictions on a single scale level with shape
                (num_points * self.num_depth_cls, H, W)
            weights (list[Tensor]): Location-aware weight maps on a single
                scale level with shape (num_points * self.weight_dim, H, W).
            attr_preds (list[Tensor]): Attribute scores for each scale level
                Has shape (N, num_points * num_attrs, H, W)
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 2).
            input_meta (dict): Metadata of input image.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool, optional): If True, return boxes in original image
                space. Defaults to False.

        Returns:
            tuples[Tensor]: Predicted 3D boxes, scores, labels, attributes and
                2D boxes (if necessary).
        """
        view = np.array(input_meta['cam2img'])
        scale_factor = input_meta['scale_factor']
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_centers2d = []
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []
        mlvl_attr_scores = []
        mlvl_centerness = []
        mlvl_depth_cls_scores = []
        mlvl_depth_uncertainty = []
        mlvl_bboxes2d = None
        if self.pred_bbox2d:
            mlvl_bboxes2d = []

        for cls_score, bbox_pred, dir_cls_pred, depth_cls_pred, weight, \
                attr_pred, centerness, points in zip(
                    cls_scores, bbox_preds, dir_cls_preds, depth_cls_preds,
                    weights, attr_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]
            depth_cls_pred = depth_cls_pred.permute(1, 2, 0).reshape(
                -1, self.num_depth_cls)
            depth_cls_score = F.softmax(
                depth_cls_pred, dim=-1).topk(
                    k=2, dim=-1)[0].mean(dim=-1)
            if self.weight_dim != -1:
                weight = weight.permute(1, 2, 0).reshape(-1, self.weight_dim)
            else:
                weight = weight.permute(1, 2, 0).reshape(-1, 1)
            depth_uncertainty = torch.exp(-weight[:, -1])
            attr_pred = attr_pred.permute(1, 2, 0).reshape(-1, self.num_attrs)
            attr_score = torch.max(attr_pred, dim=-1)[1]
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2,
                                          0).reshape(-1,
                                                     sum(self.group_reg_dims))
            bbox_pred3d = bbox_pred[:, :self.bbox_coder.bbox_code_size]
            if self.pred_bbox2d:
                bbox_pred2d = bbox_pred[:, -4:]
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                merged_scores = scores * centerness[:, None]
                if self.use_depth_classifier:
                    merged_scores *= depth_cls_score[:, None]
                    if self.weight_dim != -1:
                        merged_scores *= depth_uncertainty[:, None]
                max_scores, _ = merged_scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred3d = bbox_pred3d[topk_inds, :]
                scores = scores[topk_inds, :]
                dir_cls_pred = dir_cls_pred[topk_inds, :]
                depth_cls_pred = depth_cls_pred[topk_inds, :]
                centerness = centerness[topk_inds]
                dir_cls_score = dir_cls_score[topk_inds]
                depth_cls_score = depth_cls_score[topk_inds]
                depth_uncertainty = depth_uncertainty[topk_inds]
                attr_score = attr_score[topk_inds]
                if self.pred_bbox2d:
                    bbox_pred2d = bbox_pred2d[topk_inds, :]
            # change the offset to actual center predictions
            bbox_pred3d[:, :2] = points - bbox_pred3d[:, :2]
            if rescale:
                bbox_pred3d[:, :2] /= bbox_pred3d[:, :2].new_tensor(
                    scale_factor)
                if self.pred_bbox2d:
                    bbox_pred2d /= bbox_pred2d.new_tensor(scale_factor)
            if self.use_depth_classifier:
                prob_depth_pred = self.bbox_coder.decode_prob_depth(
                    depth_cls_pred, self.depth_range, self.depth_unit,
                    self.division, self.num_depth_cls)
                sig_alpha = torch.sigmoid(self.fuse_lambda)
                bbox_pred3d[:, 2] = sig_alpha * bbox_pred3d[:, 2] + \
                    (1 - sig_alpha) * prob_depth_pred
            pred_center2d = bbox_pred3d[:, :3].clone()
            bbox_pred3d[:, :3] = points_img2cam(bbox_pred3d[:, :3], view)
            mlvl_centers2d.append(pred_center2d)
            mlvl_bboxes.append(bbox_pred3d)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)
            mlvl_depth_cls_scores.append(depth_cls_score)
            mlvl_attr_scores.append(attr_score)
            mlvl_centerness.append(centerness)
            mlvl_depth_uncertainty.append(depth_uncertainty)
            if self.pred_bbox2d:
                bbox_pred2d = distance2bbox(
                    points, bbox_pred2d, max_shape=input_meta['img_shape'])
                mlvl_bboxes2d.append(bbox_pred2d)

        mlvl_centers2d = torch.cat(mlvl_centers2d)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)
        if self.pred_bbox2d:
            mlvl_bboxes2d = torch.cat(mlvl_bboxes2d)

        # change local yaw to global yaw for 3D nms
        cam2img = torch.eye(
            4, dtype=mlvl_centers2d.dtype, device=mlvl_centers2d.device)
        cam2img[:view.shape[0], :view.shape[1]] = \
            mlvl_centers2d.new_tensor(view)
        mlvl_bboxes = self.bbox_coder.decode_yaw(mlvl_bboxes, mlvl_centers2d,
                                                 mlvl_dir_scores,
                                                 self.dir_offset, cam2img)

        mlvl_bboxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d'](
            mlvl_bboxes,
            box_dim=self.bbox_coder.bbox_code_size,
            origin=(0.5, 0.5, 0.5)).bev)

        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_attr_scores = torch.cat(mlvl_attr_scores)
        mlvl_centerness = torch.cat(mlvl_centerness)
        # no scale_factors in box3d_multiclass_nms
        # Then we multiply it from outside
        mlvl_nms_scores = mlvl_scores * mlvl_centerness[:, None]
        if self.use_depth_classifier:  # multiply the depth confidence
            mlvl_depth_cls_scores = torch.cat(mlvl_depth_cls_scores)
            mlvl_nms_scores *= mlvl_depth_cls_scores[:, None]
            if self.weight_dim != -1:
                mlvl_depth_uncertainty = torch.cat(mlvl_depth_uncertainty)
                mlvl_nms_scores *= mlvl_depth_uncertainty[:, None]
        results = box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
                                       mlvl_nms_scores, cfg.score_thr,
                                       cfg.max_per_img, cfg, mlvl_dir_scores,
                                       mlvl_attr_scores, mlvl_bboxes2d)
        bboxes, scores, labels, dir_scores, attrs = results[0:5]
        attrs = attrs.to(labels.dtype)  # change data type to int
        bboxes = input_meta['box_type_3d'](
            bboxes,
            box_dim=self.bbox_coder.bbox_code_size,
            origin=(0.5, 0.5, 0.5))
        # Note that the predictions use origin (0.5, 0.5, 0.5)
        # Due to the ground truth centers2d are the gravity center of objects
        # v0.10.0 fix inplace operation to the input tensor of cam_box3d
        # So here we also need to add origin=(0.5, 0.5, 0.5)
        if not self.pred_attrs:
            attrs = None

        outputs = (bboxes, scores, labels, attrs)
        if self.pred_bbox2d:
            bboxes2d = results[-1]
            bboxes2d = torch.cat([bboxes2d, scores[:, None]], dim=1)
            outputs = outputs + (bboxes2d, )

        return outputs

    def get_targets(self, points, gt_bboxes_list, gt_labels_list,
                    gt_bboxes_3d_list, gt_labels_3d_list, centers2d_list,
                    depths_list, attr_labels_list):
        """Compute regression, classification and centerss targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
            gt_bboxes_3d_list (list[Tensor]): 3D Ground truth bboxes of each
                image, each has shape (num_gt, bbox_code_size).
            gt_labels_3d_list (list[Tensor]): 3D Ground truth labels of each
                box, each has shape (num_gt,).
            centers2d_list (list[Tensor]): Projected 3D centers onto 2D image,
                each has shape (num_gt, 2).
            depths_list (list[Tensor]): Depth of projected 3D centers onto 2D
                image, each has shape (num_gt, 1).
            attr_labels_list (list[Tensor]): Attribute labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        if attr_labels_list is None:
            attr_labels_list = [
                gt_labels.new_full(gt_labels.shape, self.attr_background_label)
                for gt_labels in gt_labels_list
            ]

        # get labels and bbox_targets of each image
        _, bbox_targets_list, labels_3d_list, bbox_targets_3d_list, \
            centerness_targets_list, attr_targets_list = multi_apply(
                self._get_target_single,
                gt_bboxes_list,
                gt_labels_list,
                gt_bboxes_3d_list,
                gt_labels_3d_list,
                centers2d_list,
                depths_list,
                attr_labels_list,
                points=concat_points,
                regress_ranges=concat_regress_ranges,
                num_points_per_lvl=num_points)

        # split to per img, per level
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        labels_3d_list = [
            labels_3d.split(num_points, 0) for labels_3d in labels_3d_list
        ]
        bbox_targets_3d_list = [
            bbox_targets_3d.split(num_points, 0)
            for bbox_targets_3d in bbox_targets_3d_list
        ]
        centerness_targets_list = [
            centerness_targets.split(num_points, 0)
            for centerness_targets in centerness_targets_list
        ]
        attr_targets_list = [
            attr_targets.split(num_points, 0)
            for attr_targets in attr_targets_list
        ]

        # concat per level image
        concat_lvl_labels_3d = []
        concat_lvl_bbox_targets_3d = []
        concat_lvl_centerness_targets = []
        concat_lvl_attr_targets = []
        for i in range(num_levels):
            concat_lvl_labels_3d.append(
                torch.cat([labels[i] for labels in labels_3d_list]))
            concat_lvl_centerness_targets.append(
                torch.cat([
                    centerness_targets[i]
                    for centerness_targets in centerness_targets_list
                ]))
            bbox_targets_3d = torch.cat([
                bbox_targets_3d[i] for bbox_targets_3d in bbox_targets_3d_list
            ])
            if self.pred_bbox2d:
                bbox_targets = torch.cat(
                    [bbox_targets[i] for bbox_targets in bbox_targets_list])
                bbox_targets_3d = torch.cat([bbox_targets_3d, bbox_targets],
                                            dim=1)
            concat_lvl_attr_targets.append(
                torch.cat(
                    [attr_targets[i] for attr_targets in attr_targets_list]))
            if self.norm_on_bbox:
                bbox_targets_3d[:, :2] = \
                    bbox_targets_3d[:, :2] / self.strides[i]
                if self.pred_bbox2d:
                    bbox_targets_3d[:, -4:] = \
                        bbox_targets_3d[:, -4:] / self.strides[i]
            concat_lvl_bbox_targets_3d.append(bbox_targets_3d)
        return concat_lvl_labels_3d, concat_lvl_bbox_targets_3d, \
            concat_lvl_centerness_targets, concat_lvl_attr_targets
