# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from mmcv.cnn import Scale
from mmdet.models.utils import multi_apply, select_single_mlvl
from mmengine.model import normal_init
from mmengine.structures import InstanceData
from torch import Tensor
from torch import nn as nn

from mmdet3d.models.layers import box3d_multiclass_nms
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures import limit_period, points_img2cam, xywhr2xyxyr
from mmdet3d.utils import (ConfigType, InstanceList, OptConfigType,
                           OptInstanceList)
from .anchor_free_mono3d_head import AnchorFreeMono3DHead

RangeType = Sequence[Tuple[int, int]]

INF = 1e8


@MODELS.register_module()
class FCOSMono3DHead(AnchorFreeMono3DHead):
    """Anchor-free head used in FCOS3D.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        regress_ranges (Sequence[Tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: True.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: True.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to
            https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: True.
        centerness_alpha (float): Parameter used to adjust the intensity
            attenuation from the center to the periphery. Default: 2.5.
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        loss_dir (:obj:`ConfigDict` or dict): Config of direction classification loss.
        loss_attr (:obj:`ConfigDict` or dict): Config of attribute classification loss.
        loss_centerness (:obj:`ConfigDict` or dict): Config of centerness loss.
        norm_cfg (:obj:`ConfigDict` or dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        centerness_branch (tuple[int]): Channels for centerness branch.
            Default: (64, ).
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.
    """  # noqa: E501

    def __init__(self,
                 regress_ranges: RangeType = ((-1, 48), (48, 96), (96, 192),
                                              (192, 384), (384, INF)),
                 center_sampling: bool = True,
                 center_sample_radius: float = 1.5,
                 norm_on_bbox: bool = True,
                 centerness_on_reg: bool = True,
                 centerness_alpha: float = 2.5,
                 loss_cls: ConfigType = dict(
                     type='mmdet.FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(
                     type='mmdet.SmoothL1Loss',
                     beta=1.0 / 9.0,
                     loss_weight=1.0),
                 loss_dir: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_attr: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_centerness: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 bbox_coder: ConfigType = dict(
                     type='FCOS3DBBoxCoder', code_size=9),
                 norm_cfg: ConfigType = dict(
                     type='GN', num_groups=32, requires_grad=True),
                 centerness_branch: Tuple[int] = (64, ),
                 init_cfg: OptConfigType = None,
                 **kwargs) -> None:
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        self.centerness_alpha = centerness_alpha
        self.centerness_branch = centerness_branch
        super().__init__(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_dir=loss_dir,
            loss_attr=loss_attr,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_centerness = MODELS.build(loss_centerness)
        bbox_coder['code_size'] = self.bbox_code_size
        self.bbox_coder = TASK_UTILS.build(bbox_coder)

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_centerness_prev = self._init_branch(
            conv_channels=self.centerness_branch,
            conv_strides=(1, ) * len(self.centerness_branch))
        self.conv_centerness = nn.Conv2d(self.centerness_branch[-1], 1, 1)
        self.scale_dim = 3  # only for offset, depth and size regression
        self.scales = nn.ModuleList([
            nn.ModuleList([Scale(1.0) for _ in range(self.scale_dim)])
            for _ in self.strides
        ])

    def init_weights(self):
        """Initialize weights of the head.

        We currently still use the customized init_weights because the default
        init of DCN triggered by the init_cfg will init conv_offset.weight,
        which mistakenly affects the training stability.
        """
        super().init_weights()
        for m in self.conv_centerness_prev:
            if isinstance(m.conv, nn.Conv2d):
                normal_init(m.conv, std=0.01)
        normal_init(self.conv_centerness, std=0.01)

    def forward(
        self, x: Tuple[Tensor]
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor],
               List[Tensor]]:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
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
                attr_preds (list[Tensor]): Attribute scores for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * num_attrs.
                centernesses (list[Tensor]): Centerness for each scale level,
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        # Note: we use [:5] to filter feats and only return predictions
        return multi_apply(self.forward_single, x, self.scales,
                           self.strides)[:5]

    def forward_single(self, x: Tensor, scale: Scale,
                       stride: int) -> Tuple[Tensor, ...]:
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
                predictions, centerness predictions of input feature maps.
        """
        cls_score, bbox_pred, dir_cls_pred, attr_pred, cls_feat, reg_feat = \
            super().forward_single(x)

        if self.centerness_on_reg:
            clone_reg_feat = reg_feat.clone()
            for conv_centerness_prev_layer in self.conv_centerness_prev:
                clone_reg_feat = conv_centerness_prev_layer(clone_reg_feat)
            centerness = self.conv_centerness(clone_reg_feat)
        else:
            clone_cls_feat = cls_feat.clone()
            for conv_centerness_prev_layer in self.conv_centerness_prev:
                clone_cls_feat = conv_centerness_prev_layer(clone_cls_feat)
            centerness = self.conv_centerness(clone_cls_feat)

        bbox_pred = self.bbox_coder.decode(bbox_pred, scale, stride,
                                           self.training, cls_score)

        return cls_score, bbox_pred, dir_cls_pred, attr_pred, centerness, \
            cls_feat, reg_feat

    @staticmethod
    def add_sin_difference(boxes1: Tensor,
                           boxes2: Tensor) -> Tuple[Tensor, Tensor]:
        """Convert the rotation difference to difference in sine function.

        Args:
            boxes1 (torch.Tensor): Original Boxes in shape (NxC), where C>=7
                and the 7th dimension is rotation dimension.
            boxes2 (torch.Tensor): Target boxes in shape (NxC), where C>=7 and
                the 7th dimension is rotation dimension.

        Returns:
            tuple[torch.Tensor]: ``boxes1`` and ``boxes2`` whose 7th
                dimensions are changed.
        """
        rad_pred_encoding = torch.sin(boxes1[..., 6:7]) * torch.cos(
            boxes2[..., 6:7])
        rad_tg_encoding = torch.cos(boxes1[..., 6:7]) * torch.sin(boxes2[...,
                                                                         6:7])
        boxes1 = torch.cat(
            [boxes1[..., :6], rad_pred_encoding, boxes1[..., 7:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :6], rad_tg_encoding, boxes2[..., 7:]],
                           dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(reg_targets: Tensor,
                             dir_offset: int = 0,
                             dir_limit_offset: float = 0.0,
                             num_bins: int = 2,
                             one_hot: bool = True) -> Tensor:
        """Encode direction to 0 ~ num_bins-1.

        Args:
            reg_targets (torch.Tensor): Bbox regression targets.
            dir_offset (int, optional): Direction offset. Default to 0.
            dir_limit_offset (float, optional): Offset to set the direction
                range. Default to 0.0.
            num_bins (int, optional): Number of bins to divide 2*PI.
                Default to 2.
            one_hot (bool, optional): Whether to encode as one hot.
                Default to True.

        Returns:
            torch.Tensor: Encoded direction targets.
        """
        rot_gt = reg_targets[..., 6]
        offset_rot = limit_period(rot_gt - dir_offset, dir_limit_offset,
                                  2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot /
                                      (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
        if one_hot:
            dir_targets = torch.zeros(
                *list(dir_cls_targets.shape),
                num_bins,
                dtype=reg_targets.dtype,
                device=dir_cls_targets.device)
            dir_targets.scatter_(dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            dir_cls_preds: List[Tensor],
            attr_preds: List[Tensor],
            centernesses: List[Tensor],
            batch_gt_instances_3d: InstanceList,
            batch_gt_instacnes: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
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
            attr_preds (list[Tensor]): Attribute scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_attrs.
            centernesses (list[Tensor]): Centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instance_3d.  It usually includes ``bboxes_3d``、`
                `labels_3d``、``depths``、``centers_2d`` and attributes.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes``、``labels``.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(centernesses) == len(
            attr_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        labels_3d, bbox_targets_3d, centerness_targets, attr_targets = \
            self.get_targets(all_level_points, batch_gt_instances_3d,
                             batch_gt_instacnes)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds, dir_cls_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
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
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_dir_cls_preds = torch.cat(flatten_dir_cls_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels_3d = torch.cat(labels_3d)
        flatten_bbox_targets_3d = torch.cat(bbox_targets_3d)
        flatten_centerness_targets = torch.cat(centerness_targets)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels_3d >= 0)
                    & (flatten_labels_3d < bg_class_ind)).nonzero().reshape(-1)
        num_pos = len(pos_inds)

        loss_cls = self.loss_cls(
            flatten_cls_scores,
            flatten_labels_3d,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_dir_cls_preds = flatten_dir_cls_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]

        if self.pred_attrs:
            flatten_attr_preds = [
                attr_pred.permute(0, 2, 3, 1).reshape(-1, self.num_attrs)
                for attr_pred in attr_preds
            ]
            flatten_attr_preds = torch.cat(flatten_attr_preds)
            flatten_attr_targets = torch.cat(attr_targets)
            pos_attr_preds = flatten_attr_preds[pos_inds]

        if num_pos > 0:
            pos_bbox_targets_3d = flatten_bbox_targets_3d[pos_inds]
            pos_centerness_targets = flatten_centerness_targets[pos_inds]
            if self.pred_attrs:
                pos_attr_targets = flatten_attr_targets[pos_inds]
            bbox_weights = pos_centerness_targets.new_ones(
                len(pos_centerness_targets), sum(self.group_reg_dims))
            equal_weights = pos_centerness_targets.new_ones(
                pos_centerness_targets.shape)

            code_weight = self.train_cfg.get('code_weight', None)
            if code_weight:
                assert len(code_weight) == sum(self.group_reg_dims)
                bbox_weights = bbox_weights * bbox_weights.new_tensor(
                    code_weight)

            if self.use_direction_classifier:
                pos_dir_cls_targets = self.get_direction_target(
                    pos_bbox_targets_3d,
                    self.dir_offset,
                    self.dir_limit_offset,
                    one_hot=False)

            if self.diff_rad_by_sin:
                pos_bbox_preds, pos_bbox_targets_3d = self.add_sin_difference(
                    pos_bbox_preds, pos_bbox_targets_3d)

            loss_offset = self.loss_bbox(
                pos_bbox_preds[:, :2],
                pos_bbox_targets_3d[:, :2],
                weight=bbox_weights[:, :2],
                avg_factor=equal_weights.sum())
            loss_depth = self.loss_bbox(
                pos_bbox_preds[:, 2],
                pos_bbox_targets_3d[:, 2],
                weight=bbox_weights[:, 2],
                avg_factor=equal_weights.sum())
            loss_size = self.loss_bbox(
                pos_bbox_preds[:, 3:6],
                pos_bbox_targets_3d[:, 3:6],
                weight=bbox_weights[:, 3:6],
                avg_factor=equal_weights.sum())
            loss_rotsin = self.loss_bbox(
                pos_bbox_preds[:, 6],
                pos_bbox_targets_3d[:, 6],
                weight=bbox_weights[:, 6],
                avg_factor=equal_weights.sum())
            loss_velo = None
            if self.pred_velo:
                loss_velo = self.loss_bbox(
                    pos_bbox_preds[:, 7:9],
                    pos_bbox_targets_3d[:, 7:9],
                    weight=bbox_weights[:, 7:9],
                    avg_factor=equal_weights.sum())

            loss_centerness = self.loss_centerness(pos_centerness,
                                                   pos_centerness_targets)

            # direction classification loss
            loss_dir = None
            # TODO: add more check for use_direction_classifier
            if self.use_direction_classifier:
                loss_dir = self.loss_dir(
                    pos_dir_cls_preds,
                    pos_dir_cls_targets,
                    equal_weights,
                    avg_factor=equal_weights.sum())

            # attribute classification loss
            loss_attr = None
            if self.pred_attrs:
                loss_attr = self.loss_attr(
                    pos_attr_preds,
                    pos_attr_targets,
                    pos_centerness_targets,
                    avg_factor=pos_centerness_targets.sum())

        else:
            # need absolute due to possible negative delta x/y
            loss_offset = pos_bbox_preds[:, :2].sum()
            loss_depth = pos_bbox_preds[:, 2].sum()
            loss_size = pos_bbox_preds[:, 3:6].sum()
            loss_rotsin = pos_bbox_preds[:, 6].sum()
            loss_velo = None
            if self.pred_velo:
                loss_velo = pos_bbox_preds[:, 7:9].sum()
            loss_centerness = pos_centerness.sum()
            loss_dir = None
            if self.use_direction_classifier:
                loss_dir = pos_dir_cls_preds.sum()
            loss_attr = None
            if self.pred_attrs:
                loss_attr = pos_attr_preds.sum()

        loss_dict = dict(
            loss_cls=loss_cls,
            loss_offset=loss_offset,
            loss_depth=loss_depth,
            loss_size=loss_size,
            loss_rotsin=loss_rotsin,
            loss_centerness=loss_centerness)

        if loss_velo is not None:
            loss_dict['loss_velo'] = loss_velo

        if loss_dir is not None:
            loss_dict['loss_dir'] = loss_dir

        if loss_attr is not None:
            loss_dict['loss_attr'] = loss_attr

        return loss_dict

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        dir_cls_preds: List[Tensor],
                        attr_preds: List[Tensor],
                        centernesses: List[Tensor],
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: OptConfigType = None,
                        rescale: bool = False) -> InstanceList:
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W)
            dir_cls_preds (list[Tensor]): Box scores for direction class
                predictions on each scale level, each is a 4D-tensor,
                the channel number is num_points * 2. (bin = 2)
            attr_preds (list[Tensor]): Attribute scores for each scale level
                Has shape (N, num_points * num_attrs, H, W)
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W)
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels_3d (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes_3d (Tensor): Contains a tensor with shape
                  (num_instances, C), where C >= 7.
        """
        assert len(cls_scores) == len(bbox_preds) == len(dir_cls_preds) == \
            len(centernesses) == len(attr_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        # TODO: refactor using prior_generator
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)

            if self.use_direction_classifier:
                dir_cls_pred_list = select_single_mlvl(dir_cls_preds, img_id)
            else:
                dir_cls_pred_list = [
                    cls_scores[i][img_id].new_full(
                        [2, *cls_scores[i][img_id].shape[1:]], 0).detach()
                    for i in range(num_levels)
                ]

            if self.pred_attrs:
                attr_pred_list = select_single_mlvl(attr_preds, img_id)
            else:
                attr_pred_list = [
                    cls_scores[i][img_id].new_full(
                        [self.num_attrs, *cls_scores[i][img_id].shape[1:]],
                        self.attr_background_label).detach()
                    for i in range(num_levels)
                ]

            centerness_pred_list = select_single_mlvl(centernesses, img_id)
            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                dir_cls_pred_list=dir_cls_pred_list,
                attr_pred_list=attr_pred_list,
                centerness_pred_list=centerness_pred_list,
                mlvl_points=mlvl_points,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale)
            result_list.append(results)
        result_list_2d = None
        return result_list, result_list_2d

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                dir_cls_pred_list: List[Tensor],
                                attr_pred_list: List[Tensor],
                                centerness_pred_list: List[Tensor],
                                mlvl_points: Tensor,
                                img_meta: dict,
                                cfg: ConfigType,
                                rescale: bool = False) -> InstanceData:
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                Has shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * bbox_code_size, H, W).
            dir_cls_preds (list[Tensor]): Box scores for direction class
                predictions on a single scale level with shape
                (num_points * 2, H, W)
            attr_preds (list[Tensor]): Attribute scores for each scale level
                Has shape (N, num_points * num_attrs, H, W)
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 2).
            img_meta (dict): Metadata of input image.
            cfg (mmengine.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            :obj:`InstanceData`: 3D Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels_3d (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes_3d (Tensor): Contains a tensor with shape
                  (num_instances, C), where C >= 7.
        """
        view = np.array(img_meta['cam2img'])
        scale_factor = img_meta['scale_factor']
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_points)
        mlvl_centers_2d = []
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []
        mlvl_attr_scores = []
        mlvl_centerness = []

        for cls_score, bbox_pred, dir_cls_pred, attr_pred, centerness, \
                points in zip(cls_score_list, bbox_pred_list,
                              dir_cls_pred_list, attr_pred_list,
                              centerness_pred_list, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]
            attr_pred = attr_pred.permute(1, 2, 0).reshape(-1, self.num_attrs)
            attr_score = torch.max(attr_pred, dim=-1)[1]
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2,
                                          0).reshape(-1,
                                                     sum(self.group_reg_dims))
            bbox_pred = bbox_pred[:, :self.bbox_code_size]
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                dir_cls_pred = dir_cls_pred[topk_inds, :]
                centerness = centerness[topk_inds]
                dir_cls_score = dir_cls_score[topk_inds]
                attr_score = attr_score[topk_inds]
            # change the offset to actual center predictions
            bbox_pred[:, :2] = points - bbox_pred[:, :2]
            if rescale:
                bbox_pred[:, :2] /= bbox_pred[:, :2].new_tensor(scale_factor)
            pred_center2d = bbox_pred[:, :3].clone()
            bbox_pred[:, :3] = points_img2cam(bbox_pred[:, :3], view)
            mlvl_centers_2d.append(pred_center2d)
            mlvl_bboxes.append(bbox_pred)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)
            mlvl_attr_scores.append(attr_score)
            mlvl_centerness.append(centerness)

        mlvl_centers_2d = torch.cat(mlvl_centers_2d)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)

        # change local yaw to global yaw for 3D nms
        cam2img = mlvl_centers_2d.new_zeros((4, 4))
        cam2img[:view.shape[0], :view.shape[1]] = \
            mlvl_centers_2d.new_tensor(view)
        mlvl_bboxes = self.bbox_coder.decode_yaw(mlvl_bboxes, mlvl_centers_2d,
                                                 mlvl_dir_scores,
                                                 self.dir_offset, cam2img)

        mlvl_bboxes_for_nms = xywhr2xyxyr(img_meta['box_type_3d'](
            mlvl_bboxes, box_dim=self.bbox_code_size,
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
        results = box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
                                       mlvl_nms_scores, cfg.score_thr,
                                       cfg.max_per_img, cfg, mlvl_dir_scores,
                                       mlvl_attr_scores)
        bboxes, scores, labels, dir_scores, attrs = results
        attrs = attrs.to(labels.dtype)  # change data type to int
        bboxes = img_meta['box_type_3d'](
            bboxes, box_dim=self.bbox_code_size, origin=(0.5, 0.5, 0.5))
        # Note that the predictions use origin (0.5, 0.5, 0.5)
        # Due to the ground truth centers_2d are the gravity center of objects
        # v0.10.0 fix inplace operation to the input tensor of cam_box3d
        # So here we also need to add origin=(0.5, 0.5, 0.5)

        results = InstanceData()
        results.bboxes_3d = bboxes
        results.scores_3d = scores
        results.labels_3d = labels
        if self.pred_attrs and attrs is not None:
            results.attr_labels = attrs

        return results

    def _get_points_single(self,
                           featmap_size: Tuple[int],
                           stride: int,
                           dtype: torch.dtype,
                           device: torch.device,
                           flatten: bool = False) -> Tensor:
        """Get points of a single scale level.

        Args:
            featmap_size (tuple[int]): Single scale level feature map size.
            stride (int): Downsample factor of the feature map.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.
            flatten (bool): Whether to flatten the tensor.
                Defaults to False.

        Returns:
            Tensor: points of each image.
        """
        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points

    def get_targets(
        self,
        points: List[Tensor],
        batch_gt_instances_3d: InstanceList,
        batch_gt_instances: InstanceList,
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        """Compute regression, classification and centerss targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instance_3d.  It usually includes ``bboxes_3d``、
                ``labels_3d``、``depths``、``centers_2d`` and attributes.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes``、``labels``.

        Returns:
            tuple: Targets of each level.

            - concat_lvl_labels_3d (list[Tensor]): 3D Labels of each level.
            - concat_lvl_bbox_targets_3d (list[Tensor]): 3D BBox targets of
                each level.
            - concat_lvl_centerness_targets (list[Tensor]): Centerness targets
                of each level.
            - concat_lvl_attr_targets (list[Tensor]): Attribute targets of
                each level.
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

        if 'attr_labels' not in batch_gt_instances_3d[0]:
            for gt_instances_3d in batch_gt_instances_3d:
                gt_instances_3d.attr_labels = \
                    gt_instances_3d.labels_3d.new_full(
                        gt_instances_3d.labels_3d.shape,
                        self.attr_background_label
                    )

        # get labels and bbox_targets of each image
        _, _, labels_3d_list, bbox_targets_3d_list, centerness_targets_list, \
            attr_targets_list = multi_apply(
                self._get_target_single,
                batch_gt_instances_3d,
                batch_gt_instances,
                points=concat_points,
                regress_ranges=concat_regress_ranges,
                num_points_per_lvl=num_points)

        # split to per img, per level
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
            concat_lvl_attr_targets.append(
                torch.cat(
                    [attr_targets[i] for attr_targets in attr_targets_list]))
            if self.norm_on_bbox:
                bbox_targets_3d[:, :
                                2] = bbox_targets_3d[:, :2] / self.strides[i]
            concat_lvl_bbox_targets_3d.append(bbox_targets_3d)
        return concat_lvl_labels_3d, concat_lvl_bbox_targets_3d, \
            concat_lvl_centerness_targets, concat_lvl_attr_targets

    def _get_target_single(
            self, gt_instances_3d: InstanceData, gt_instances: InstanceData,
            points: Tensor, regress_ranges: Tensor,
            num_points_per_lvl: List[int]) -> Tuple[Tensor, ...]:
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = len(gt_instances_3d)
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        gt_bboxes_3d = gt_instances_3d.bboxes_3d
        gt_labels_3d = gt_instances_3d.labels_3d
        centers_2d = gt_instances_3d.centers_2d
        depths = gt_instances_3d.depths
        attr_labels = gt_instances_3d.attr_labels

        if not isinstance(gt_bboxes_3d, torch.Tensor):
            gt_bboxes_3d = gt_bboxes_3d.tensor.to(gt_bboxes.device)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.background_label), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_labels_3d.new_full(
                       (num_points,), self.background_label), \
                   gt_bboxes_3d.new_zeros((num_points, self.bbox_code_size)), \
                   gt_bboxes_3d.new_zeros((num_points,)), \
                   attr_labels.new_full(
                       (num_points,), self.attr_background_label)

        # change orientation to local yaw
        gt_bboxes_3d[..., 6] = -torch.atan2(
            gt_bboxes_3d[..., 0], gt_bboxes_3d[..., 2]) + gt_bboxes_3d[..., 6]

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        centers_2d = centers_2d[None].expand(num_points, num_gts, 2)
        gt_bboxes_3d = gt_bboxes_3d[None].expand(num_points, num_gts,
                                                 self.bbox_code_size)
        depths = depths[None, :, None].expand(num_points, num_gts, 1)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        delta_xs = (xs - centers_2d[..., 0])[..., None]
        delta_ys = (ys - centers_2d[..., 1])[..., None]
        bbox_targets_3d = torch.cat(
            (delta_xs, delta_ys, depths, gt_bboxes_3d[..., 3:]), dim=-1)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        assert self.center_sampling is True, 'Setting center_sampling to '\
            'False has not been implemented for FCOS3D.'
        # condition1: inside a `center bbox`
        radius = self.center_sample_radius
        center_xs = centers_2d[..., 0]
        center_ys = centers_2d[..., 1]
        center_gts = torch.zeros_like(gt_bboxes)
        stride = center_xs.new_zeros(center_xs.shape)

        # project the points on current lvl back to the `original` sizes
        lvl_begin = 0
        for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
            lvl_end = lvl_begin + num_points_lvl
            stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
            lvl_begin = lvl_end

        center_gts[..., 0] = center_xs - stride
        center_gts[..., 1] = center_ys - stride
        center_gts[..., 2] = center_xs + stride
        center_gts[..., 3] = center_ys + stride

        cb_dist_left = xs - center_gts[..., 0]
        cb_dist_right = center_gts[..., 2] - xs
        cb_dist_top = ys - center_gts[..., 1]
        cb_dist_bottom = center_gts[..., 3] - ys
        center_bbox = torch.stack(
            (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # center-based criterion to deal with ambiguity
        dists = torch.sqrt(torch.sum(bbox_targets_3d[..., :2]**2, dim=-1))
        dists[inside_gt_bbox_mask == 0] = INF
        dists[inside_regress_range == 0] = INF
        min_dist, min_dist_inds = dists.min(dim=1)

        labels = gt_labels[min_dist_inds]
        labels_3d = gt_labels_3d[min_dist_inds]
        attr_labels = attr_labels[min_dist_inds]
        labels[min_dist == INF] = self.background_label  # set as BG
        labels_3d[min_dist == INF] = self.background_label  # set as BG
        attr_labels[min_dist == INF] = self.attr_background_label

        bbox_targets = bbox_targets[range(num_points), min_dist_inds]
        bbox_targets_3d = bbox_targets_3d[range(num_points), min_dist_inds]
        relative_dists = torch.sqrt(
            torch.sum(bbox_targets_3d[..., :2]**2,
                      dim=-1)) / (1.414 * stride[:, 0])
        # [N, 1] / [N, 1]
        centerness_targets = torch.exp(-self.centerness_alpha * relative_dists)

        return labels, bbox_targets, labels_3d, bbox_targets_3d, \
            centerness_targets, attr_labels
