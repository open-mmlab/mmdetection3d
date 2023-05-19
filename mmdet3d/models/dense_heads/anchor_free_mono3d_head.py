# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import Any, List, Sequence, Tuple, Union

import torch
from mmcv.cnn import ConvModule
from mmdet.models.utils import multi_apply
from mmengine.model import bias_init_with_prob, normal_init
from torch import Tensor
from torch import nn as nn

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, InstanceList, OptConfigType
from .base_mono3d_dense_head import BaseMono3DDenseHead


@MODELS.register_module()
class AnchorFreeMono3DHead(BaseMono3DDenseHead):
    """Anchor-free head for monocular 3D object detection.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels.
            Used in child classes. Defaults to 256.
        stacked_convs (int): Number of stacking convs of the head.
        strides (Sequence[int] or Sequence[Tuple[int, int]]): Downsample
            factor of each feature map.
        dcn_on_last_conv (bool): If true, use dcn in the last
            layer of towers. Default: False.
        conv_bias (bool or str): If specified as `auto`, it will be
            decided by the norm_cfg. Bias of conv will be set as True
            if `norm_cfg` is None, otherwise False. Default: 'auto'.
        background_label (bool, Optional): Label ID of background,
            set as 0 for RPN and num_classes for other heads.
            It will automatically set as `num_classes` if None is given.
        use_direction_classifier (bool):
            Whether to add a direction classifier.
        diff_rad_by_sin (bool): Whether to change the difference
            into sin difference for box regression loss. Defaults to True.
        dir_offset (float): Parameter used in direction
            classification. Defaults to 0.
        dir_limit_offset (float): Parameter used in direction
            classification. Defaults to 0.
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        loss_dir (:obj:`ConfigDict` or dict): Config of direction classifier
            loss.
        loss_attr (:obj:`ConfigDict` or dict): Config of attribute classifier
            loss, which is only active when `pred_attrs=True`.
        bbox_code_size (int): Dimensions of predicted bounding boxes.
        pred_attrs (bool): Whether to predict attributes.
            Defaults to False.
        num_attrs (int): The number of attributes to be predicted.
            Default: 9.
        pred_velo (bool): Whether to predict velocity.
            Defaults to False.
        pred_bbox2d (bool): Whether to predict 2D boxes.
            Defaults to False.
        group_reg_dims (tuple[int], optional): The dimension of each regression
            target group. Default: (2, 1, 3, 1, 2).
        cls_branch (tuple[int], optional): Channels for classification branch.
            Default: (128, 64).
        reg_branch (tuple[tuple], optional): Channels for regression branch.
            Default: (
                (128, 64),  # offset
                (128, 64),  # depth
                (64, ),  # size
                (64, ),  # rot
                ()  # velo
            ),
        dir_branch (Sequence[int]): Channels for direction
            classification branch. Default: (64, ).
        attr_branch (Sequence[int]): Channels for classification branch.
            Default: (64, ).
        conv_cfg (:obj:`ConfigDict` or dict, Optional): Config dict for
            convolution layer. Default: None.
        norm_cfg (:obj:`ConfigDict` or dict, Optional): Config dict for
            normalization layer. Default: None.
        train_cfg (:obj:`ConfigDict` or dict, Optional): Training config
            of anchor head.
        test_cfg (:obj:`ConfigDict` or dict, Optional): Testing config of
            anchor head.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.
    """  # noqa: W605

    _version = 1

    def __init__(
            self,
            num_classes: int,
            in_channels: int,
            feat_channels: int = 256,
            stacked_convs: int = 4,
            strides: Sequence[int] = (4, 8, 16, 32, 64),
            dcn_on_last_conv: bool = False,
            conv_bias: Union[bool, str] = 'auto',
            background_label: bool = None,
            use_direction_classifier: bool = True,
            diff_rad_by_sin: bool = True,
            dir_offset: int = 0,
            dir_limit_offset: int = 0,
            loss_cls: ConfigType = dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox: ConfigType = dict(
                type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
            loss_dir: ConfigType = dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_attr: ConfigType = dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            bbox_code_size: int = 9,  # For nuscenes
            pred_attrs: bool = False,
            num_attrs: int = 9,  # For nuscenes
            pred_velo: bool = False,
            pred_bbox2d: bool = False,
            group_reg_dims: Sequence[int] = (
                2, 1, 3, 1, 2),  # offset, depth, size, rot, velo,
            cls_branch: Sequence[int] = (128, 64),
            reg_branch: Sequence[Tuple[int, int]] = (
                (128, 64),  # offset
                (128, 64),  # depth
                (64, ),  # size
                (64, ),  # rot
                ()  # velo
            ),
            dir_branch: Sequence[int] = (64, ),
            attr_branch: Sequence[int] = (64, ),
            conv_cfg: OptConfigType = None,
            norm_cfg: OptConfigType = None,
            train_cfg: OptConfigType = None,
            test_cfg: OptConfigType = None,
            init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.use_direction_classifier = use_direction_classifier
        self.diff_rad_by_sin = diff_rad_by_sin
        self.dir_offset = dir_offset
        self.dir_limit_offset = dir_limit_offset
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_dir = MODELS.build(loss_dir)
        self.bbox_code_size = bbox_code_size
        self.group_reg_dims = list(group_reg_dims)
        self.cls_branch = cls_branch
        self.reg_branch = reg_branch
        assert len(reg_branch) == len(group_reg_dims), 'The number of '\
            'element in reg_branch and group_reg_dims should be the same.'
        self.pred_velo = pred_velo
        self.pred_bbox2d = pred_bbox2d
        self.out_channels = []
        for reg_branch_channels in reg_branch:
            if len(reg_branch_channels) > 0:
                self.out_channels.append(reg_branch_channels[-1])
            else:
                self.out_channels.append(-1)
        self.dir_branch = dir_branch
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.background_label = (
            num_classes if background_label is None else background_label)
        # background_label should be either 0 or num_classes
        assert (self.background_label == 0
                or self.background_label == num_classes)
        self.pred_attrs = pred_attrs
        self.attr_background_label = -1
        self.num_attrs = num_attrs
        if self.pred_attrs:
            self.attr_background_label = num_attrs
            self.loss_attr = MODELS.build(loss_attr)
            self.attr_branch = attr_branch

        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self._init_cls_convs()
        self._init_reg_convs()
        self._init_predictor()

    def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def _init_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def _init_branch(self, conv_channels=(64), conv_strides=(1)):
        """Initialize conv layers as a prediction branch."""
        conv_before_pred = nn.ModuleList()
        if isinstance(conv_channels, int):
            conv_channels = [self.feat_channels] + [conv_channels]
            conv_strides = [conv_strides]
        else:
            conv_channels = [self.feat_channels] + list(conv_channels)
            conv_strides = list(conv_strides)
        for i in range(len(conv_strides)):
            conv_before_pred.append(
                ConvModule(
                    conv_channels[i],
                    conv_channels[i + 1],
                    3,
                    stride=conv_strides[i],
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

        return conv_before_pred

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        self.conv_cls_prev = self._init_branch(
            conv_channels=self.cls_branch,
            conv_strides=(1, ) * len(self.cls_branch))
        self.conv_cls = nn.Conv2d(self.cls_branch[-1], self.cls_out_channels,
                                  1)
        self.conv_reg_prevs = nn.ModuleList()
        self.conv_regs = nn.ModuleList()
        for i in range(len(self.group_reg_dims)):
            reg_dim = self.group_reg_dims[i]
            reg_branch_channels = self.reg_branch[i]
            out_channel = self.out_channels[i]
            if len(reg_branch_channels) > 0:
                self.conv_reg_prevs.append(
                    self._init_branch(
                        conv_channels=reg_branch_channels,
                        conv_strides=(1, ) * len(reg_branch_channels)))
                self.conv_regs.append(nn.Conv2d(out_channel, reg_dim, 1))
            else:
                self.conv_reg_prevs.append(None)
                self.conv_regs.append(
                    nn.Conv2d(self.feat_channels, reg_dim, 1))
        if self.use_direction_classifier:
            self.conv_dir_cls_prev = self._init_branch(
                conv_channels=self.dir_branch,
                conv_strides=(1, ) * len(self.dir_branch))
            self.conv_dir_cls = nn.Conv2d(self.dir_branch[-1], 2, 1)
        if self.pred_attrs:
            self.conv_attr_prev = self._init_branch(
                conv_channels=self.attr_branch,
                conv_strides=(1, ) * len(self.attr_branch))
            self.conv_attr = nn.Conv2d(self.attr_branch[-1], self.num_attrs, 1)

    def init_weights(self):
        """Initialize weights of the head.

        We currently still use the customized defined init_weights because the
        default init of DCN triggered by the init_cfg will init
        conv_offset.weight, which mistakenly affects the training stability.
        """
        for modules in [self.cls_convs, self.reg_convs, self.conv_cls_prev]:
            for m in modules:
                if isinstance(m.conv, nn.Conv2d):
                    normal_init(m.conv, std=0.01)
        for conv_reg_prev in self.conv_reg_prevs:
            if conv_reg_prev is None:
                continue
            for m in conv_reg_prev:
                if isinstance(m.conv, nn.Conv2d):
                    normal_init(m.conv, std=0.01)
        if self.use_direction_classifier:
            for m in self.conv_dir_cls_prev:
                if isinstance(m.conv, nn.Conv2d):
                    normal_init(m.conv, std=0.01)
        if self.pred_attrs:
            for m in self.conv_attr_prev:
                if isinstance(m.conv, nn.Conv2d):
                    normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        for conv_reg in self.conv_regs:
            normal_init(conv_reg, std=0.01)
        if self.use_direction_classifier:
            normal_init(self.conv_dir_cls, std=0.01, bias=bias_cls)
        if self.pred_attrs:
            normal_init(self.conv_attr, std=0.01, bias=bias_cls)

    def forward(
        self, x: Tuple[Tensor]
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually contain classification scores, bbox predictions,
                and direction class predictions.
                cls_scores (list[Tensor]): Box scores for each scale level,
                    each is a 4D-tensor, the channel number is
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * bbox_code_size.
                dir_cls_preds (list[Tensor]): Box scores for direction class
                    predictions on each scale level, each is a 4D-tensor,
                    the channel number is num_points * 2. (bin = 2)
                attr_preds (list[Tensor]): Attribute scores for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * num_attrs.
        """
        return multi_apply(self.forward_single, x)[:5]

    def forward_single(self, x: Tensor) -> Tuple[Tensor, ...]:
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.

        Returns:
            tuple: Scores for each class, bbox predictions, direction class,
                and attributes, features after classification and regression
                conv layers, some models needs these features like FCOS.
        """
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        # clone the cls_feat for reusing the feature map afterwards
        clone_cls_feat = cls_feat.clone()
        for conv_cls_prev_layer in self.conv_cls_prev:
            clone_cls_feat = conv_cls_prev_layer(clone_cls_feat)
        cls_score = self.conv_cls(clone_cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = []
        for i in range(len(self.group_reg_dims)):
            # clone the reg_feat for reusing the feature map afterwards
            clone_reg_feat = reg_feat.clone()
            if len(self.reg_branch[i]) > 0:
                for conv_reg_prev_layer in self.conv_reg_prevs[i]:
                    clone_reg_feat = conv_reg_prev_layer(clone_reg_feat)
            bbox_pred.append(self.conv_regs[i](clone_reg_feat))
        bbox_pred = torch.cat(bbox_pred, dim=1)

        dir_cls_pred = None
        if self.use_direction_classifier:
            clone_reg_feat = reg_feat.clone()
            for conv_dir_cls_prev_layer in self.conv_dir_cls_prev:
                clone_reg_feat = conv_dir_cls_prev_layer(clone_reg_feat)
            dir_cls_pred = self.conv_dir_cls(clone_reg_feat)

        attr_pred = None
        if self.pred_attrs:
            # clone the cls_feat for reusing the feature map afterwards
            clone_cls_feat = cls_feat.clone()
            for conv_attr_prev_layer in self.conv_attr_prev:
                clone_cls_feat = conv_attr_prev_layer(clone_cls_feat)
            attr_pred = self.conv_attr(clone_cls_feat)

        return cls_score, bbox_pred, dir_cls_pred, attr_pred, cls_feat, \
            reg_feat

    @abstractmethod
    def get_targets(self, points: List[Tensor],
                    batch_gt_instances: InstanceList) -> Any:
        """Compute regression, classification and centerss targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instance_3d.  It usually includes ``bboxes``、``labels``
                、``bboxes_3d``、``labels_3d``、``depths``、``centers_2d``
                and attributes.
        """
        raise NotImplementedError

    # TODO: Refactor using MlvlPointGenerator in MMDet.
    def _get_points_single(self,
                           featmap_size: Tuple[int],
                           stride: int,
                           dtype: torch.dtype,
                           device: torch.device,
                           flatten: bool = False) -> Tuple[Tensor, Tensor]:
        """Get points of a single scale level.

        Args:
            featmap_size (tuple[int]): Single scale level feature map
                size.
            stride (int): Downsample factor of the feature map.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.
            flatten (bool): Whether to flatten the tensor.
                Defaults to False.

        Returns:
            tuple: points of each image.
        """
        h, w = featmap_size
        x_range = torch.arange(w, dtype=dtype, device=device)
        y_range = torch.arange(h, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        if flatten:
            y = y.flatten()
            x = x.flatten()
        return y, x

    # TODO: Refactor using MlvlPointGenerator in MMDet.
    def get_points(self,
                   featmap_sizes: List[Tuple[int]],
                   dtype: torch.dtype,
                   device: torch.device,
                   flatten: bool = False) -> List[Tuple[Tensor, Tensor]]:
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.
            flatten (bool): Whether to flatten the tensor.
                Defaults to False.

        Returns:
            list[tuple]: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self._get_points_single(featmap_sizes[i], self.strides[i],
                                        dtype, device, flatten))
        return mlvl_points
