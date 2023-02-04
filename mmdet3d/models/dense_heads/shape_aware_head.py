# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from mmcv.cnn import ConvModule
from mmdet.models.utils import multi_apply
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor
from torch import nn as nn

from mmdet3d.models.layers import box3d_multiclass_nms
from mmdet3d.registry import MODELS
from mmdet3d.structures import limit_period, xywhr2xyxyr
from mmdet3d.utils import InstanceList, OptInstanceList
from .anchor3d_head import Anchor3DHead


@MODELS.register_module()
class BaseShapeHead(BaseModule):
    """Base Shape-aware Head in Shape Signature Network.

    Note:
        This base shape-aware grouping head uses default settings for small
        objects. For large and huge objects, it is recommended to use
        heavier heads, like (64, 64, 64) and (128, 128, 64, 64, 64) in
        shared conv channels, (2, 1, 1) and (2, 1, 2, 1, 1) in shared
        conv strides. For tiny objects, we can use smaller heads, like
        (32, 32) channels and (1, 1) strides.

    Args:
        num_cls (int): Number of classes.
        num_base_anchors (int): Number of anchors per location.
        box_code_size (int): The dimension of boxes to be encoded.
        in_channels (int): Input channels for convolutional layers.
        shared_conv_channels (tuple, optional): Channels for shared
            convolutional layers. Default: (64, 64).
        shared_conv_strides (tuple): Strides for shared
            convolutional layers. Default: (1, 1).
        use_direction_classifier (bool): Whether to use direction
            classifier. Default: True.
        conv_cfg (dict): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict): Config of norm layer.
            Default: dict(type='BN2d').
        bias (bool | str): Type of bias. Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_cls: int,
                 num_base_anchors: int,
                 box_code_size: int,
                 in_channels: int,
                 shared_conv_channels: Tuple = (64, 64),
                 shared_conv_strides: Tuple = (1, 1),
                 use_direction_classifier: bool = True,
                 conv_cfg: Dict = dict(type='Conv2d'),
                 norm_cfg: Dict = dict(type='BN2d'),
                 bias: bool = False,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_cls = num_cls
        self.num_base_anchors = num_base_anchors
        self.use_direction_classifier = use_direction_classifier
        self.box_code_size = box_code_size

        assert len(shared_conv_channels) == len(shared_conv_strides), \
            'Lengths of channels and strides list should be equal.'

        self.shared_conv_channels = [in_channels] + list(shared_conv_channels)
        self.shared_conv_strides = list(shared_conv_strides)

        shared_conv = []
        for i in range(len(self.shared_conv_strides)):
            shared_conv.append(
                ConvModule(
                    self.shared_conv_channels[i],
                    self.shared_conv_channels[i + 1],
                    kernel_size=3,
                    stride=self.shared_conv_strides[i],
                    padding=1,
                    conv_cfg=conv_cfg,
                    bias=bias,
                    norm_cfg=norm_cfg))

        self.shared_conv = nn.Sequential(*shared_conv)

        out_channels = self.shared_conv_channels[-1]
        self.conv_cls = nn.Conv2d(out_channels, num_base_anchors * num_cls, 1)
        self.conv_reg = nn.Conv2d(out_channels,
                                  num_base_anchors * box_code_size, 1)

        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(out_channels, num_base_anchors * 2,
                                          1)
        if init_cfg is None:
            if use_direction_classifier:
                self.init_cfg = dict(
                    type='Kaiming',
                    layer='Conv2d',
                    override=[
                        dict(type='Normal', name='conv_reg', std=0.01),
                        dict(
                            type='Normal',
                            name='conv_cls',
                            std=0.01,
                            bias_prob=0.01),
                        dict(
                            type='Normal',
                            name='conv_dir_cls',
                            std=0.01,
                            bias_prob=0.01)
                    ])
            else:
                self.init_cfg = dict(
                    type='Kaiming',
                    layer='Conv2d',
                    override=[
                        dict(type='Normal', name='conv_reg', std=0.01),
                        dict(
                            type='Normal',
                            name='conv_cls',
                            std=0.01,
                            bias_prob=0.01)
                    ])

    def forward(self, x: Tensor) -> Dict:
        """Forward function for SmallHead.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, C, H, W].

        Returns:
            dict[torch.Tensor]: Contain score of each class, bbox
                regression and direction classification predictions.
                Note that all the returned tensors are reshaped as
                [bs*num_base_anchors*H*W, num_cls/box_code_size/dir_bins].
                It is more convenient to concat anchors for different
                classes even though they have different feature map sizes.
        """
        x = self.shared_conv(x)
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        featmap_size = bbox_pred.shape[-2:]
        H, W = featmap_size
        B = bbox_pred.shape[0]
        cls_score = cls_score.view(-1, self.num_base_anchors, self.num_cls, H,
                                   W).permute(0, 1, 3, 4,
                                              2).reshape(B, -1, self.num_cls)
        bbox_pred = bbox_pred.view(-1, self.num_base_anchors,
                                   self.box_code_size, H, W).permute(
                                       0, 1, 3, 4,
                                       2).reshape(B, -1, self.box_code_size)

        dir_cls_preds = None
        if self.use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.view(-1, self.num_base_anchors, 2, H,
                                               W).permute(0, 1, 3, 4,
                                                          2).reshape(B, -1, 2)
        ret = dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            dir_cls_preds=dir_cls_preds,
            featmap_size=featmap_size)
        return ret


@MODELS.register_module()
class ShapeAwareHead(Anchor3DHead):
    """Shape-aware grouping head for SSN.

    Args:
        tasks (dict): Shape-aware groups of multi-class objects.
        assign_per_class (bool): Whether to do assignment for each
            class. Default: True.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 tasks: Dict,
                 assign_per_class: bool = True,
                 init_cfg: Optional[dict] = None,
                 **kwargs) -> Dict:
        self.tasks = tasks
        self.featmap_sizes = []
        super().__init__(
            assign_per_class=assign_per_class, init_cfg=init_cfg, **kwargs)

    def init_weights(self):
        if not self._is_init:
            for m in self.heads:
                if hasattr(m, 'init_weights'):
                    m.init_weights()
            self._is_init = True
        else:
            warnings.warn(f'init_weights of {self.__class__.__name__} has '
                          f'been called more than once.')

    def _init_layers(self):
        """Initialize neural network layers of the head."""
        self.heads = nn.ModuleList()
        cls_ptr = 0
        for task in self.tasks:
            sizes = self.prior_generator.sizes[cls_ptr:cls_ptr +
                                               task['num_class']]
            num_size = torch.tensor(sizes).reshape(-1, 3).size(0)
            num_rot = len(self.prior_generator.rotations)
            num_base_anchors = num_rot * num_size
            branch = dict(
                type='BaseShapeHead',
                num_cls=self.num_classes,
                num_base_anchors=num_base_anchors,
                box_code_size=self.box_code_size,
                in_channels=self.in_channels,
                shared_conv_channels=task['shared_conv_channels'],
                shared_conv_strides=task['shared_conv_strides'])
            self.heads.append(MODELS.build(branch))
            cls_ptr += task['num_class']

    def forward_single(self, x: Tensor) -> Tuple[Tensor]:
        """Forward function on a single-scale feature map.

        Args:
            x (torch.Tensor): Input features.
        Returns:
            tuple[torch.Tensor]: Contain score of each class, bbox
                regression and direction classification predictions.
        """
        results = []

        for head in self.heads:
            results.append(head(x))

        cls_score = torch.cat([result['cls_score'] for result in results],
                              dim=1)
        bbox_pred = torch.cat([result['bbox_pred'] for result in results],
                              dim=1)
        dir_cls_preds = None
        if self.use_direction_classifier:
            dir_cls_preds = torch.cat(
                [result['dir_cls_preds'] for result in results], dim=1)

        self.featmap_sizes = []
        for i, task in enumerate(self.tasks):
            for _ in range(task['num_class']):
                self.featmap_sizes.append(results[i]['featmap_size'])
        assert len(self.featmap_sizes) == len(self.prior_generator.ranges), \
            'Length of feature map sizes must be equal to length of ' + \
            'different ranges of anchor generator.'

        return cls_score, bbox_pred, dir_cls_preds

    def loss_single(self, cls_score: Tensor, bbox_pred: Tensor,
                    dir_cls_preds: Tensor, labels: Tensor,
                    label_weights: Tensor, bbox_targets: Tensor,
                    bbox_weights: Tensor, dir_targets: Tensor,
                    dir_weights: Tensor,
                    num_total_samples: int) -> Tuple[Tensor]:
        """Calculate loss of Single-level results.

        Args:
            cls_score (torch.Tensor): Class score in single-level.
            bbox_pred (torch.Tensor): Bbox prediction in single-level.
            dir_cls_preds (torch.Tensor): Predictions of direction class
                in single-level.
            labels (torch.Tensor): Labels of class.
            label_weights (torch.Tensor): Weights of class loss.
            bbox_targets (torch.Tensor): Targets of bbox predictions.
            bbox_weights (torch.Tensor): Weights of bbox loss.
            dir_targets (torch.Tensor): Targets of direction predictions.
            dir_weights (torch.Tensor): Weights of direction loss.
            num_total_samples (int): The number of valid samples.

        Returns:
            tuple[torch.Tensor]: Losses of class, bbox
                and direction, respectively.
        """
        # classification loss
        if num_total_samples is None:
            num_total_samples = int(cls_score.shape[0])
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.reshape(-1, self.num_classes)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)

        # regression loss
        bbox_targets = bbox_targets.reshape(-1, self.box_code_size)
        bbox_weights = bbox_weights.reshape(-1, self.box_code_size)
        code_weight = self.train_cfg.get('code_weight', None)

        if code_weight:
            bbox_weights = bbox_weights * bbox_weights.new_tensor(code_weight)
        bbox_pred = bbox_pred.reshape(-1, self.box_code_size)
        if self.diff_rad_by_sin:
            bbox_pred, bbox_targets = self.add_sin_difference(
                bbox_pred, bbox_targets)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)

        # direction classification loss
        loss_dir = None
        if self.use_direction_classifier:
            dir_cls_preds = dir_cls_preds.reshape(-1, 2)
            dir_targets = dir_targets.reshape(-1)
            dir_weights = dir_weights.reshape(-1)
            loss_dir = self.loss_dir(
                dir_cls_preds,
                dir_targets,
                dir_weights,
                avg_factor=num_total_samples)

        return loss_cls, loss_bbox, loss_dir

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            dir_cls_preds: List[Tensor],
            batch_gt_instances_3d: InstanceList,
            batch_input_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> Dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[torch.Tensor]): Multi-level class scores.
            bbox_preds (list[torch.Tensor]): Multi-level bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Multi-level direction
                class predictions.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances. It usually includes ``bboxes_3d`` and
                ``labels_3d`` attributes.
            batch_input_metas (list[dict]): Contain pcd and sample's meta info.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, list[torch.Tensor]]: Classification, bbox, and
                direction losses of each level.

                - loss_cls (list[torch.Tensor]): Classification losses.
                - loss_bbox (list[torch.Tensor]): Box regression losses.
                - loss_dir (list[torch.Tensor]): Direction classification
                    losses.
        """
        device = cls_scores[0].device
        anchor_list = self.get_anchors(
            self.featmap_sizes, batch_input_metas, device=device)
        cls_reg_targets = self.anchor_target_3d(
            anchor_list,
            batch_gt_instances_3d,
            batch_input_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
            num_classes=self.num_classes,
            sampling=self.sampling)

        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         dir_targets_list, dir_weights_list, num_total_pos,
         num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # num_total_samples = None
        losses_cls, losses_bbox, losses_dir = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            dir_cls_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            dir_targets_list,
            dir_weights_list,
            num_total_samples=num_total_samples)
        return dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dir=losses_dir)

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        dir_cls_preds: List[Tensor],
                        batch_input_metas: List[dict],
                        cfg: Optional[dict] = None,
                        rescale: List[Tensor] = False) -> List[tuple]:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Args:
            cls_scores (list[torch.Tensor]): Multi-level class scores.
            bbox_preds (list[torch.Tensor]): Multi-level bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Multi-level direction
                class predictions.
            batch_input_metas (list[dict]): Contain pcd and img's meta info.
            cfg (:obj:`ConfigDict`, optional): Training or testing config.
                Default: None.
            rescale (list[torch.Tensor], optional): Whether to rescale bbox.
                Default: False.

        Returns:
            list[tuple]: Prediction resultes of batches.
        """
        assert len(cls_scores) == len(bbox_preds)
        assert len(cls_scores) == len(dir_cls_preds)
        num_levels = len(cls_scores)
        assert num_levels == 1, 'Only support single level inference.'
        device = cls_scores[0].device
        mlvl_anchors = self.prior_generator.grid_anchors(
            self.featmap_sizes, device=device)
        # `anchor` is a list of anchors for different classes
        mlvl_anchors = [torch.cat(anchor, dim=0) for anchor in mlvl_anchors]

        result_list = []
        for img_id in range(len(batch_input_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            dir_cls_pred_list = [
                dir_cls_preds[i][img_id].detach() for i in range(num_levels)
            ]

            input_meta = batch_input_metas[img_id]
            proposals = self._predict_by_feat_single(cls_score_list,
                                                     bbox_pred_list,
                                                     dir_cls_pred_list,
                                                     mlvl_anchors, input_meta,
                                                     cfg, rescale)
            result_list.append(proposals)
        return result_list

    def _predict_by_feat_single(self,
                                cls_scores: Tensor,
                                bbox_preds: Tensor,
                                dir_cls_preds: Tensor,
                                mlvl_anchors: List[Tensor],
                                input_meta: List[dict],
                                cfg: Dict = None,
                                rescale: List[Tensor] = False):
        """Transform a single point's features extracted from the head into
        bbox results.

        Args:
            cls_scores (torch.Tensor): Class score in single batch.
            bbox_preds (torch.Tensor): Bbox prediction in single batch.
            dir_cls_preds (torch.Tensor): Predictions of direction class
                in single batch.
            mlvl_anchors (List[torch.Tensor]): Multi-level anchors
                in single batch.
            input_meta (list[dict]): Contain pcd and img's meta info.
            cfg (:obj:`ConfigDict`): Training or testing config.
            rescale (list[torch.Tensor]): whether to rescale bbox.
                Default: False.

        Returns:
            tuple: Contain predictions of single batch.

                - bboxes (:obj:`BaseInstance3DBoxes`): Predicted 3d bboxes.
                - scores (torch.Tensor): Class score of each bbox.
                - labels (torch.Tensor): Label of each bbox.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []
        for cls_score, bbox_pred, dir_cls_pred, anchors in zip(
                cls_scores, bbox_preds, dir_cls_preds, mlvl_anchors):
            assert cls_score.size()[-2] == bbox_pred.size()[-2]
            assert cls_score.size()[-2] == dir_cls_pred.size()[-2]
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]

            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                dir_cls_score = dir_cls_score[topk_inds]

            bboxes = self.bbox_coder.decode(anchors, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_bboxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d'](
            mlvl_bboxes, box_dim=self.box_code_size).bev)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)

        if self.use_sigmoid_cls:
            # Add a dummy background class to the front when using sigmoid
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        score_thr = cfg.get('score_thr', 0)
        results = box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
                                       mlvl_scores, score_thr, cfg.max_num,
                                       cfg, mlvl_dir_scores)
        bboxes, scores, labels, dir_scores = results
        if bboxes.shape[0] > 0:
            dir_rot = limit_period(bboxes[..., 6] - self.dir_offset,
                                   self.dir_limit_offset, np.pi)
            bboxes[..., 6] = (
                dir_rot + self.dir_offset +
                np.pi * dir_scores.to(bboxes.dtype))
        bboxes = input_meta['box_type_3d'](bboxes, box_dim=self.box_code_size)
        results = InstanceData()
        results.bboxes_3d = bboxes
        results.scores_3d = scores
        results.labels_3d = labels
        return results
