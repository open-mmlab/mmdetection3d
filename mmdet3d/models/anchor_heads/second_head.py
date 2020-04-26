import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import bias_init_with_prob, normal_init

from mmdet3d.core import (PseudoSampler, box_torch_ops,
                          boxes3d_to_bev_torch_lidar, build_anchor_generator,
                          build_assigner, build_bbox_coder, build_sampler,
                          multi_apply)
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu, nms_normal_gpu
from mmdet.models import HEADS
from ..builder import build_loss
from .train_mixins import AnchorTrainMixin


@HEADS.register_module
class SECONDHead(nn.Module, AnchorTrainMixin):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).
    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels of the feature map.
        anchor_scales (Iterable): Anchor scales.
        anchor_ratios (Iterable): Anchor aspect ratios.
        anchor_strides (Iterable): Anchor strides.
        anchor_base_sizes (Iterable): Anchor base sizes.
        target_means (Iterable): Mean values of regression targets.
        target_stds (Iterable): Std values of regression targets.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
    """  # noqa: W605

    def __init__(self,
                 class_name,
                 in_channels,
                 train_cfg,
                 test_cfg,
                 feat_channels=256,
                 use_direction_classifier=True,
                 encode_bg_as_zeros=False,
                 box_code_size=7,
                 anchor_generator=dict(
                     type='Anchor3DRangeGenerator',
                     range=[0, -39.68, -1.78, 69.12, 39.68, -1.78],
                     strides=[2],
                     sizes=[[1.6, 3.9, 1.56]],
                     rotations=[0, 1.57],
                     custom_values=[],
                     reshape_out=False),
                 assigner_per_size=False,
                 assign_per_class=False,
                 diff_rad_by_sin=True,
                 dir_offset=0,
                 dir_limit_offset=1,
                 bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
                 loss_dir=dict(type='CrossEntropyLoss', loss_weight=0.2)):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = len(class_name)
        self.feat_channels = feat_channels
        self.diff_rad_by_sin = diff_rad_by_sin
        self.use_direction_classifier = use_direction_classifier
        # self.encode_background_as_zeros = encode_bg_as_zeros
        self.box_code_size = box_code_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.assigner_per_size = assigner_per_size
        self.assign_per_class = assign_per_class
        self.dir_offset = dir_offset
        self.dir_limit_offset = dir_limit_offset

        # build target assigner & sampler
        if train_cfg is not None:
            self.sampling = loss_cls['type'] not in ['FocalLoss', 'GHMC']
            if self.sampling:
                self.bbox_sampler = build_sampler(train_cfg.sampler)
            else:
                self.bbox_sampler = PseudoSampler()
            if isinstance(train_cfg.assigner, dict):
                self.bbox_assigner = build_assigner(train_cfg.assigner)
            elif isinstance(train_cfg.assigner, list):
                self.bbox_assigner = [
                    build_assigner(res) for res in train_cfg.assigner
                ]

        # build anchor generator
        self.anchor_generator = build_anchor_generator(anchor_generator)
        # In 3D detection, the anchor stride is connected with anchor size
        self.num_anchors = self.anchor_generator.num_base_anchors

        self._init_layers()
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_dir = build_loss(loss_dir)
        self.fp16_enabled = False

    def _init_layers(self):
        self.cls_out_channels = self.num_anchors * self.num_classes
        self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.feat_channels,
                                  self.num_anchors * self.box_code_size, 1)
        if self.use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(self.feat_channels,
                                          self.num_anchors * 2, 1)

    def init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        normal_init(self.conv_reg, std=0.01)

    def forward_single(self, x):
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        dir_cls_preds = None
        if self.use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
        return cls_score, bbox_pred, dir_cls_preds

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, input_metas, device='cuda'):
        """Get anchors according to feature map sizes.
        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            input_metas (list[dict]): contain pcd and img's meta info.
        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        num_imgs = len(input_metas)
        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]
        return anchor_list

    def loss_single(self, cls_score, bbox_pred, dir_cls_preds, labels,
                    label_weights, bbox_targets, bbox_weights, dir_targets,
                    dir_weights, num_total_samples):
        # classification loss
        if num_total_samples is None:
            num_total_samples = int(cls_score.shape[0])
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)

        # regression loss
        bbox_targets = bbox_targets.reshape(-1, self.box_code_size)
        bbox_weights = bbox_weights.reshape(-1, self.box_code_size)
        code_weight = self.train_cfg.get('code_weight', None)

        if code_weight:
            bbox_weights = bbox_weights * bbox_weights.new_tensor(code_weight)
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1, self.box_code_size)
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
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).reshape(-1, 2)
            dir_targets = dir_targets.reshape(-1)
            dir_weights = dir_weights.reshape(-1)
            loss_dir = self.loss_dir(
                dir_cls_preds,
                dir_targets,
                dir_weights,
                avg_factor=num_total_samples)

        return loss_cls, loss_bbox, loss_dir

    @staticmethod
    def add_sin_difference(boxes1, boxes2):
        rad_pred_encoding = torch.sin(boxes1[..., -1:]) * torch.cos(
            boxes2[..., -1:])
        rad_tg_encoding = torch.cos(boxes1[..., -1:]) * torch.sin(boxes2[...,
                                                                         -1:])
        boxes1 = torch.cat([boxes1[..., :-1], rad_pred_encoding], dim=-1)
        boxes2 = torch.cat([boxes2[..., :-1], rad_tg_encoding], dim=-1)
        return boxes1, boxes2

    def loss(self,
             cls_scores,
             bbox_preds,
             dir_cls_preds,
             gt_bboxes,
             gt_labels,
             input_metas,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        device = cls_scores[0].device
        anchor_list = self.get_anchors(
            featmap_sizes, input_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.anchor_target_3d(
            anchor_list,
            gt_bboxes,
            input_metas,
            self.target_means,
            self.target_stds,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            num_classes=self.num_classes,
            label_channels=label_channels,
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
            loss_cls_3d=losses_cls,
            loss_bbox_3d=losses_bbox,
            loss_dir_3d=losses_dir)

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   dir_cls_preds,
                   input_metas,
                   rescale=False):
        assert len(cls_scores) == len(bbox_preds)
        assert len(cls_scores) == len(dir_cls_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        device = cls_scores[0].device
        mlvl_anchors = self.anchor_generators.grid_anchors(
            featmap_sizes, device=device)
        mlvl_anchors = [
            anchor.reshape(-1, self.box_code_size) for anchor in mlvl_anchors
        ]

        result_list = []
        for img_id in range(len(input_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            dir_cls_pred_list = [
                dir_cls_preds[i][img_id].detach() for i in range(num_levels)
            ]

            input_meta = input_metas[img_id]
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               dir_cls_pred_list, mlvl_anchors,
                                               input_meta, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          dir_cls_preds,
                          mlvl_anchors,
                          input_meta,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []
        mlvl_bboxes_for_nms = []
        for cls_score, bbox_pred, dir_cls_pred, anchors in zip(
                cls_scores, bbox_preds, dir_cls_preds, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            if self.use_direction_classifier:
                assert cls_score.size()[-2:] == dir_cls_pred.size()[-2:]

            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.num_classes)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2,
                                          0).reshape(-1, self.box_code_size)
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]

            score_thr = self.test_cfg.get('score_thr', 0)
            if score_thr > 0:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                thr_inds = (max_scores >= score_thr)
                anchors = anchors[thr_inds]
                bbox_pred = bbox_pred[thr_inds]
                scores = scores[thr_inds]
                dir_cls_scores = dir_cls_score[thr_inds]
            bboxes = self.bbox_coder.decode(anchors, bbox_pred)
            bboxes_for_nms = boxes3d_to_bev_torch_lidar(bboxes)
            mlvl_bboxes_for_nms.append(bboxes_for_nms)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_scores)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_bboxes_for_nms = torch.cat(mlvl_bboxes_for_nms)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)

        if len(mlvl_scores) > 0:
            mlvl_scores, mlvl_label_preds = mlvl_scores.max(dim=-1)
            if self.test_cfg.use_rotate_nms:
                nms_func = nms_gpu
            else:
                nms_func = nms_normal_gpu
            selected = nms_func(mlvl_bboxes_for_nms, mlvl_scores,
                                self.test_cfg.nms_thr)
        else:
            selected = []

        if len(selected) > 0:
            selected_bboxes = mlvl_bboxes[selected]
            selected_scores = mlvl_scores[selected]
            selected_label_preds = mlvl_label_preds[selected]
            selected_dir_scores = mlvl_dir_scores[selected]
            # TODO: move dir_offset to box coder
            dir_rot = box_torch_ops.limit_period(
                selected_bboxes[..., -1] - self.dir_offset,
                self.dir_limit_offset, np.pi)
            selected_bboxes[..., -1] = (
                dir_rot + self.dir_offset +
                np.pi * selected_dir_scores.to(selected_bboxes.dtype))

            return dict(
                box3d_lidar=selected_bboxes.cpu(),
                scores=selected_scores.cpu(),
                label_preds=selected_label_preds.cpu(),
                sample_idx=input_meta['sample_idx'],
            )

        return dict(
            box3d_lidar=mlvl_scores.new_zeros([0, 7]).cpu(),
            scores=mlvl_scores.new_zeros([0]).cpu(),
            label_preds=mlvl_scores.new_zeros([0, 4]).cpu(),
            sample_idx=input_meta['sample_idx'],
        )
