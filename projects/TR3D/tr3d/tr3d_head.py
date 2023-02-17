# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from https://github.com/SamsungLabs/tr3d/blob/master/mmdet3d/models/dense_heads/tr3d_head.py # noqa
from typing import List, Optional, Tuple

try:
    import MinkowskiEngine as ME
    from MinkowskiEngine import SparseTensor
except ImportError:
    # Please follow getting_started.md to install MinkowskiEngine.
    ME = SparseTensor = None
    pass

import torch
from mmcv.ops import nms3d, nms3d_normal
from mmengine.model import bias_init_with_prob
from mmengine.structures import InstanceData
from torch import Tensor, nn

from mmdet3d.models import Base3DDenseHead
from mmdet3d.registry import MODELS
from mmdet3d.structures import BaseInstance3DBoxes
from mmdet3d.utils import InstanceList, OptInstanceList


@MODELS.register_module()
class TR3DHead(Base3DDenseHead):
    r"""Bbox head of `TR3D <https://arxiv.org/abs/2302.02858>`_.

    Args:
        in_channels (int): Number of channels in input tensors.
        num_reg_outs (int): Number of regression layer channels.
        voxel_size (float): Voxel size in meters.
        pts_center_threshold (int): Box to location assigner parameter.
            After feature level for the box is determined, assigner selects
            pts_center_threshold locations closest to the box center.
        bbox_loss (dict): Config of bbox loss. Defaults to
            dict(type='AxisAlignedIoULoss', mode='diou', reduction=None).
        cls_loss (dict): Config of classification loss. Defaults to
            dict = dict(type='mmdet.FocalLoss', reduction=None).
        train_cfg (dict, optional): Config for train stage. Defaults to None.
        test_cfg (dict, optional): Config for test stage. Defaults to None.
        init_cfg (dict, optional): Config for weight initialization.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 num_reg_outs: int,
                 voxel_size: int,
                 pts_center_threshold: int,
                 label2level: Tuple[int],
                 bbox_loss: dict = dict(
                     type='TR3DAxisAlignedIoULoss',
                     mode='diou',
                     reduction='none'),
                 cls_loss: dict = dict(
                     type='mmdet.FocalLoss', reduction='none'),
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super(TR3DHead, self).__init__(init_cfg)
        if ME is None:
            raise ImportError(
                'Please follow `getting_started.md` to install MinkowskiEngine.`'  # noqa: E501
            )
        self.voxel_size = voxel_size
        self.pts_center_threshold = pts_center_threshold
        self.label2level = label2level
        self.bbox_loss = MODELS.build(bbox_loss)
        self.cls_loss = MODELS.build(cls_loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(len(self.label2level), in_channels, num_reg_outs)

    def _init_layers(self, num_classes: int, in_channels: int,
                     num_reg_outs: int):
        """Initialize layers.

        Args:
            in_channels (int): Number of channels in input tensors.
            num_reg_outs (int): Number of regression layer channels.
            num_classes (int): Number of classes.
        """
        self.conv_reg = ME.MinkowskiConvolution(
            in_channels, num_reg_outs, kernel_size=1, bias=True, dimension=3)
        self.conv_cls = ME.MinkowskiConvolution(
            in_channels, num_classes, kernel_size=1, bias=True, dimension=3)

    def init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.conv_reg.kernel, std=.01)
        nn.init.normal_(self.conv_cls.kernel, std=.01)
        nn.init.constant_(self.conv_cls.bias, bias_init_with_prob(.01))

    def _forward_single(self, x: SparseTensor) -> Tuple[Tensor, ...]:
        """Forward pass per level.

        Args:
            x (SparseTensor): Per level neck output tensor.

        Returns:
            tuple[Tensor]: Per level head predictions.
        """
        reg_final = self.conv_reg(x).features
        reg_distance = torch.exp(reg_final[:, 3:6])
        reg_angle = reg_final[:, 6:]
        bbox_pred = torch.cat((reg_final[:, :3], reg_distance, reg_angle),
                              dim=1)
        cls_pred = self.conv_cls(x).features

        bbox_preds, cls_preds, points = [], [], []
        for permutation in x.decomposition_permutations:
            bbox_preds.append(bbox_pred[permutation])
            cls_preds.append(cls_pred[permutation])
            points.append(x.coordinates[permutation][:, 1:] * self.voxel_size)

        return bbox_preds, cls_preds, points

    def forward(self, x: List[Tensor]) -> Tuple[List[Tensor], ...]:
        """Forward pass.

        Args:
            x (list[Tensor]): Features from the backbone.

        Returns:
            Tuple[List[Tensor], ...]: Predictions of the head.
        """
        bbox_preds, cls_preds, points = [], [], []
        for i in range(len(x)):
            bbox_pred, cls_pred, point = self._forward_single(x[i])
            bbox_preds.append(bbox_pred)
            cls_preds.append(cls_pred)
            points.append(point)
        return bbox_preds, cls_preds, points

    def _loss_by_feat_single(self, bbox_preds: List[Tensor],
                             cls_preds: List[Tensor], points: List[Tensor],
                             gt_bboxes: BaseInstance3DBoxes, gt_labels: Tensor,
                             input_meta: dict) -> Tuple[Tensor, ...]:
        """Loss function of single sample.

        Args:
            bbox_preds (list[Tensor]): Bbox predictions for all levels.
            cls_preds (list[Tensor]): Classification predictions for all
                levels.
            points (list[Tensor]): Final location coordinates for all levels.
            gt_bboxes (:obj:`BaseInstance3DBoxes`): Ground truth boxes.
            gt_labels (Tensor): Ground truth labels.
            input_meta (dict): Scene meta info.

        Returns:
            tuple[Tensor, ...]: Bbox and classification loss
                values and a boolean mask of assigned points.
        """
        num_classes = cls_preds[0].shape[1]
        bbox_targets, cls_targets = self.get_targets(points, gt_bboxes,
                                                     gt_labels, num_classes)
        bbox_preds = torch.cat(bbox_preds)
        cls_preds = torch.cat(cls_preds)
        points = torch.cat(points)

        # cls loss
        cls_loss = self.cls_loss(cls_preds, cls_targets)

        # bbox loss
        pos_mask = cls_targets < num_classes
        pos_bbox_preds = bbox_preds[pos_mask]
        if pos_mask.sum() > 0:
            pos_points = points[pos_mask]
            pos_bbox_preds = bbox_preds[pos_mask]
            pos_bbox_targets = bbox_targets[pos_mask]
            bbox_loss = self.bbox_loss(
                self._bbox_to_loss(
                    self._bbox_pred_to_bbox(pos_points, pos_bbox_preds)),
                self._bbox_to_loss(pos_bbox_targets))
        else:
            bbox_loss = pos_bbox_preds
        return bbox_loss, cls_loss, pos_mask

    def loss_by_feat(self,
                     bbox_preds: List[List[Tensor]],
                     cls_preds: List[List[Tensor]],
                     points: List[List[Tensor]],
                     batch_gt_instances_3d: InstanceList,
                     batch_input_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None,
                     **kwargs) -> dict:
        """Loss function about feature.

        Args:
            bbox_preds (list[list[Tensor]]): Bbox predictions for all scenes.
                The first list contains predictions from different
                levels. The second list contains predictions in a mini-batch.
            cls_preds (list[list[Tensor]]): Classification predictions for all
                scenes. The first list contains predictions from different
                levels. The second list contains predictions in a mini-batch.
            points (list[list[Tensor]]): Final location coordinates for all
                scenes. The first list contains predictions from different
                levels. The second list contains predictions in a mini-batch.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instance_3d.  It usually includes ``bboxes_3d``、`
                `labels_3d``、``depths``、``centers_2d`` and attributes.
            batch_input_metas (list[dict]): Meta information of each image,
                e.g., image size, scaling factor, etc.

        Returns:
            dict: Bbox, and classification losses.
        """
        bbox_losses, cls_losses, pos_masks = [], [], []
        for i in range(len(batch_input_metas)):
            bbox_loss, cls_loss, pos_mask = self._loss_by_feat_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                points=[x[i] for x in points],
                input_meta=batch_input_metas[i],
                gt_bboxes=batch_gt_instances_3d[i].bboxes_3d,
                gt_labels=batch_gt_instances_3d[i].labels_3d)
            if len(bbox_loss) > 0:
                bbox_losses.append(bbox_loss)
            cls_losses.append(cls_loss)
            pos_masks.append(pos_mask)
        return dict(
            bbox_loss=torch.mean(torch.cat(bbox_losses)),
            cls_loss=torch.sum(torch.cat(cls_losses)) /
            torch.sum(torch.cat(pos_masks)))

    def _predict_by_feat_single(self, bbox_preds: List[Tensor],
                                cls_preds: List[Tensor], points: List[Tensor],
                                input_meta: dict) -> InstanceData:
        """Generate boxes for single sample.

        Args:
            center_preds (list[Tensor]): Centerness predictions for all levels.
            bbox_preds (list[Tensor]): Bbox predictions for all levels.
            cls_preds (list[Tensor]): Classification predictions for all
                levels.
            points (list[Tensor]): Final location coordinates for all levels.
            input_meta (dict): Scene meta info.

        Returns:
            InstanceData: Predicted bounding boxes, scores and labels.
        """
        scores = torch.cat(cls_preds).sigmoid()
        bbox_preds = torch.cat(bbox_preds)
        points = torch.cat(points)
        max_scores, _ = scores.max(dim=1)

        if len(scores) > self.test_cfg.nms_pre > 0:
            _, ids = max_scores.topk(self.test_cfg.nms_pre)
            bbox_preds = bbox_preds[ids]
            scores = scores[ids]
            points = points[ids]

        bboxes = self._bbox_pred_to_bbox(points, bbox_preds)
        bboxes, scores, labels = self._single_scene_multiclass_nms(
            bboxes, scores, input_meta)

        bboxes = input_meta['box_type_3d'](
            bboxes,
            box_dim=bboxes.shape[1],
            with_yaw=bboxes.shape[1] == 7,
            origin=(.5, .5, .5))

        results = InstanceData()
        results.bboxes_3d = bboxes
        results.scores_3d = scores
        results.labels_3d = labels
        return results

    def predict_by_feat(self, bbox_preds: List[List[Tensor]], cls_preds,
                        points: List[List[Tensor]],
                        batch_input_metas: List[dict],
                        **kwargs) -> List[InstanceData]:
        """Generate boxes for all scenes.

        Args:
            bbox_preds (list[list[Tensor]]): Bbox predictions for all scenes.
            cls_preds (list[list[Tensor]]): Classification predictions for all
                scenes.
            points (list[list[Tensor]]): Final location coordinates for all
                scenes.
            batch_input_metas (list[dict]): Meta infos for all scenes.

        Returns:
            list[InstanceData]: Predicted bboxes, scores, and labels for
            all scenes.
        """
        results = []
        for i in range(len(batch_input_metas)):
            result = self._predict_by_feat_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                points=[x[i] for x in points],
                input_meta=batch_input_metas[i])
            results.append(result)
        return results

    @staticmethod
    def _bbox_to_loss(bbox):
        """Transform box to the axis-aligned or rotated iou loss format.

        Args:
            bbox (Tensor): 3D box of shape (N, 6) or (N, 7).

        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        # rotated iou loss accepts (x, y, z, w, h, l, heading)
        if bbox.shape[-1] != 6:
            return bbox

        # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
        return torch.stack(
            (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
             bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
             bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
            dim=-1)

    @staticmethod
    def _bbox_pred_to_bbox(points, bbox_pred):
        """Transform predicted bbox parameters to bbox.

        Args:
            points (Tensor): Final locations of shape (N, 3)
            bbox_pred (Tensor): Predicted bbox parameters of shape (N, 6)
                or (N, 8).
        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        x_center = points[:, 0] + bbox_pred[:, 0]
        y_center = points[:, 1] + bbox_pred[:, 1]
        z_center = points[:, 2] + bbox_pred[:, 2]
        base_bbox = torch.stack([
            x_center, y_center, z_center, bbox_pred[:, 3], bbox_pred[:, 4],
            bbox_pred[:, 5]
        ], -1)

        # axis-aligned case
        if bbox_pred.shape[1] == 6:
            return base_bbox

        # rotated case: ..., sin(2a)ln(q), cos(2a)ln(q)
        scale = bbox_pred[:, 3] + bbox_pred[:, 4]
        q = torch.exp(
            torch.sqrt(
                torch.pow(bbox_pred[:, 6], 2) + torch.pow(bbox_pred[:, 7], 2)))
        alpha = 0.5 * torch.atan2(bbox_pred[:, 6], bbox_pred[:, 7])
        return torch.stack(
            (x_center, y_center, z_center, scale / (1 + q), scale /
             (1 + q) * q, bbox_pred[:, 5] + bbox_pred[:, 4], alpha),
            dim=-1)

    @torch.no_grad()
    def get_targets(self, points: Tensor, gt_bboxes: BaseInstance3DBoxes,
                    gt_labels: Tensor, num_classes: int) -> Tuple[Tensor, ...]:
        """Compute targets for final locations for a single scene.

        Args:
            points (list[Tensor]): Final locations for all levels.
            gt_bboxes (BaseInstance3DBoxes): Ground truth boxes.
            gt_labels (Tensor): Ground truth labels.
            num_classes (int): Number of classes.

        Returns:
            tuple[Tensor, ...]: Bbox and classification targets for all
                locations.
        """
        float_max = points[0].new_tensor(1e8)
        levels = torch.cat([
            points[i].new_tensor(i, dtype=torch.long).expand(len(points[i]))
            for i in range(len(points))
        ])
        points = torch.cat(points)
        n_points = len(points)
        n_boxes = len(gt_bboxes)

        if len(gt_labels) == 0:
            return points.new_tensor([]), \
                gt_labels.new_full((n_points,), num_classes)

        boxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
                          dim=1)
        boxes = boxes.to(points.device).expand(n_points, n_boxes, 7)
        points = points.unsqueeze(1).expand(n_points, n_boxes, 3)

        # condition 1: fix level for label
        label2level = gt_labels.new_tensor(self.label2level)
        label_levels = label2level[gt_labels].unsqueeze(0).expand(
            n_points, n_boxes)
        point_levels = torch.unsqueeze(levels, 1).expand(n_points, n_boxes)
        level_condition = label_levels == point_levels

        # condition 2: keep topk location per box by center distance
        center = boxes[..., :3]
        center_distances = torch.sum(torch.pow(center - points, 2), dim=-1)
        center_distances = torch.where(level_condition, center_distances,
                                       float_max)
        topk_distances = torch.topk(
            center_distances,
            min(self.pts_center_threshold + 1, len(center_distances)),
            largest=False,
            dim=0).values[-1]
        topk_condition = center_distances < topk_distances.unsqueeze(0)

        # condition 3: min center distance to box per point
        center_distances = torch.where(topk_condition, center_distances,
                                       float_max)
        min_values, min_ids = center_distances.min(dim=1)
        min_inds = torch.where(min_values < float_max, min_ids, -1)

        bbox_targets = boxes[0][min_inds]
        if not gt_bboxes.with_yaw:
            bbox_targets = bbox_targets[:, :-1]
        cls_targets = torch.where(min_inds >= 0, gt_labels[min_inds],
                                  num_classes)
        return bbox_targets, cls_targets

    def _single_scene_multiclass_nms(self, bboxes: Tensor, scores: Tensor,
                                     input_meta: dict) -> Tuple[Tensor, ...]:
        """Multi-class nms for a single scene.

        Args:
            bboxes (Tensor): Predicted boxes of shape (N_boxes, 6) or
                (N_boxes, 7).
            scores (Tensor): Predicted scores of shape (N_boxes, N_classes).
            input_meta (dict): Scene meta data.

        Returns:
            tuple[Tensor, ...]: Predicted bboxes, scores and labels.
        """
        num_classes = scores.shape[1]
        with_yaw = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for i in range(num_classes):
            ids = scores[:, i] > self.test_cfg.score_thr
            if not ids.any():
                continue

            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]
            if with_yaw:
                nms_function = nms3d
            else:
                class_bboxes = torch.cat(
                    (class_bboxes, torch.zeros_like(class_bboxes[:, :1])),
                    dim=1)
                nms_function = nms3d_normal

            nms_ids = nms_function(class_bboxes, class_scores,
                                   self.test_cfg.iou_thr)
            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(
                bboxes.new_full(
                    class_scores[nms_ids].shape, i, dtype=torch.long))

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0, ))
            nms_labels = bboxes.new_zeros((0, ))

        if not with_yaw:
            nms_bboxes = nms_bboxes[:, :6]

        return nms_bboxes, nms_scores, nms_labels
