# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
from mmcv.cnn import Scale
# from mmcv.ops import nms3d, nms3d_normal
from mmdet.models.utils import multi_apply
from mmdet.utils import reduce_mean
# from mmengine.config import ConfigDict
from mmengine.model import BaseModule, bias_init_with_prob, normal_init
from mmengine.structures import InstanceData
from torch import Tensor, nn

from mmdet3d.registry import MODELS, TASK_UTILS
# from mmdet3d.structures.bbox_3d.utils import rotation_3d_in_axis
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils.typing_utils import (ConfigType, InstanceList,
                                        OptConfigType, OptInstanceList)


@torch.no_grad()
def get_points(n_voxels, voxel_size, origin):
    # origin: point-cloud center.
    points = torch.stack(
        torch.meshgrid([
            torch.arange(n_voxels[0]),  # 40 W width, x
            torch.arange(n_voxels[1]),  # 40 D depth, y
            torch.arange(n_voxels[2])  # 16 H Height, z
        ]))
    new_origin = origin - n_voxels / 2. * voxel_size
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points


@MODELS.register_module()
class NerfDetHead(BaseModule):
    r"""`ImVoxelNet<https://arxiv.org/abs/2106.01178>`_ head for indoor
    datasets.

    Args:
        n_classes (int): Number of classes.
        n_levels (int): Number of feature levels.
        n_channels (int): Number of channels in input tensors.
        n_reg_outs (int): Number of regression layer channels.
        pts_assign_threshold (int): Min number of location per box to
            be assigned with.
        pts_center_threshold (int): Max number of locations per box to
            be assigned with.
        center_loss (dict, optional): Config of centerness loss.
            Default: dict(type='CrossEntropyLoss', use_sigmoid=True).
        bbox_loss (dict, optional): Config of bbox loss.
            Default: dict(type='RotatedIoU3DLoss').
        cls_loss (dict, optional): Config of classification loss.
            Default: dict(type='FocalLoss').
        train_cfg (dict, optional): Config for train stage. Defaults to None.
        test_cfg (dict, optional): Config for test stage. Defaults to None.
        init_cfg (dict, optional): Config for weight initialization.
            Defaults to None.
    """

    def __init__(self,
                 n_classes: int,
                 n_levels: int,
                 n_channels: int,
                 n_reg_outs: int,
                 pts_assign_threshold: int,
                 pts_center_threshold: int,
                 prior_generator: ConfigType,
                 center_loss: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss', use_sigmoid=True),
                 bbox_loss: ConfigType = dict(type='RotatedIoU3DLoss'),
                 cls_loss: ConfigType = dict(type='mmdet.FocalLoss'),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        super(NerfDetHead, self).__init__(init_cfg)
        self.n_classes = n_classes
        self.n_levels = n_levels
        self.n_reg_outs = n_reg_outs
        self.pts_assign_threshold = pts_assign_threshold
        self.pts_center_threshold = pts_center_threshold
        self.prior_generator = TASK_UTILS.build(prior_generator)
        self.center_loss = MODELS.build(center_loss)
        self.bbox_loss = MODELS.build(bbox_loss)
        self.cls_loss = MODELS.build(cls_loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(n_channels, n_reg_outs, n_classes, n_levels)

    def _init_layers(self, n_channels, n_reg_outs, n_classes, n_levels):
        """Initialize neural network layers of the head."""
        self.conv_center = nn.Conv3d(n_channels, 1, 3, padding=1, bias=False)
        self.conv_reg = nn.Conv3d(
            n_channels, n_reg_outs, 3, padding=1, bias=False)
        self.conv_cls = nn.Conv3d(n_channels, n_classes, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.) for _ in range(n_levels)])

    def init_weights(self):
        """Initialize all layer weights."""
        normal_init(self.conv_center, std=.01)
        normal_init(self.conv_reg, std=.01)
        normal_init(self.conv_cls, std=.01, bias=bias_init_with_prob(.01))

    def _forward_single(self, x: Tensor, scale: Scale):
        """Forward pass per level.

        Args:
            x (Tensor): Per level 3d neck output tensor.
            scale (mmcv.cnn.Scale): Per level multiplication weight.

        Returns:
            tuple[Tensor]: Centerness, bbox and classification predictions.
        """
        return (self.conv_center(x), torch.exp(scale(self.conv_reg(x))),
                self.conv_cls(x))

    def forward(self, x):
        return multi_apply(self._forward_single, x, self.scales)

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             **kwargs) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`NeRFDet3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        valid_pred = x[-1]
        outs = self(x[:-1])

        batch_gt_instances_3d = []
        batch_gt_instances_ignore = []
        batch_input_metas = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
            batch_gt_instances_ignore.append(
                data_sample.get('ignored_instances', None))

        loss_inputs = outs + (valid_pred, batch_gt_instances_3d,
                              batch_input_metas, batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def loss_by_feat(self,
                     center_preds: List[List[Tensor]],
                     bbox_preds: List[List[Tensor]],
                     cls_preds: List[List[Tensor]],
                     valid_pred: Tensor,
                     batch_gt_instances_3d: InstanceList,
                     batch_input_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None,
                     **kwargs) -> dict:
        """Per scene loss function.

        Args:
            center_preds (list[list[Tensor]]): Centerness predictions for
                all scenes. The first list contains predictions from different
                levels. The second list contains predictions in a mini-batch.
            bbox_preds (list[list[Tensor]]): Bbox predictions for all scenes.
                The first list contains predictions from different
                levels. The second list contains predictions in a mini-batch.
            cls_preds (list[list[Tensor]]): Classification predictions for all
                scenes. The first list contains predictions from different
                levels. The second list contains predictions in a mini-batch.
            valid_pred (Tensor): Valid mask prediction for all scenes.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instance_3d.  It usually includes ``bboxes_3d``、`
                `labels_3d``、``depths``、``centers_2d`` and attributes.
            batch_input_metas (list[dict]): Meta information of each image,
                e.g., image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict: Centerness, bbox, and classification loss values.
        """
        valid_preds = self._upsample_valid_preds(valid_pred, center_preds)
        center_losses, bbox_losses, cls_losses = [], [], []
        for i in range(len(batch_input_metas)):
            center_loss, bbox_loss, cls_loss = self._loss_by_feat_single(
                center_preds=[x[i] for x in center_preds],
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                valid_preds=[x[i] for x in valid_preds],
                input_meta=batch_input_metas[i],
                gt_bboxes=batch_gt_instances_3d[i].bboxes_3d,
                gt_labels=batch_gt_instances_3d[i].labels_3d)
            center_losses.append(center_loss)
            bbox_losses.append(bbox_loss)
            cls_losses.append(cls_loss)
        return dict(
            center_loss=torch.mean(torch.stack(center_losses)),
            bbox_loss=torch.mean(torch.stack(bbox_losses)),
            cls_loss=torch.mean(torch.stack(cls_losses)))

    def _loss_by_feat_single(self, center_preds, bbox_preds, cls_preds,
                             valid_preds, input_meta, gt_bboxes, gt_labels):
        featmap_sizes = [featmap.size()[-3:] for featmap in center_preds]
        points = self._get_points(
            featmap_sizes=featmap_sizes,
            origin=input_meta['lidar2img']['origin'],
            device=gt_bboxes.device)
        center_targets, bbox_targets, cls_targets = self._get_targets(
            points, gt_bboxes, gt_labels)

        center_preds = torch.cat(
            [x.permute(1, 2, 3, 0).reshape(-1) for x in center_preds])
        bbox_preds = torch.cat([
            x.permute(1, 2, 3, 0).reshape(-1, x.shape[0]) for x in bbox_preds
        ])
        cls_preds = torch.cat(
            [x.permute(1, 2, 3, 0).reshape(-1, x.shape[0]) for x in cls_preds])
        valid_preds = torch.cat(
            [x.permute(1, 2, 3, 0).reshape(-1) for x in valid_preds])
        points = torch.cat(points)

        # cls loss
        pos_inds = torch.nonzero(
            torch.logical_and(cls_targets >= 0, valid_preds)).squeeze(1)
        n_pos = points.new_tensor(len(pos_inds))
        n_pos = max(reduce_mean(n_pos), 1.)
        if torch.any(valid_preds):
            cls_loss = self.cls_loss(
                cls_preds[valid_preds],
                cls_targets[valid_preds],
                avg_factor=n_pos)
        else:
            cls_loss = cls_preds[valid_preds].sum()

        # bbox and centerness losses
        pos_center_preds = center_preds[pos_inds]
        pos_bbox_preds = bbox_preds[pos_inds]
        if len(pos_inds) > 0:
            pos_center_targets = center_targets[pos_inds]
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_points = points[pos_inds]
            center_loss = self.center_loss(
                pos_center_preds, pos_center_targets, avg_factor=n_pos)
            bbox_loss = self.bbox_loss(
                self._bbox_pred_to_bbox(pos_points, pos_bbox_preds),
                pos_bbox_targets,
                weight=pos_center_targets,
                avg_factor=pos_center_targets.sum())
        else:
            center_loss = pos_center_preds.sum()
            bbox_loss = pos_bbox_preds.sum()
        return center_loss, bbox_loss, cls_loss

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the 3D detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`NeRFDet3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_pts_panoptic_seg` and
                `gt_pts_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each sample
            after the post process.
            Each item usually contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (BaseInstance3DBoxes): Prediction of bboxes,
              contains a tensor with shape (num_instances, C), where
              C >= 6.
        """
        batch_input_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        valid_pred = x[-1]
        outs = self(x[:-1])
        predictions = self.predict_by_feat(
            *outs,
            valid_pred=valid_pred,
            batch_input_metas=batch_input_metas,
            rescale=rescale)
        return predictions

    def predict_by_feat(self, center_preds: List[List[Tensor]],
                        bbox_preds: List[List[Tensor]],
                        cls_preds: List[List[Tensor]], valid_pred: Tensor,
                        batch_input_metas: List[dict],
                        **kwargs) -> List[InstanceData]:
        """Generate boxes for all scenes.

        Args:
            center_preds (list[list[Tensor]]): Centerness predictions for
                all scenes.
            bbox_preds (list[list[Tensor]]): Bbox predictions for all scenes.
            cls_preds (list[list[Tensor]]): Classification predictions for all
                scenes.
            valid_pred (Tensor): Valid mask prediction for all scenes.
            batch_input_metas (list[dict]): Meta infos for all scenes.

        Returns:
            list[tuple[Tensor]]: Predicted bboxes, scores, and labels for
                all scenes.
        """
        valid_preds = self._upsample_valid_preds(valid_pred, center_preds)
        results = []
        for i in range(len(batch_input_metas)):
            results.append(
                self._predict_by_feat_single(
                    center_preds=[x[i] for x in center_preds],
                    bbox_preds=[x[i] for x in bbox_preds],
                    cls_preds=[x[i] for x in cls_preds],
                    valid_preds=[x[i] for x in valid_preds],
                    input_meta=batch_input_metas[i]))
        return results

    def _predict_by_feat_single(self, center_preds: List[Tensor],
                                bbox_preds: List[Tensor],
                                cls_preds: List[Tensor],
                                valid_preds: List[Tensor],
                                input_meta: dict) -> InstanceData:
        """Generate boxes for single sample.

        Args:
            center_preds (list[Tensor]): Centerness predictions for all levels.
            bbox_preds (list[Tensor]): Bbox predictions for all levels.
            cls_preds (list[Tensor]): Classification predictions for all
                levels.
            valid_preds (tuple[Tensor]): Upsampled valid masks for all feature
                levels.
            input_meta (dict): Scene meta info.

        Returns:
            tuple[Tensor]: Predicted bounding boxes, scores and labels.
        """
        featmap_sizes = [featmap.size()[-3:] for featmap in center_preds]
        points = self._get_points(
            featmap_sizes=featmap_sizes,
            origin=input_meta['lidar2img']['origin'],
            device=center_preds[0].device)
        mlvl_bboxes, mlvl_scores = [], []
        for center_pred, bbox_pred, cls_pred, valid_pred, point in zip(
                center_preds, bbox_preds, cls_preds, valid_preds, points):
            center_pred = center_pred.permute(1, 2, 3, 0).reshape(-1, 1)
            bbox_pred = bbox_pred.permute(1, 2, 3,
                                          0).reshape(-1, bbox_pred.shape[0])
            cls_pred = cls_pred.permute(1, 2, 3,
                                        0).reshape(-1, cls_pred.shape[0])
            valid_pred = valid_pred.permute(1, 2, 3, 0).reshape(-1, 1)
            scores = cls_pred.sigmoid() * center_pred.sigmoid() * valid_pred
            max_scores, _ = scores.max(dim=1)

            if len(scores) > self.test_cfg.nms_pre > 0:
                _, ids = max_scores.topk(self.test_cfg.nms_pre)
                bbox_pred = bbox_pred[ids]
                scores = scores[ids]
                point = point[ids]

            bboxes = self._bbox_pred_to_bbox(point, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        bboxes = torch.cat(mlvl_bboxes)
        scores = torch.cat(mlvl_scores)
        bboxes, scores, labels = self._nms(bboxes, scores, input_meta)

        bboxes = input_meta['box_type_3d'](
            bboxes, box_dim=6, with_yaw=False, origin=(.5, .5, .5))

        results = InstanceData()
        results.bboxes_3d = bboxes
        results.scores_3d = scores
        results.labels_3d = labels
        return results

    @staticmethod
    def _upsample_valid_preds(valid_pred, features):
        """Upsample valid mask predictions.

        Args:
            valid_pred (Tensor): Valid mask prediction.
            features (Tensor): Feature tensor.

        Returns:
            tuple[Tensor]: Upsampled valid masks for all feature levels.
        """
        return [
            nn.Upsample(size=x.shape[-3:],
                        mode='trilinear')(valid_pred).round().bool()
            for x in features
        ]

    @torch.no_grad()
    def _get_points(self, featmap_sizes, origin, device):
        mlvl_points = []
        tmp_voxel_size = [.16, .16, .2]
        for i, featmap_size in enumerate(featmap_sizes):
            mlvl_points.append(
                get_points(
                    n_voxels=torch.tensor(featmap_size),
                    voxel_size=torch.tensor(tmp_voxel_size) * (2**i),
                    origin=torch.tensor(origin)).reshape(3, -1).transpose(
                        0, 1).to(device))
        return mlvl_points

    def _bbox_pred_to_bbox(self, points, bbox_pred):
        return torch.stack([
            points[:, 0] - bbox_pred[:, 0], points[:, 1] - bbox_pred[:, 2],
            points[:, 2] - bbox_pred[:, 4], points[:, 0] + bbox_pred[:, 1],
            points[:, 1] + bbox_pred[:, 3], points[:, 2] + bbox_pred[:, 5]
        ], -1)

    def _bbox_pred_to_loss(self, points, bbox_preds):
        return self._bbox_pred_to_bbox(points, bbox_preds)

    # The function is directly copied from FCAF3DHead.
    @staticmethod
    def _get_face_distances(points, boxes):
        """Calculate distances from point to box faces.

        Args:
            points (Tensor): Final locations of shape (N_points, N_boxes, 3).
            boxes (Tensor): 3D boxes of shape (N_points, N_boxes, 7)

        Returns:
            Tensor: Face distances of shape (N_points, N_boxes, 6),
                (dx_min, dx_max, dy_min, dy_max, dz_min, dz_max).
        """
        dx_min = points[..., 0] - boxes[..., 0] + boxes[..., 3] / 2
        dx_max = boxes[..., 0] + boxes[..., 3] / 2 - points[..., 0]
        dy_min = points[..., 1] - boxes[..., 1] + boxes[..., 4] / 2
        dy_max = boxes[..., 1] + boxes[..., 4] / 2 - points[..., 1]
        dz_min = points[..., 2] - boxes[..., 2] + boxes[..., 5] / 2
        dz_max = boxes[..., 2] + boxes[..., 5] / 2 - points[..., 2]
        return torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max),
                           dim=-1)

    @staticmethod
    def _get_centerness(face_distances):
        """Compute point centerness w.r.t containing box.

        Args:
            face_distances (Tensor): Face distances of shape (B, N, 6),
                (dx_min, dx_max, dy_min, dy_max, dz_min, dz_max).

        Returns:
            Tensor: Centerness of shape (B, N).
        """
        x_dims = face_distances[..., [0, 1]]
        y_dims = face_distances[..., [2, 3]]
        z_dims = face_distances[..., [4, 5]]
        centerness_targets = x_dims.min(dim=-1)[0] / x_dims.max(dim=-1)[0] * \
            y_dims.min(dim=-1)[0] / y_dims.max(dim=-1)[0] * \
            z_dims.min(dim=-1)[0] / z_dims.max(dim=-1)[0]
        return torch.sqrt(centerness_targets)

    @torch.no_grad()
    def _get_targets(self, points, gt_bboxes, gt_labels):
        """Compute targets for final locations for a single scene.

        Args:
            points (list[Tensor]): Final locations for all levels.
            gt_bboxes (BaseInstance3DBoxes): Ground truth boxes.
            gt_labels (Tensor): Ground truth labels.

        Returns:
            tuple[Tensor]: Centerness, bbox and classification
                targets for all locations.
        """
        float_max = 1e8
        expanded_scales = [
            points[i].new_tensor(i).expand(len(points[i])).to(gt_labels.device)
            for i in range(len(points))
        ]
        points = torch.cat(points, dim=0).to(gt_labels.device)
        scales = torch.cat(expanded_scales, dim=0)

        # below is based on FCOSHead._get_target_single
        n_points = len(points)
        n_boxes = len(gt_bboxes)
        volumes = gt_bboxes.volume.to(points.device)
        volumes = volumes.expand(n_points, n_boxes).contiguous()
        gt_bboxes = torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:6]), dim=1)
        gt_bboxes = gt_bboxes.to(points.device).expand(n_points, n_boxes, 6)
        expanded_points = points.unsqueeze(1).expand(n_points, n_boxes, 3)
        bbox_targets = self._get_face_distances(expanded_points, gt_bboxes)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets[..., :6].min(
            -1)[0] > 0  # skip angle

        # condition2: positive points per scale >= limit
        # calculate positive points per scale
        n_pos_points_per_scale = []
        for i in range(self.n_levels):
            n_pos_points_per_scale.append(
                torch.sum(inside_gt_bbox_mask[scales == i], dim=0))
        # find best scale
        n_pos_points_per_scale = torch.stack(n_pos_points_per_scale, dim=0)
        lower_limit_mask = n_pos_points_per_scale < self.pts_assign_threshold
        # fix nondeterministic argmax for torch<1.7
        extra = torch.arange(self.n_levels, 0, -1).unsqueeze(1).expand(
            self.n_levels, n_boxes).to(lower_limit_mask.device)
        lower_index = torch.argmax(lower_limit_mask.int() * extra, dim=0) - 1
        lower_index = torch.where(lower_index < 0,
                                  torch.zeros_like(lower_index), lower_index)
        all_upper_limit_mask = torch.all(
            torch.logical_not(lower_limit_mask), dim=0)
        best_scale = torch.where(
            all_upper_limit_mask,
            torch.ones_like(all_upper_limit_mask) * self.n_levels - 1,
            lower_index)
        # keep only points with best scale
        best_scale = torch.unsqueeze(best_scale, 0).expand(n_points, n_boxes)
        scales = torch.unsqueeze(scales, 1).expand(n_points, n_boxes)
        inside_best_scale_mask = best_scale == scales

        # condition3: limit topk locations per box by centerness
        centerness = self._get_centerness(bbox_targets)
        centerness = torch.where(inside_gt_bbox_mask, centerness,
                                 torch.ones_like(centerness) * -1)
        centerness = torch.where(inside_best_scale_mask, centerness,
                                 torch.ones_like(centerness) * -1)
        top_centerness = torch.topk(
            centerness, self.pts_center_threshold + 1, dim=0).values[-1]
        inside_top_centerness_mask = centerness > top_centerness.unsqueeze(0)

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        volumes = torch.where(inside_gt_bbox_mask, volumes,
                              torch.ones_like(volumes) * float_max)
        volumes = torch.where(inside_best_scale_mask, volumes,
                              torch.ones_like(volumes) * float_max)
        volumes = torch.where(inside_top_centerness_mask, volumes,
                              torch.ones_like(volumes) * float_max)
        min_area, min_area_inds = volumes.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels = torch.where(min_area == float_max,
                             torch.ones_like(labels) * -1, labels)
        bbox_targets = bbox_targets[range(n_points), min_area_inds]
        centerness_targets = self._get_centerness(bbox_targets)

        return centerness_targets, self._bbox_pred_to_bbox(
            points, bbox_targets), labels

    def _nms(self, bboxes, scores, img_meta):
        scores, labels = scores.max(dim=1)
        ids = scores > self.test_cfg.score_thr
        bboxes = bboxes[ids]
        scores = scores[ids]
        labels = labels[ids]
        ids = self.aligned_3d_nms(bboxes, scores, labels,
                                  self.test_cfg.iou_thr)
        bboxes = bboxes[ids]
        bboxes = torch.stack(
            ((bboxes[:, 0] + bboxes[:, 3]) / 2.,
             (bboxes[:, 1] + bboxes[:, 4]) / 2.,
             (bboxes[:, 2] + bboxes[:, 5]) / 2., bboxes[:, 3] - bboxes[:, 0],
             bboxes[:, 4] - bboxes[:, 1], bboxes[:, 5] - bboxes[:, 2]),
            dim=1)
        return bboxes, scores[ids], labels[ids]

    @staticmethod
    def aligned_3d_nms(boxes, scores, classes, thresh):
        """3d nms for aligned boxes.

        Args:
            boxes (torch.Tensor): Aligned box with shape [n, 6].
            scores (torch.Tensor): Scores of each box.
            classes (torch.Tensor): Class of each box.
            thresh (float): Iou threshold for nms.

        Returns:
            torch.Tensor: Indices of selected boxes.
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        z1 = boxes[:, 2]
        x2 = boxes[:, 3]
        y2 = boxes[:, 4]
        z2 = boxes[:, 5]
        area = (x2 - x1) * (y2 - y1) * (z2 - z1)
        zero = boxes.new_zeros(1, )

        score_sorted = torch.argsort(scores)
        pick = []
        while (score_sorted.shape[0] != 0):
            last = score_sorted.shape[0]
            i = score_sorted[-1]
            pick.append(i)

            xx1 = torch.max(x1[i], x1[score_sorted[:last - 1]])
            yy1 = torch.max(y1[i], y1[score_sorted[:last - 1]])
            zz1 = torch.max(z1[i], z1[score_sorted[:last - 1]])
            xx2 = torch.min(x2[i], x2[score_sorted[:last - 1]])
            yy2 = torch.min(y2[i], y2[score_sorted[:last - 1]])
            zz2 = torch.min(z2[i], z2[score_sorted[:last - 1]])
            classes1 = classes[i]
            classes2 = classes[score_sorted[:last - 1]]
            inter_l = torch.max(zero, xx2 - xx1)
            inter_w = torch.max(zero, yy2 - yy1)
            inter_h = torch.max(zero, zz2 - zz1)

            inter = inter_l * inter_w * inter_h
            iou = inter / (area[i] + area[score_sorted[:last - 1]] - inter)
            iou = iou * (classes1 == classes2).float()
            score_sorted = score_sorted[torch.nonzero(
                iou <= thresh, as_tuple=False).flatten()]

        indices = boxes.new_tensor(pick, dtype=torch.long)
        return indices
