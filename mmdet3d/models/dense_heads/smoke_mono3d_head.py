# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
from mmdet.models.utils import (gaussian_radius, gen_gaussian_target,
                                multi_apply)
from mmdet.models.utils.gaussian_target import (get_local_maximum,
                                                get_topk_from_heatmap,
                                                transpose_and_gather_feat)
from mmengine.structures import InstanceData
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.utils import (ConfigType, InstanceList, OptConfigType,
                           OptInstanceList, OptMultiConfig)
from .anchor_free_mono3d_head import AnchorFreeMono3DHead


@MODELS.register_module()
class SMOKEMono3DHead(AnchorFreeMono3DHead):
    r"""Anchor-free head used in `SMOKE <https://arxiv.org/abs/2002.10111>`_

    .. code-block:: none

                /-----> 3*3 conv -----> 1*1 conv -----> cls
        feature
                \-----> 3*3 conv -----> 1*1 conv -----> reg

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        dim_channel (list[int]): indices of dimension offset preds in
            regression heatmap channels.
        ori_channel (list[int]): indices of orientation offset pred in
            regression heatmap channels.
        bbox_coder (:obj:`ConfigDict` or dict): Bbox coder for encoding
            and decoding boxes.
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
            Default: loss_cls=dict(type='GaussionFocalLoss', loss_weight=1.0).
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
            Default: loss_bbox=dict(type='L1Loss', loss_weight=10.0).
        loss_dir (:obj:`ConfigDict` or dict, Optional): Config of direction
            classification loss. In SMOKE, Default: None.
        loss_attr (:obj:`ConfigDict` or dict, Optional): Config of attribute
            classification loss. In SMOKE, Default: None.
        norm_cfg (:obj:`ConfigDict` or dict): Dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict. Defaults to None.
    """  # noqa: E501

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 dim_channel: List[int],
                 ori_channel: List[int],
                 bbox_coder: ConfigType,
                 loss_cls: ConfigType = dict(
                     type='mmdet.GaussionFocalLoss', loss_weight=1.0),
                 loss_bbox: ConfigType = dict(
                     type='mmdet.L1Loss', loss_weight=0.1),
                 loss_dir: OptConfigType = None,
                 loss_attr: OptConfigType = None,
                 norm_cfg: OptConfigType = dict(
                     type='GN', num_groups=32, requires_grad=True),
                 init_cfg: OptMultiConfig = None,
                 **kwargs) -> None:
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_dir=loss_dir,
            loss_attr=loss_attr,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.dim_channel = dim_channel
        self.ori_channel = ori_channel
        self.bbox_coder = TASK_UTILS.build(bbox_coder)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor]]:
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
        """
        return multi_apply(self.forward_single, x)

    def forward_single(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward features of a single scale level.

        Args:
            x (Tensor): Input feature map.

        Returns:
            tuple: Scores for each class, bbox of input feature maps.
        """
        cls_score, bbox_pred, dir_cls_pred, attr_pred, cls_feat, reg_feat = \
            super().forward_single(x)
        cls_score = cls_score.sigmoid()  # turn to 0-1
        cls_score = cls_score.clamp(min=1e-4, max=1 - 1e-4)
        # (N, C, H, W)
        offset_dims = bbox_pred[:, self.dim_channel, ...]
        bbox_pred[:, self.dim_channel, ...] = offset_dims.sigmoid() - 0.5
        # (N, C, H, W)
        vector_ori = bbox_pred[:, self.ori_channel, ...]
        bbox_pred[:, self.ori_channel, ...] = F.normalize(vector_ori)
        return cls_score, bbox_pred

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        batch_img_metas: Optional[List[dict]] = None,
                        rescale: bool = None) -> InstanceList:
        """Generate bboxes from bbox head predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level.
            bbox_preds (list[Tensor]): Box regression for each scale.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            list[:obj:`InstanceData`]: 3D Detection results of each image
            after the post process.
            Each item usually contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instance, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bboxes_3d (Tensor): Contains a tensor with shape
                (num_instances, 7).
        """
        assert len(cls_scores) == len(bbox_preds) == 1
        cam2imgs = torch.stack([
            cls_scores[0].new_tensor(img_meta['cam2img'])
            for img_meta in batch_img_metas
        ])
        trans_mats = torch.stack([
            cls_scores[0].new_tensor(img_meta['trans_mat'])
            for img_meta in batch_img_metas
        ])
        batch_bboxes, batch_scores, batch_topk_labels = self._decode_heatmap(
            cls_scores[0],
            bbox_preds[0],
            batch_img_metas,
            cam2imgs=cam2imgs,
            trans_mats=trans_mats,
            topk=100,
            kernel=3)

        result_list = []
        for img_id in range(len(batch_img_metas)):

            bboxes = batch_bboxes[img_id]
            scores = batch_scores[img_id]
            labels = batch_topk_labels[img_id]

            keep_idx = scores > 0.25
            bboxes = bboxes[keep_idx]
            scores = scores[keep_idx]
            labels = labels[keep_idx]

            bboxes = batch_img_metas[img_id]['box_type_3d'](
                bboxes, box_dim=self.bbox_code_size, origin=(0.5, 0.5, 0.5))
            attrs = None

            results = InstanceData()
            results.bboxes_3d = bboxes
            results.labels_3d = labels
            results.scores_3d = scores

            if attrs is not None:
                results.attr_labels = attrs

            result_list.append(results)

        return result_list

    def _decode_heatmap(self,
                        cls_score: Tensor,
                        reg_pred: Tensor,
                        batch_img_metas: List[dict],
                        cam2imgs: Tensor,
                        trans_mats: Tensor,
                        topk: int = 100,
                        kernel: int = 3) -> Tuple[Tensor, Tensor, Tensor]:
        """Transform outputs into detections raw bbox predictions.

        Args:
            class_score (Tensor): Center predict heatmap,
                shape (B, num_classes, H, W).
            reg_pred (Tensor): Box regression map.
                shape (B, channel, H , W).
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cam2imgs (Tensor): Camera intrinsic matrixs.
                shape (B, 4, 4)
            trans_mats (Tensor): Transformation matrix from original image
                to feature map.
                shape: (batch, 3, 3)
            topk (int): Get top k center keypoints from heatmap. Default 100.
            kernel (int): Max pooling kernel for extract local maximum pixels.
               Default 3.

        Returns:
            tuple[torch.Tensor]: Decoded output of SMOKEHead, containing
               the following Tensors:

              - batch_bboxes (Tensor): Coords of each 3D box.
                    shape (B, k, 7)
              - batch_scores (Tensor): Scores of each 3D box.
                    shape (B, k)
              - batch_topk_labels (Tensor): Categories of each 3D box.
                    shape (B, k)
        """
        img_h, img_w = batch_img_metas[0]['pad_shape'][:2]
        bs, _, feat_h, feat_w = cls_score.shape

        center_heatmap_pred = get_local_maximum(cls_score, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
            center_heatmap_pred, k=topk)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        regression = transpose_and_gather_feat(reg_pred, batch_index)
        regression = regression.view(-1, 8)

        points = torch.cat([topk_xs.view(-1, 1),
                            topk_ys.view(-1, 1).float()],
                           dim=1)
        locations, dimensions, orientations = self.bbox_coder.decode(
            regression, points, batch_topk_labels, cam2imgs, trans_mats)

        batch_bboxes = torch.cat((locations, dimensions, orientations), dim=1)
        batch_bboxes = batch_bboxes.view(bs, -1, self.bbox_code_size)
        return batch_bboxes, batch_scores, batch_topk_labels

    def get_predictions(self, labels_3d: Tensor, centers_2d: Tensor,
                        gt_locations: Tensor, gt_dimensions: Tensor,
                        gt_orientations: Tensor, indices: Tensor,
                        batch_img_metas: List[dict], pred_reg: Tensor) -> dict:
        """Prepare predictions for computing loss.

        Args:
            labels_3d (Tensor): Labels of each 3D box.
                shape (B, max_objs, )
            centers_2d (Tensor): Coords of each projected 3D box
                center on image. shape (B * max_objs, 2)
            gt_locations (Tensor): Coords of each 3D box's location.
                shape (B * max_objs, 3)
            gt_dimensions (Tensor): Dimensions of each 3D box.
                shape (N, 3)
            gt_orientations (Tensor): Orientation(yaw) of each 3D box.
                shape (N, 1)
            indices (Tensor): Indices of the existence of the 3D box.
                shape (B * max_objs, )
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            pre_reg (Tensor): Box regression map.
                shape (B, channel, H , W).

        Returns:
            dict: the dict has components below:

            - bbox3d_yaws (:obj:`CameraInstance3DBoxes`):
                bbox calculated using pred orientations.
            - bbox3d_dims (:obj:`CameraInstance3DBoxes`):
                bbox calculated using pred dimensions.
            - bbox3d_locs (:obj:`CameraInstance3DBoxes`):
                bbox calculated using pred locations.
        """
        batch, channel = pred_reg.shape[0], pred_reg.shape[1]
        w = pred_reg.shape[3]
        cam2imgs = torch.stack([
            gt_locations.new_tensor(img_meta['cam2img'])
            for img_meta in batch_img_metas
        ])
        trans_mats = torch.stack([
            gt_locations.new_tensor(img_meta['trans_mat'])
            for img_meta in batch_img_metas
        ])
        centers_2d_inds = centers_2d[:, 1] * w + centers_2d[:, 0]
        centers_2d_inds = centers_2d_inds.view(batch, -1)
        pred_regression = transpose_and_gather_feat(pred_reg, centers_2d_inds)
        pred_regression_pois = pred_regression.view(-1, channel)
        locations, dimensions, orientations = self.bbox_coder.decode(
            pred_regression_pois, centers_2d, labels_3d, cam2imgs, trans_mats,
            gt_locations)

        locations, dimensions, orientations = locations[indices], dimensions[
            indices], orientations[indices]

        locations[:, 1] += dimensions[:, 1] / 2

        gt_locations = gt_locations[indices]

        assert len(locations) == len(gt_locations)
        assert len(dimensions) == len(gt_dimensions)
        assert len(orientations) == len(gt_orientations)
        bbox3d_yaws = self.bbox_coder.encode(gt_locations, gt_dimensions,
                                             orientations, batch_img_metas)
        bbox3d_dims = self.bbox_coder.encode(gt_locations, dimensions,
                                             gt_orientations, batch_img_metas)
        bbox3d_locs = self.bbox_coder.encode(locations, gt_dimensions,
                                             gt_orientations, batch_img_metas)

        pred_bboxes = dict(ori=bbox3d_yaws, dim=bbox3d_dims, loc=bbox3d_locs)

        return pred_bboxes

    def get_targets(self, batch_gt_instances_3d: InstanceList,
                    batch_gt_instances: InstanceList, feat_shape: Tuple[int],
                    batch_img_metas: List[dict]) -> Tuple[Tensor, int, dict]:
        """Get training targets for batch images.

        Args:
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instance_3d.  It usually includes ``bboxes_3d``、
                ``labels_3d``、``depths``、``centers_2d`` and attributes.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes``、``labels``.
            feat_shape (tuple[int]): Feature map shape with value,
                shape (B, _, H, W).
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            tuple[Tensor, int, dict]: The Tensor value is the targets of
                center heatmap, the dict has components below:

              - gt_centers_2d (Tensor): Coords of each projected 3D box
                    center on image. shape (B * max_objs, 2)
              - gt_labels_3d (Tensor): Labels of each 3D box.
                    shape (B, max_objs, )
              - indices (Tensor): Indices of the existence of the 3D box.
                    shape (B * max_objs, )
              - affine_indices (Tensor): Indices of the affine of the 3D box.
                    shape (N, )
              - gt_locs (Tensor): Coords of each 3D box's location.
                    shape (N, 3)
              - gt_dims (Tensor): Dimensions of each 3D box.
                    shape (N, 3)
              - gt_yaws (Tensor): Orientation(yaw) of each 3D box.
                    shape (N, 1)
              - gt_cors (Tensor): Coords of the corners of each 3D box.
                    shape (N, 8, 3)
        """

        gt_bboxes = [
            gt_instances.bboxes for gt_instances in batch_gt_instances
        ]
        gt_labels = [
            gt_instances.labels for gt_instances in batch_gt_instances
        ]
        gt_bboxes_3d = [
            gt_instances_3d.bboxes_3d
            for gt_instances_3d in batch_gt_instances_3d
        ]
        gt_labels_3d = [
            gt_instances_3d.labels_3d
            for gt_instances_3d in batch_gt_instances_3d
        ]
        centers_2d = [
            gt_instances_3d.centers_2d
            for gt_instances_3d in batch_gt_instances_3d
        ]
        img_shape = batch_img_metas[0]['pad_shape']

        reg_mask = torch.stack([
            gt_bboxes[0].new_tensor(
                not img_meta['affine_aug'], dtype=torch.bool)
            for img_meta in batch_img_metas
        ])

        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)  # 1/4
        height_ratio = float(feat_h / img_h)  # 1/4

        assert width_ratio == height_ratio

        center_heatmap_target = gt_bboxes[-1].new_zeros(
            [bs, self.num_classes, feat_h, feat_w])

        gt_centers_2d = centers_2d.copy()

        for batch_id in range(bs):
            gt_bbox = gt_bboxes[batch_id]
            gt_label = gt_labels[batch_id]
            # project centers_2d from input image to feat map
            gt_center_2d = gt_centers_2d[batch_id] * width_ratio

            for j, center in enumerate(gt_center_2d):
                center_x_int, center_y_int = center.int()
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio
                radius = gaussian_radius([scale_box_h, scale_box_w],
                                         min_overlap=0.7)
                radius = max(0, int(radius))
                ind = gt_label[j]
                gen_gaussian_target(center_heatmap_target[batch_id, ind],
                                    [center_x_int, center_y_int], radius)

        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        num_ctrs = [center_2d.shape[0] for center_2d in centers_2d]
        max_objs = max(num_ctrs)

        reg_inds = torch.cat(
            [reg_mask[i].repeat(num_ctrs[i]) for i in range(bs)])

        inds = torch.zeros((bs, max_objs),
                           dtype=torch.bool).to(centers_2d[0].device)

        # put gt 3d bboxes to gpu
        gt_bboxes_3d = [
            gt_bbox_3d.to(centers_2d[0].device) for gt_bbox_3d in gt_bboxes_3d
        ]

        batch_centers_2d = centers_2d[0].new_zeros((bs, max_objs, 2))
        batch_labels_3d = gt_labels_3d[0].new_zeros((bs, max_objs))
        batch_gt_locations = \
            gt_bboxes_3d[0].tensor.new_zeros((bs, max_objs, 3))
        for i in range(bs):
            inds[i, :num_ctrs[i]] = 1
            batch_centers_2d[i, :num_ctrs[i]] = centers_2d[i]
            batch_labels_3d[i, :num_ctrs[i]] = gt_labels_3d[i]
            batch_gt_locations[i, :num_ctrs[i]] = \
                gt_bboxes_3d[i].tensor[:, :3]

        inds = inds.flatten()
        batch_centers_2d = batch_centers_2d.view(-1, 2) * width_ratio
        batch_gt_locations = batch_gt_locations.view(-1, 3)

        # filter the empty image, without gt_bboxes_3d
        gt_bboxes_3d = [
            gt_bbox_3d for gt_bbox_3d in gt_bboxes_3d
            if gt_bbox_3d.tensor.shape[0] > 0
        ]

        gt_dimensions = torch.cat(
            [gt_bbox_3d.tensor[:, 3:6] for gt_bbox_3d in gt_bboxes_3d])
        gt_orientations = torch.cat([
            gt_bbox_3d.tensor[:, 6].unsqueeze(-1)
            for gt_bbox_3d in gt_bboxes_3d
        ])
        gt_corners = torch.cat(
            [gt_bbox_3d.corners for gt_bbox_3d in gt_bboxes_3d])

        target_labels = dict(
            gt_centers_2d=batch_centers_2d.long(),
            gt_labels_3d=batch_labels_3d,
            indices=inds,
            reg_indices=reg_inds,
            gt_locs=batch_gt_locations,
            gt_dims=gt_dimensions,
            gt_yaws=gt_orientations,
            gt_cors=gt_corners)

        return center_heatmap_target, avg_factor, target_labels

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            batch_gt_instances_3d: InstanceList,
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level.
                shape (num_gt, 4).
            bbox_preds (list[Tensor]): Box dims is a 4D-tensor, the channel
                number is bbox_code_size.
                shape (B, 7, H, W).
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instance_3d.  It usually includes ``bboxes_3d``、
                ``labels_3d``、``depths``、``centers_2d`` and attributes.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes``、``labels``.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components, which has
                components below:

            - loss_cls (Tensor): loss of cls heatmap.
            - loss_bbox (Tensor): loss of bbox heatmap.
        """
        assert len(cls_scores) == len(bbox_preds) == 1
        center_2d_heatmap = cls_scores[0]
        pred_reg = bbox_preds[0]

        center_2d_heatmap_target, avg_factor, target_labels = \
            self.get_targets(batch_gt_instances_3d,
                             batch_gt_instances,
                             center_2d_heatmap.shape,
                             batch_img_metas)

        pred_bboxes = self.get_predictions(
            labels_3d=target_labels['gt_labels_3d'],
            centers_2d=target_labels['gt_centers_2d'],
            gt_locations=target_labels['gt_locs'],
            gt_dimensions=target_labels['gt_dims'],
            gt_orientations=target_labels['gt_yaws'],
            indices=target_labels['indices'],
            batch_img_metas=batch_img_metas,
            pred_reg=pred_reg)

        loss_cls = self.loss_cls(
            center_2d_heatmap, center_2d_heatmap_target, avg_factor=avg_factor)

        reg_inds = target_labels['reg_indices']

        loss_bbox_oris = self.loss_bbox(
            pred_bboxes['ori'].corners[reg_inds, ...],
            target_labels['gt_cors'][reg_inds, ...])

        loss_bbox_dims = self.loss_bbox(
            pred_bboxes['dim'].corners[reg_inds, ...],
            target_labels['gt_cors'][reg_inds, ...])

        loss_bbox_locs = self.loss_bbox(
            pred_bboxes['loc'].corners[reg_inds, ...],
            target_labels['gt_cors'][reg_inds, ...])

        loss_bbox = loss_bbox_dims + loss_bbox_locs + loss_bbox_oris

        loss_dict = dict(loss_cls=loss_cls, loss_bbox=loss_bbox)

        return loss_dict
