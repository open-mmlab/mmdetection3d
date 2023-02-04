# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
from mmdet.models.utils import (gaussian_radius, gen_gaussian_target,
                                multi_apply)
from mmdet.models.utils.gaussian_target import (get_local_maximum,
                                                get_topk_from_heatmap,
                                                transpose_and_gather_feat)
from mmengine.config import ConfigDict
from mmengine.model import xavier_init
from mmengine.structures import InstanceData
from torch import Tensor
from torch import nn as nn

from mmdet3d.models.layers import EdgeFusionModule
from mmdet3d.models.task_modules.builder import build_bbox_coder
from mmdet3d.models.utils import (filter_outside_objs, get_edge_indices,
                                  get_ellip_gaussian_2D, get_keypoints,
                                  handle_proj_objs)
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from .anchor_free_mono3d_head import AnchorFreeMono3DHead


@MODELS.register_module()
class MonoFlexHead(AnchorFreeMono3DHead):
    r"""MonoFlex head used in `MonoFlex <https://arxiv.org/abs/2104.02323>`_

    .. code-block:: none

                / --> 3 x 3 conv --> 1 x 1 conv --> [edge fusion] --> cls
                |
                | --> 3 x 3 conv --> 1 x 1 conv --> 2d bbox
                |
                | --> 3 x 3 conv --> 1 x 1 conv --> [edge fusion] --> 2d offsets
                |
                | --> 3 x 3 conv --> 1 x 1 conv -->  keypoints offsets
                |
                | --> 3 x 3 conv --> 1 x 1 conv -->  keypoints uncertainty
        feature
                | --> 3 x 3 conv --> 1 x 1 conv -->  keypoints uncertainty
                |
                | --> 3 x 3 conv --> 1 x 1 conv -->   3d dimensions
                |
                |                  |--- 1 x 1 conv -->  ori cls
                | --> 3 x 3 conv --|
                |                  |--- 1 x 1 conv -->  ori offsets
                |
                | --> 3 x 3 conv --> 1 x 1 conv -->  depth
                |
                \ --> 3 x 3 conv --> 1 x 1 conv -->  depth uncertainty

    Args:
        use_edge_fusion (bool): Whether to use edge fusion module while
            feature extraction.
        edge_fusion_inds (list[tuple]): Indices of feature to use edge fusion.
        edge_heatmap_ratio (float): Ratio of generating target heatmap.
        filter_outside_objs (bool, optional): Whether to filter the
            outside objects. Default: True.
        loss_cls (dict, optional): Config of classification loss.
            Default: loss_cls=dict(type='GaussionFocalLoss', loss_weight=1.0).
        loss_bbox (dict, optional): Config of localization loss.
            Default: loss_bbox=dict(type='IOULoss', loss_weight=10.0).
        loss_dir (dict, optional): Config of direction classification loss.
            Default: dict(type='MultibinLoss', loss_weight=0.1).
        loss_keypoints (dict, optional): Config of keypoints loss.
            Default: dict(type='L1Loss', loss_weight=0.1).
        loss_dims: (dict, optional): Config of dimensions loss.
            Default: dict(type='L1Loss', loss_weight=0.1).
        loss_offsets_2d: (dict, optional): Config of offsets_2d loss.
            Default: dict(type='L1Loss', loss_weight=0.1).
        loss_direct_depth: (dict, optional): Config of directly regression depth loss.
            Default: dict(type='L1Loss', loss_weight=0.1).
        loss_keypoints_depth: (dict, optional): Config of keypoints decoded depth loss.
            Default: dict(type='L1Loss', loss_weight=0.1).
        loss_combined_depth: (dict, optional): Config of combined depth loss.
            Default: dict(type='L1Loss', loss_weight=0.1).
        loss_attr (dict, optional): Config of attribute classification loss.
            In MonoFlex, Default: None.
        bbox_coder (dict, optional): Bbox coder for encoding and decoding boxes.
            Default: dict(type='MonoFlexCoder', code_size=7).
        norm_cfg (dict, optional): Dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        init_cfg (dict): Initialization config dict. Default: None.
    """  # noqa: E501

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 use_edge_fusion: bool,
                 edge_fusion_inds: List[Tuple],
                 edge_heatmap_ratio: float,
                 filter_outside_objs: bool = True,
                 loss_cls: dict = dict(
                     type='mmdet.GaussianFocalLoss', loss_weight=1.0),
                 loss_bbox: dict = dict(type='mmdet.IoULoss', loss_weight=0.1),
                 loss_dir: dict = dict(type='MultiBinLoss', loss_weight=0.1),
                 loss_keypoints: dict = dict(
                     type='mmdet.L1Loss', loss_weight=0.1),
                 loss_dims: dict = dict(type='mmdet.L1Loss', loss_weight=0.1),
                 loss_offsets_2d: dict = dict(
                     type='mmdet.L1Loss', loss_weight=0.1),
                 loss_direct_depth: dict = dict(
                     type='mmdet.L1Loss', loss_weight=0.1),
                 loss_keypoints_depth: dict = dict(
                     type='mmdet.L1Loss', loss_weight=0.1),
                 loss_combined_depth: dict = dict(
                     type='mmdet.L1Loss', loss_weight=0.1),
                 loss_attr: Optional[dict] = None,
                 bbox_coder: dict = dict(type='MonoFlexCoder', code_size=7),
                 norm_cfg: Union[ConfigDict, dict] = dict(type='BN'),
                 init_cfg: Optional[Union[ConfigDict, dict]] = None,
                 init_bias: float = -2.19,
                 **kwargs) -> None:
        self.use_edge_fusion = use_edge_fusion
        self.edge_fusion_inds = edge_fusion_inds
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
        self.filter_outside_objs = filter_outside_objs
        self.edge_heatmap_ratio = edge_heatmap_ratio
        self.init_bias = init_bias
        self.loss_dir = MODELS.build(loss_dir)
        self.loss_keypoints = MODELS.build(loss_keypoints)
        self.loss_dims = MODELS.build(loss_dims)
        self.loss_offsets_2d = MODELS.build(loss_offsets_2d)
        self.loss_direct_depth = MODELS.build(loss_direct_depth)
        self.loss_keypoints_depth = MODELS.build(loss_keypoints_depth)
        self.loss_combined_depth = MODELS.build(loss_combined_depth)
        self.bbox_coder = build_bbox_coder(bbox_coder)

    def _init_edge_module(self):
        """Initialize edge fusion module for feature extraction."""
        self.edge_fuse_cls = EdgeFusionModule(self.num_classes, 256)
        for i in range(len(self.edge_fusion_inds)):
            reg_inds, out_inds = self.edge_fusion_inds[i]
            out_channels = self.group_reg_dims[reg_inds][out_inds]
            fusion_layer = EdgeFusionModule(out_channels, 256)
            layer_name = f'edge_fuse_reg_{reg_inds}_{out_inds}'
            self.add_module(layer_name, fusion_layer)

    def init_weights(self):
        """Initialize weights."""
        super().init_weights()
        self.conv_cls.bias.data.fill_(self.init_bias)
        xavier_init(self.conv_regs[4][0], gain=0.01)
        xavier_init(self.conv_regs[7][0], gain=0.01)
        for m in self.conv_regs.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        self.conv_cls_prev = self._init_branch(
            conv_channels=self.cls_branch,
            conv_strides=(1, ) * len(self.cls_branch))
        self.conv_cls = nn.Conv2d(self.cls_branch[-1], self.cls_out_channels,
                                  1)
        # init regression head
        self.conv_reg_prevs = nn.ModuleList()
        # init output head
        self.conv_regs = nn.ModuleList()
        # group_reg_dims:
        # ((4, ), (2, ), (20, ), (3, ), (3, ), (8, 8), (1, ), (1, ))
        for i in range(len(self.group_reg_dims)):
            reg_dims = self.group_reg_dims[i]
            reg_branch_channels = self.reg_branch[i]
            out_channel = self.out_channels[i]
            reg_list = nn.ModuleList()
            if len(reg_branch_channels) > 0:
                self.conv_reg_prevs.append(
                    self._init_branch(
                        conv_channels=reg_branch_channels,
                        conv_strides=(1, ) * len(reg_branch_channels)))
                for reg_dim in reg_dims:
                    reg_list.append(nn.Conv2d(out_channel, reg_dim, 1))
                self.conv_regs.append(reg_list)
            else:
                self.conv_reg_prevs.append(None)
                for reg_dim in reg_dims:
                    reg_list.append(nn.Conv2d(self.feat_channels, reg_dim, 1))
                self.conv_regs.append(reg_list)

    def _init_layers(self):
        """Initialize layers of the head."""
        self._init_predictor()
        if self.use_edge_fusion:
            self._init_edge_module()

    def loss(self, x: List[Tensor], batch_data_samples: List[Det3DDataSample],
             **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            batch_data_samples (list[:obj:`Det3DDataSample`]): Each item
                contains the meta information of each image and corresponding
                annotations.
            proposal_cfg (mmengine.Config, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.

        Returns:
            tuple or Tensor: When `proposal_cfg` is None, the detector is a \
            normal one-stage detector, The return value is the losses.

            - losses: (dict[str, Tensor]): A dictionary of loss components.

            When the `proposal_cfg` is not None, the head is used as a
            `rpn_head`, the return value is a tuple contains:

            - losses: (dict[str, Tensor]): A dictionary of loss components.
            - results_list (list[:obj:`InstanceData`]): Detection
              results of each image after the post process.
              Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (:obj:`BaseInstance3DBoxes`): Contains a tensor
                  with shape (num_instances, C), the last dimension C of a
                  3D box is (x, y, z, x_size, y_size, z_size, yaw, ...), where
                  C >= 7. C = 7 for kitti and C = 9 for nuscenes with extra 2
                  dims of velocity.
        """

        batch_gt_instances_3d = []
        batch_gt_instances = []
        batch_gt_instances_ignore = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
            batch_gt_instances.append(data_sample.gt_instances)
            batch_gt_instances_ignore.append(
                data_sample.get('ignored_instances', None))

        # monoflex head needs img_metas for feature extraction
        outs = self(x, batch_img_metas)
        loss_inputs = outs + (batch_gt_instances_3d, batch_img_metas,
                              batch_gt_instances_ignore)
        losses = self.loss(*loss_inputs)

        return losses

    def forward(self, feats: List[Tensor], batch_img_metas: List[dict]):
        """Forward features from the upstream network.

        Args:
            feats (list[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level,
                    each is a 4D-tensor, the channel number is
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * bbox_code_size.
        """
        mlvl_batch_img_metas = [batch_img_metas for i in range(len(feats))]
        return multi_apply(self.forward_single, feats, mlvl_batch_img_metas)

    def forward_single(self, x: Tensor, batch_img_metas: List[dict]):
        """Forward features of a single scale level.

        Args:
            x (Tensor): Feature maps from a specific FPN feature level.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            tuple: Scores for each class, bbox predictions.
        """
        img_h, img_w = batch_img_metas[0]['pad_shape'][:2]
        batch_size, _, feat_h, feat_w = x.shape
        downsample_ratio = img_h / feat_h

        for conv_cls_prev_layer in self.conv_cls_prev:
            cls_feat = conv_cls_prev_layer(x)
        out_cls = self.conv_cls(cls_feat)

        if self.use_edge_fusion:
            # calculate the edge indices for the batch data
            edge_indices_list = get_edge_indices(
                batch_img_metas, downsample_ratio, device=x.device)
            edge_lens = [
                edge_indices.shape[0] for edge_indices in edge_indices_list
            ]
            max_edge_len = max(edge_lens)
            edge_indices = x.new_zeros((batch_size, max_edge_len, 2),
                                       dtype=torch.long)
            for i in range(batch_size):
                edge_indices[i, :edge_lens[i]] = edge_indices_list[i]
            # cls feature map edge fusion
            out_cls = self.edge_fuse_cls(cls_feat, out_cls, edge_indices,
                                         edge_lens, feat_h, feat_w)

        bbox_pred = []

        for i in range(len(self.group_reg_dims)):
            reg_feat = x.clone()
            # feature regression head
            if len(self.reg_branch[i]) > 0:
                for conv_reg_prev_layer in self.conv_reg_prevs[i]:
                    reg_feat = conv_reg_prev_layer(reg_feat)

            for j, conv_reg in enumerate(self.conv_regs[i]):
                out_reg = conv_reg(reg_feat)
                #  Use Edge Fusion Module
                if self.use_edge_fusion and (i, j) in self.edge_fusion_inds:
                    # reg feature map edge fusion
                    out_reg = getattr(self, 'edge_fuse_reg_{}_{}'.format(
                        i, j))(reg_feat, out_reg, edge_indices, edge_lens,
                               feat_h, feat_w)
                bbox_pred.append(out_reg)

        bbox_pred = torch.cat(bbox_pred, dim=1)
        cls_score = out_cls.sigmoid()  # turn to 0-1
        cls_score = cls_score.clamp(min=1e-4, max=1 - 1e-4)

        return cls_score, bbox_pred

    def predict_by_feat(self, cls_scores: List[Tensor],
                        bbox_preds: List[Tensor], batch_img_metas: List[dict]):
        """Generate bboxes from bbox head predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level.
            bbox_preds (list[Tensor]): Box regression for each scale.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
        Returns:
            list[tuple[:obj:`CameraInstance3DBoxes`, Tensor, Tensor, None]]:
                Each item in result_list is 4-tuple.
        """
        assert len(cls_scores) == len(bbox_preds) == 1
        cam2imgs = torch.stack([
            cls_scores[0].new_tensor(input_meta['cam2img'])
            for input_meta in batch_img_metas
        ])
        batch_bboxes, batch_scores, batch_topk_labels = self._decode_heatmap(
            cls_scores[0],
            bbox_preds[0],
            batch_img_metas,
            cam2imgs=cam2imgs,
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
            results.scores_3d = scores
            results.labels_3d = labels

            if attrs is not None:
                results.attr_labels = attrs

            result_list.append(results)

        return result_list

    def _decode_heatmap(self,
                        cls_score: Tensor,
                        reg_pred: Tensor,
                        batch_img_metas: List[dict],
                        cam2imgs: Tensor,
                        topk: int = 100,
                        kernel: int = 3):
        """Transform outputs into detections raw bbox predictions.

        Args:
            class_score (Tensor): Center predict heatmap,
                shape (B, num_classes, H, W).
            reg_pred (Tensor): Box regression map.
                shape (B, channel, H , W).
            batch_img_metas (List[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cam2imgs (Tensor): Camera intrinsic matrix.
                shape (N, 4, 4)
            topk (int, optional): Get top k center keypoints from heatmap.
                Default 100.
            kernel (int, optional): Max pooling kernel for extract local
                maximum pixels. Default 3.

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
        batch_size, _, feat_h, feat_w = cls_score.shape

        downsample_ratio = img_h / feat_h
        center_heatmap_pred = get_local_maximum(cls_score, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
            center_heatmap_pred, k=topk)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        regression = transpose_and_gather_feat(reg_pred, batch_index)
        regression = regression.view(-1, 8)

        pred_base_centers_2d = torch.cat(
            [topk_xs.view(-1, 1),
             topk_ys.view(-1, 1).float()], dim=1)
        preds = self.bbox_coder.decode(regression, batch_topk_labels,
                                       downsample_ratio, cam2imgs)
        pred_locations = self.bbox_coder.decode_location(
            pred_base_centers_2d, preds['offsets_2d'], preds['combined_depth'],
            cam2imgs, downsample_ratio)
        pred_yaws = self.bbox_coder.decode_orientation(
            preds['orientations']).unsqueeze(-1)
        pred_dims = preds['dimensions']
        batch_bboxes = torch.cat((pred_locations, pred_dims, pred_yaws), dim=1)
        batch_bboxes = batch_bboxes.view(batch_size, -1, self.bbox_code_size)
        return batch_bboxes, batch_scores, batch_topk_labels

    def get_predictions(self, pred_reg, labels3d, centers_2d, reg_mask,
                        batch_indices, batch_img_metas, downsample_ratio):
        """Prepare predictions for computing loss.

        Args:
            pred_reg (Tensor): Box regression map.
                shape (B, channel, H , W).
            labels3d (Tensor): Labels of each 3D box.
                shape (B * max_objs, )
            centers_2d (Tensor): Coords of each projected 3D box
                center on image. shape (N, 2)
            reg_mask (Tensor): Indexes of the existence of the 3D box.
                shape (B * max_objs, )
            batch_indices (Tenosr): Batch indices of the 3D box.
                shape (N, 3)
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            downsample_ratio (int): The stride of feature map.

        Returns:
            dict: The predictions for computing loss.
        """
        batch, channel = pred_reg.shape[0], pred_reg.shape[1]
        w = pred_reg.shape[3]
        cam2imgs = torch.stack([
            centers_2d.new_tensor(img_meta['cam2img'])
            for img_meta in batch_img_metas
        ])
        # (batch_size, 4, 4) -> (N, 4, 4)
        cam2imgs = cam2imgs[batch_indices, :, :]
        centers_2d_inds = centers_2d[:, 1] * w + centers_2d[:, 0]
        centers_2d_inds = centers_2d_inds.view(batch, -1)
        pred_regression = transpose_and_gather_feat(pred_reg, centers_2d_inds)
        pred_regression_pois = pred_regression.view(-1, channel)[reg_mask]
        preds = self.bbox_coder.decode(pred_regression_pois, labels3d,
                                       downsample_ratio, cam2imgs)

        return preds

    def get_targets(self, batch_gt_instances_3d: List[InstanceData],
                    batch_gt_instances: List[InstanceData],
                    feat_shape: Tuple[int], batch_img_metas: List[dict]):
        """Get training targets for batch images.
``
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
            tuple[Tensor, dict]: The Tensor value is the targets of
                center heatmap, the dict has components below:
              - base_centers_2d_target (Tensor): Coords of each projected
                    3D box center on image. shape (B * max_objs, 2),
                    [dtype: int]
              - labels3d (Tensor): Labels of each 3D box.
                    shape (N, )
              - reg_mask (Tensor): Mask of the existence of the 3D box.
                    shape (B * max_objs, )
              - batch_indices (Tensor): Batch id of the 3D box.
                    shape (N, )
              - depth_target (Tensor): Depth target of each 3D box.
                    shape (N, )
              - keypoints2d_target (Tensor): Keypoints of each projected 3D box
                    on image. shape (N, 10, 2)
              - keypoints_mask (Tensor): Keypoints mask of each projected 3D
                    box on image. shape (N, 10)
              - keypoints_depth_mask (Tensor): Depths decoded from keypoints
                    of each 3D box. shape (N, 3)
              - orientations_target (Tensor): Orientation (encoded local yaw)
                    target of each 3D box. shape (N, )
              - offsets_2d_target (Tensor): Offsets target of each projected
                    3D box. shape (N, 2)
              - dimensions_target (Tensor): Dimensions target of each 3D box.
                    shape (N, 3)
              - downsample_ratio (int): The stride of feature map.
        """

        gt_bboxes_list = [
            gt_instances.bboxes for gt_instances in batch_gt_instances
        ]
        gt_labels_list = [
            gt_instances.labels for gt_instances in batch_gt_instances
        ]
        gt_bboxes_3d_list = [
            gt_instances_3d.bboxes_3d
            for gt_instances_3d in batch_gt_instances_3d
        ]
        gt_labels_3d_list = [
            gt_instances_3d.labels_3d
            for gt_instances_3d in batch_gt_instances_3d
        ]
        centers_2d_list = [
            gt_instances_3d.centers_2d
            for gt_instances_3d in batch_gt_instances_3d
        ]
        depths_list = [
            gt_instances_3d.depths for gt_instances_3d in batch_gt_instances_3d
        ]

        img_h, img_w = batch_img_metas[0]['pad_shape'][:2]
        batch_size, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)  # 1/4
        height_ratio = float(feat_h / img_h)  # 1/4

        assert width_ratio == height_ratio

        # Whether to filter the objects which are not in FOV.
        if self.filter_outside_objs:
            filter_outside_objs(gt_bboxes_list, gt_labels_list,
                                gt_bboxes_3d_list, gt_labels_3d_list,
                                centers_2d_list, batch_img_metas)

        # transform centers_2d to base centers_2d for regression and
        # heatmap generation.
        # centers_2d = int(base_centers_2d) + offsets_2d
        base_centers_2d_list, offsets_2d_list, trunc_mask_list = \
            handle_proj_objs(centers_2d_list, gt_bboxes_list, batch_img_metas)

        keypoints2d_list, keypoints_mask_list, keypoints_depth_mask_list = \
            get_keypoints(gt_bboxes_3d_list, centers_2d_list, batch_img_metas)

        center_heatmap_target = gt_bboxes_list[-1].new_zeros(
            [batch_size, self.num_classes, feat_h, feat_w])

        for batch_id in range(batch_size):
            # project gt_bboxes from input image to feat map
            gt_bboxes = gt_bboxes_list[batch_id] * width_ratio
            gt_labels = gt_labels_list[batch_id]

            # project base centers_2d from input image to feat map
            gt_base_centers_2d = base_centers_2d_list[batch_id] * width_ratio
            trunc_masks = trunc_mask_list[batch_id]

            for j, base_center2d in enumerate(gt_base_centers_2d):
                if trunc_masks[j]:
                    # for outside objects, generate ellipse heatmap
                    base_center2d_x_int, base_center2d_y_int = \
                        base_center2d.int()
                    scale_box_w = min(base_center2d_x_int - gt_bboxes[j][0],
                                      gt_bboxes[j][2] - base_center2d_x_int)
                    scale_box_h = min(base_center2d_y_int - gt_bboxes[j][1],
                                      gt_bboxes[j][3] - base_center2d_y_int)
                    radius_x = scale_box_w * self.edge_heatmap_ratio
                    radius_y = scale_box_h * self.edge_heatmap_ratio
                    radius_x, radius_y = max(0, int(radius_x)), max(
                        0, int(radius_y))
                    assert min(radius_x, radius_y) == 0
                    ind = gt_labels[j]
                    get_ellip_gaussian_2D(
                        center_heatmap_target[batch_id, ind],
                        [base_center2d_x_int, base_center2d_y_int], radius_x,
                        radius_y)
                else:
                    base_center2d_x_int, base_center2d_y_int = \
                        base_center2d.int()
                    scale_box_h = (gt_bboxes[j][3] - gt_bboxes[j][1])
                    scale_box_w = (gt_bboxes[j][2] - gt_bboxes[j][0])
                    radius = gaussian_radius([scale_box_h, scale_box_w],
                                             min_overlap=0.7)
                    radius = max(0, int(radius))
                    ind = gt_labels[j]
                    gen_gaussian_target(
                        center_heatmap_target[batch_id, ind],
                        [base_center2d_x_int, base_center2d_y_int], radius)

        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        num_ctrs = [centers_2d.shape[0] for centers_2d in centers_2d_list]
        max_objs = max(num_ctrs)
        batch_indices = [
            centers_2d_list[0].new_full((num_ctrs[i], ), i)
            for i in range(batch_size)
        ]
        batch_indices = torch.cat(batch_indices, dim=0)
        reg_mask = torch.zeros(
            (batch_size, max_objs),
            dtype=torch.bool).to(base_centers_2d_list[0].device)
        gt_bboxes_3d = batch_img_metas[0]['box_type_3d'].cat(gt_bboxes_3d_list)
        gt_bboxes_3d = gt_bboxes_3d.to(base_centers_2d_list[0].device)

        # encode original local yaw to multibin format
        orienations_target = self.bbox_coder.encode(gt_bboxes_3d)

        batch_base_centers_2d = base_centers_2d_list[0].new_zeros(
            (batch_size, max_objs, 2))

        for i in range(batch_size):
            reg_mask[i, :num_ctrs[i]] = 1
            batch_base_centers_2d[i, :num_ctrs[i]] = base_centers_2d_list[i]

        flatten_reg_mask = reg_mask.flatten()

        # transform base centers_2d from input scale to output scale
        batch_base_centers_2d = batch_base_centers_2d.view(-1, 2) * width_ratio

        dimensions_target = gt_bboxes_3d.tensor[:, 3:6]
        labels_3d = torch.cat(gt_labels_3d_list)
        keypoints2d_target = torch.cat(keypoints2d_list)
        keypoints_mask = torch.cat(keypoints_mask_list)
        keypoints_depth_mask = torch.cat(keypoints_depth_mask_list)
        offsets_2d_target = torch.cat(offsets_2d_list)
        bboxes2d = torch.cat(gt_bboxes_list)

        # transform FCOS style bbox into [x1, y1, x2, y2] format.
        bboxes2d_target = torch.cat([bboxes2d[:, 0:2] * -1, bboxes2d[:, 2:]],
                                    dim=-1)
        depths = torch.cat(depths_list)

        target_labels = dict(
            base_centers_2d_target=batch_base_centers_2d.int(),
            labels3d=labels_3d,
            reg_mask=flatten_reg_mask,
            batch_indices=batch_indices,
            bboxes2d_target=bboxes2d_target,
            depth_target=depths,
            keypoints2d_target=keypoints2d_target,
            keypoints_mask=keypoints_mask,
            keypoints_depth_mask=keypoints_depth_mask,
            orienations_target=orienations_target,
            offsets_2d_target=offsets_2d_target,
            dimensions_target=dimensions_target,
            downsample_ratio=1 / width_ratio)

        return center_heatmap_target, avg_factor, target_labels

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            batch_gt_instances_3d: List[InstanceData],
            batch_gt_instances: List[InstanceData],
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: Optional[List[InstanceData]] = None):
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
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == 1
        assert batch_gt_instances_ignore is None
        center2d_heatmap = cls_scores[0]
        pred_reg = bbox_preds[0]

        center2d_heatmap_target, avg_factor, target_labels = \
            self.get_targets(batch_gt_instances_3d,
                             batch_gt_instances,
                             center2d_heatmap.shape,
                             batch_img_metas)

        preds = self.get_predictions(
            pred_reg=pred_reg,
            labels3d=target_labels['labels3d'],
            centers_2d=target_labels['base_centers_2d_target'],
            reg_mask=target_labels['reg_mask'],
            batch_indices=target_labels['batch_indices'],
            batch_img_metas=batch_img_metas,
            downsample_ratio=target_labels['downsample_ratio'])

        # heatmap loss
        loss_cls = self.loss_cls(
            center2d_heatmap, center2d_heatmap_target, avg_factor=avg_factor)

        # bbox2d regression loss
        loss_bbox = self.loss_bbox(preds['bboxes2d'],
                                   target_labels['bboxes2d_target'])

        # keypoints loss, the keypoints in predictions and target are all
        # local coordinates. Check the mask dtype should be bool, not int
        # or float to ensure the indexing is bool index
        keypoints2d_mask = target_labels['keypoints2d_mask']
        loss_keypoints = self.loss_keypoints(
            preds['keypoints2d'][keypoints2d_mask],
            target_labels['keypoints2d_target'][keypoints2d_mask])

        # orientations loss
        loss_dir = self.loss_dir(preds['orientations'],
                                 target_labels['orientations_target'])

        # dimensions loss
        loss_dims = self.loss_dims(preds['dimensions'],
                                   target_labels['dimensions_target'])

        # offsets for center heatmap
        loss_offsets_2d = self.loss_offsets_2d(
            preds['offsets_2d'], target_labels['offsets_2d_target'])

        # directly regressed depth loss with direct depth uncertainty loss
        direct_depth_weights = torch.exp(-preds['direct_depth_uncertainty'])
        loss_weight_1 = self.loss_direct_depth.loss_weight
        loss_direct_depth = self.loss_direct_depth(
            preds['direct_depth'], target_labels['depth_target'],
            direct_depth_weights)
        loss_uncertainty_1 =\
            preds['direct_depth_uncertainty'] * loss_weight_1
        loss_direct_depth = loss_direct_depth + loss_uncertainty_1.mean()

        # keypoints decoded depth loss with keypoints depth uncertainty loss
        depth_mask = target_labels['keypoints_depth_mask']
        depth_target = target_labels['depth_target'].unsqueeze(-1).repeat(1, 3)
        valid_keypoints_depth_uncertainty = preds[
            'keypoints_depth_uncertainty'][depth_mask]
        valid_keypoints_depth_weights = torch.exp(
            -valid_keypoints_depth_uncertainty)
        loss_keypoints_depth = self.loss_keypoint_depth(
            preds['keypoints_depth'][depth_mask], depth_target[depth_mask],
            valid_keypoints_depth_weights)
        loss_weight_2 = self.loss_keypoints_depth.loss_weight
        loss_uncertainty_2 =\
            valid_keypoints_depth_uncertainty * loss_weight_2
        loss_keypoints_depth = loss_keypoints_depth + loss_uncertainty_2.mean()

        # combined depth loss for optimiaze the uncertainty
        loss_combined_depth = self.loss_combined_depth(
            preds['combined_depth'], target_labels['depth_target'])

        loss_dict = dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_keypoints=loss_keypoints,
            loss_dir=loss_dir,
            loss_dims=loss_dims,
            loss_offsets_2d=loss_offsets_2d,
            loss_direct_depth=loss_direct_depth,
            loss_keypoints_depth=loss_keypoints_depth,
            loss_combined_depth=loss_combined_depth)

        return loss_dict
