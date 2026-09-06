# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates.
# All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FocalFormer3D/blob/main/LICENSE

# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FocalFormer3D/blob/main/LICENSE

import numpy as np
import torch
from mmcv.ops import Voxelization
from mmengine.structures import InstanceData
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import LiDARInstance3DBoxes, bbox3d2result
from ...core.post_processing.merge_augs import merge_aug_bboxes_3d


# ===================================================================
# FocalFormer3D Detector
# ===================================================================
@MODELS.register_module()
class FocalFormer3D(MVXTwoStageDetector):
    """FocalFormer3D: Focusing on Hard Instance for 3D Object Detection.

    Refactored for mmdet3d >= 1.1 (mmengine-based).

    Key changes from old version:
      - ``forward_train()`` → ``loss()``
      - ``simple_test()`` → ``predict()``
      - Data comes via ``batch_inputs_dict`` and ``batch_data_samples``
      - ``@force_fp32`` removed (use AmpOptimWrapper)
      - ``builder.build_neck`` → ``MODELS.build``
    """

    def __init__(self,
                 freeze_img=False,
                 freeze_img_level=None,
                 freeze_camlss=False,
                 freeze_pts=False,
                 trainneck_ms=False,
                 train_middle_encoder=False,
                 pts_pillar_layer=None,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 imgpts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 input_img=True,
                 use_grid_mask=False,
                 input_pts=True,
                 init_cfg=None,
                 data_preprocessor=None,
                 **kwargs):
        # NOTE: In mmdet3d 1.4, MVXTwoStageDetector no longer accepts
        # pts_voxel_layer as an __init__ kwarg (it's handled by
        # data_preprocessor in the standard pipeline). Since FocalFormer3D
        # does its own voxelization, we intercept pts_voxel_layer here
        # and build it ourselves, then call super() without it.
        super(FocalFormer3D, self).__init__(
            pts_voxel_encoder=pts_voxel_encoder,
            pts_middle_encoder=pts_middle_encoder,
            pts_fusion_layer=pts_fusion_layer,
            img_backbone=img_backbone,
            pts_backbone=pts_backbone,
            img_neck=img_neck,
            pts_neck=pts_neck,
            pts_bbox_head=pts_bbox_head,
            img_roi_head=img_roi_head,
            img_rpn_head=img_rpn_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor,
            **kwargs)

        # Build voxel layers ourselves (removed from super() call)
        if pts_voxel_layer:
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer)

        if pts_pillar_layer:
            self.pts_pillar_layer = Voxelization(**pts_pillar_layer)

        self.freeze_img_level = freeze_img_level
        self.freeze_camlss = freeze_camlss

        # --- builder.build_neck → MODELS.build ---
        self.imgpts_neck = MODELS.build(imgpts_neck)

        self.freeze_img = freeze_img
        self.freeze_pts = freeze_pts
        self.trainneck_ms = trainneck_ms
        self.train_middle_encoder = train_middle_encoder

        self.input_img = input_img
        self.input_pts = input_pts

        self.use_grid_mask = use_grid_mask
        if self.use_grid_mask:
            from ..utils.grid_mask import GridMask
            self.grid_mask = GridMask(
                True,
                True,
                rotate=1,
                offset=False,
                ratio=0.5,
                mode=1,
                prob=0.7)

        self.apply_dynamic_voxelize = 'Dynamic' in pts_voxel_encoder['type']

    def init_weights(self):
        """Initialize model weights."""
        super(FocalFormer3D, self).init_weights()
        if self.input_img and self.freeze_img:
            if self.with_img_backbone:
                if self.freeze_img_level:
                    param_levels = [['conv1', 'bn1'], ['layer1'], ['layer2'],
                                    ['layer3'], ['layer4']]
                    for i in range(self.freeze_img_level):
                        for pn in param_levels[i]:
                            for param in self.img_backbone.get_submodule(
                                    pn).parameters():
                                param.requires_grad = False
                else:
                    for param in self.img_backbone.parameters():
                        param.requires_grad = False

            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False

            if self.freeze_camlss and hasattr(self.imgpts_neck, 'cam_lss'):
                for param in self.imgpts_neck.cam_lss.parameters():
                    param.requires_grad = False

        if self.freeze_pts:
            for name, param in self.named_parameters():
                if ('pts' in name and 'pts_bbox_head' not in name
                        and 'imgpts_neck' not in name):
                    if self.trainneck_ms:
                        if 'pts_backbone' in name:
                            continue
                        if 'pts_neck' in name:
                            continue
                    if self.train_middle_encoder:
                        if 'pts' in name:
                            continue
                    param.requires_grad = False

            def fix_bn(m):
                if isinstance(m, nn.BatchNorm1d) or isinstance(
                        m, nn.BatchNorm2d):
                    m.track_running_stats = False

            if not self.train_middle_encoder:
                self.pts_voxel_layer.apply(fix_bn)
                self.pts_voxel_encoder.apply(fix_bn)
                self.pts_middle_encoder.apply(fix_bn)

            if not self.trainneck_ms:
                self.pts_backbone.apply(fix_bn)
                if self.with_pts_neck:
                    self.pts_neck.apply(fix_bn)

        if not self.input_pts:
            self.voxelize = None
            self.pts_voxel_encoder = None
            self.pts_middle_encoder = None
            self.pts_backbone = None
            if self.with_pts_neck:
                self.pts_neck = None

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)

            if self.use_grid_mask and self.training:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img.float())
        else:
            return None

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        return img_feats

    def extract_pts_feat(self, pts, img_feats=None, img_metas=None):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None

        if self.apply_dynamic_voxelize:
            voxels, coors = self.dynamic_voxelize(pts)
            voxel_features, feature_coors = self.pts_voxel_encoder(
                voxels, coors)
            batch_size = coors[-1, 0] + 1
            coors = feature_coors  # update
        else:
            voxels, num_points, coors = self.voxelize(pts, voxel_type='voxel')
            voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
            batch_size = coors[-1, 0] + 1

        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)

        if self.with_pts_neck:
            x = self.pts_neck(x)
        else:
            x = [x]

        return x

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points.

        This is the shared feature extraction used by both loss() and
        predict().
        """
        if self.input_img:
            img_feats = self.extract_img_feat(img, img_metas)
        else:
            img_feats = [None]
        if self.input_pts:
            pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        else:
            pts_feats = [None]
        new_img_feat, new_pts_feat = self.imgpts_neck(img_feats[0],
                                                      pts_feats[0], img_metas)

        return (new_img_feat, new_pts_feat)

    @torch.no_grad()
    # NOTE: @force_fp32() REMOVED — handled by AmpOptimWrapper
    def voxelize(self, points, voxel_type='voxel'):
        """Apply voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            voxel_type (str): 'voxel' or 'pillar'.

        Returns:
            tuple: voxels, num_points, coors_batch
        """
        assert voxel_type == 'voxel' or voxel_type == 'pillar'
        voxels, coors, num_points = [], [], []

        for res in points:
            if voxel_type == 'voxel':
                res_voxels, res_coors, res_num_points = self.pts_voxel_layer(
                    res)
            elif voxel_type == 'pillar':
                res_voxels, res_coors, res_num_points = self.pts_pillar_layer(
                    res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)

        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []

        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)

        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    @torch.no_grad()
    # NOTE: @force_fp32() REMOVED — handled by AmpOptimWrapper
    def dynamic_voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provides a coors mapping
        for res in points:
            res_coors = self.pts_voxel_layer(res)
            coors.append(res_coors)

        points = torch.cat(points, dim=0)
        coors_batch = []

        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)

        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch

    @staticmethod
    def _dataset_name(cfg):
        """Read ``cfg.pts.dataset``, tolerating a missing or None config.

        Every shipped config sets ``pts.dataset`` on both ``train_cfg`` and
        ``test_cfg``, so each side of the box-convention conversion can read
        whichever config is in scope for that call.
        """
        if cfg is None:
            return ''
        return cfg.get('pts', {}).get('dataset', '')

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs and data samples."""
        points = batch_inputs_dict.get('points', None)
        imgs = batch_inputs_dict.get('imgs', None)

        batch_gt_instances_3d = []
        img_metas = []

        # The released checkpoints were trained under the mmdet3d v0.x box
        # convention, whose nuScenes converter wrote dimensions as (w, l, h);
        # v1.x writes (l, w, h). The head regresses whatever column order it
        # is trained on, so put the GT back into the v0.x order before it
        # reaches the head and undo it on the predictions in
        # add_pred_to_datasample.
        #
        _is_nuscenes = self._dataset_name(self.train_cfg) == 'nuScenes'

        for data_sample in batch_data_samples:
            bboxes = data_sample.gt_instances_3d.bboxes_3d
            if bboxes is not None and len(bboxes) > 0 and _is_nuscenes:
                tensor = bboxes.tensor
                l_v1 = tensor[:, 3].clone()
                w_v1 = tensor[:, 4].clone()
                tensor[:, 3] = w_v1
                tensor[:, 4] = l_v1
                # yaw_v0 = -yaw_v1 - pi/2. The map is its own inverse, so this
                # is the same expression add_pred_to_datasample uses to go the
                # other way, and the two compose to the identity.
                tensor[:, 6] = -tensor[:, 6] - (np.pi / 2)

            img_metas.append(data_sample.metainfo)

            gt_instances = InstanceData()
            gt_instances.bboxes_3d = bboxes
            gt_instances.labels_3d = data_sample.gt_instances_3d.labels_3d
            batch_gt_instances_3d.append(gt_instances)

        img_feats, pts_feats = self.extract_feat(
            points, img=imgs, img_metas=img_metas)

        losses = dict()
        losses_pts = self.forward_pts_train(pts_feats, img_feats,
                                            batch_gt_instances_3d, img_metas)
        losses.update(losses_pts)

        return losses

    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          batch_gt_instances_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for the point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Point cloud BEV features.
            img_feats (torch.Tensor or None): Image features.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Ground truth
                per sample, carrying ``bboxes_3d`` and ``labels_3d``.
            img_metas (list[dict]): Meta information.
            gt_bboxes_ignore (None): Unused, kept for signature compatibility.

        Returns:
            dict[str, torch.Tensor]: Losses of the point cloud branch.
        """
        # FocalDecoder keeps the v0.x head signature: its forward and its loss
        # both take the boxes and the labels as two parallel lists rather than
        # one list of InstanceData, so unpack the batch once here.
        gt_bboxes_3d = [inst.bboxes_3d for inst in batch_gt_instances_3d]
        gt_labels_3d = [inst.labels_3d for inst in batch_gt_instances_3d]

        outs = self.pts_bbox_head(
            pts_feats,
            img_feats,
            img_metas,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d)

        return self.pts_bbox_head.loss(gt_bboxes_3d, gt_labels_3d, outs)

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples.

        This replaces the old ``simple_test()`` method.

        Args:
            batch_inputs_dict (dict): Contains 'points', 'imgs', etc.
            batch_data_samples (list[:obj:`Det3DDataSample`]): Each
                contains metainfo (the old img_metas).

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results with
                pred_instances_3d populated.
        """
        points = batch_inputs_dict.get('points', None)
        imgs = batch_inputs_dict.get('imgs', None)
        img_metas = [ds.metainfo for ds in batch_data_samples]

        gt_bboxes_3d = None
        gt_labels_3d = None
        if hasattr(batch_data_samples[0], 'gt_instances_3d'):
            try:
                gt_bboxes_3d = [
                    ds.gt_instances_3d.bboxes_3d for ds in batch_data_samples
                ]
                gt_labels_3d = [
                    ds.gt_instances_3d.labels_3d for ds in batch_data_samples
                ]
            except AttributeError:
                pass

        # --- Feature extraction ---
        img_feats, pts_feats = self.extract_feat(
            points, img=imgs, img_metas=img_metas)

        # --- Point cloud branch inference ---
        bbox_pts = self.simple_test_pts(
            pts_feats,
            img_feats,
            img_metas,
            rescale=kwargs.get('rescale', False),
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d)

        # --- Package results into Det3DDataSample ---
        results_list = self.add_pred_to_datasample(batch_data_samples,
                                                   bbox_pts)

        return results_list

    def simple_test_pts(self,
                        x,
                        x_img,
                        img_metas,
                        rescale=False,
                        gt_bboxes_3d=None,
                        gt_labels_3d=None,
                        **kwargs):
        """Test function of point cloud branch.

        Args:
            x: Point cloud features.
            x_img: Image features.
            img_metas (list[dict]): Meta information.
            rescale (bool): Whether to rescale results.
            gt_bboxes_3d: Optional GT boxes for debug/oracle testing.
            gt_labels_3d: Optional GT labels for debug/oracle testing.

        Returns:
            list[dict]: Detection results per sample.
        """
        outs = self.pts_bbox_head(
            x,
            x_img,
            img_metas,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            **kwargs)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        return bbox_results

    def aug_test(self,
                 batch_inputs_dict_list,
                 batch_data_samples_list,
                 rescale=False):
        """Test function with augmentation.

        In mmdet3d 1.1+, aug_test is called differently than in 1.0.
        This method handles both patterns.

        Args:
            batch_inputs_dict_list: List of input dicts per augmentation.
            batch_data_samples_list: List of data samples per augmentation.
            rescale (bool): Whether to rescale results.

        Returns:
            list[dict]: Merged detection results.
        """
        precompute = False

        if not precompute:

            # Extract points and images for each augmentation
            all_pts_feats = []
            all_img_feats = []
            all_img_metas = []

            for batch_inputs, batch_samples in zip(batch_inputs_dict_list,
                                                   batch_data_samples_list):
                points = batch_inputs.get('points', None)
                imgs = batch_inputs.get('imgs', None)
                img_metas = [ds.metainfo for ds in batch_samples]

                img_feats, pts_feats = self.extract_feat(
                    points, img=imgs, img_metas=img_metas)
                all_pts_feats.append(pts_feats)
                all_img_feats.append(img_feats)
                all_img_metas.append(img_metas)

            bbox_list = dict()
            if self.with_pts_bbox:
                bbox_pts = self.aug_test_pts(
                    all_pts_feats,
                    all_img_feats,
                    all_img_metas,
                    rescale=rescale)
                bbox_list.update(pts_bbox=bbox_pts)
        else:
            all_img_metas = [[ds.metainfo for ds in batch_samples]
                             for batch_samples in batch_data_samples_list]
            bbox_list = dict()
            bbox_pts = self.aug_test_pts(
                None, None, all_img_metas, rescale=rescale)
            bbox_list.update(pts_bbox=bbox_pts)

        return [bbox_list]

    def aug_test_pts(self, xs, x_imgs, img_metas, rescale=False):
        """Test function of point cloud branch with augmentation.

        Args:
            xs (list): Point features per augmentation, or None if
                precomputed.
            x_imgs (list): Image features per augmentation (or None).
            img_metas (list[list[dict]]): Meta info per augmentation.
            rescale (bool): Whether to rescale results.

        Returns:
            dict: Merged bounding box results.
        """
        if xs is not None:
            # only support aug_test for one sample
            aug_bboxes = []
            for x, x_img, img_meta in zip(xs, x_imgs, img_metas):
                outs = self.pts_bbox_head(x, x_img, img_meta)
                bbox_list = self.pts_bbox_head.get_bboxes(
                    outs, img_meta, rescale=rescale)
                bbox_list = [
                    dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                    for bboxes, scores, labels in bbox_list
                ]
                aug_bboxes.append(bbox_list[0])

            # after merging, bboxes will be rescaled to the original image size
            merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                                self.pts_bbox_head.test_cfg)
        else:
            merged_bboxes = merge_aug_bboxes_3d(None, img_metas,
                                                self.pts_bbox_head.test_cfg)

        return merged_bboxes

    def add_pred_to_datasample(self, batch_data_samples, bbox_results):
        """Add prediction results to Det3DDataSample."""
        # =====================================================================
        # Both NuScenes and Waymo checkpoints were trained with the v0.x
        # convention where the head outputs (w, l) and yaw in the old sense.
        # mmdet3d v1.x evaluators expect (l, w) and yaw = -yaw_old - pi/2.
        # Apply the same fix for both datasets.
        _needs_fix = self._dataset_name(self.test_cfg) in ('nuScenes', 'Waymo')

        for result in bbox_results:
            bboxes_3d = result.get(
                'bboxes_3d', result.get('boxes_3d', result.get('bboxes',
                                                               None)))
            if bboxes_3d is not None and len(bboxes_3d) > 0:
                tensor = bboxes_3d.tensor
                if _needs_fix:
                    # swap predicted (w, l) -> (l, w) to match v1.x evaluator
                    w_v0 = tensor[:, 3].clone()
                    l_v0 = tensor[:, 4].clone()
                    tensor[:, 3] = l_v0  # index 3 = length
                    tensor[:, 4] = w_v0  # index 4 = width
                    # yaw convention: v0 head outputs sin/cos of old yaw,
                    # v1.x evaluator expects yaw_new = -yaw_old - pi/2
                    tensor[:, 6] = -tensor[:, 6] - (np.pi / 2)
        # =====================================================================

        for data_sample, bbox_result in zip(batch_data_samples, bbox_results):
            pred_instances_3d = InstanceData()

            # Now we extract the mathematically corrected boxes
            bboxes_3d = bbox_result.get('bboxes_3d',
                                        bbox_result.get(
                                            'boxes_3d',
                                            bbox_result.get('bboxes',
                                                            None)))  # 2
            scores_3d = bbox_result.get('scores_3d',
                                        bbox_result.get('scores', None))
            labels_3d = bbox_result.get('labels_3d',
                                        bbox_result.get('labels', None))

            if bboxes_3d is not None and hasattr(
                    bboxes_3d, '__len__') and len(bboxes_3d) > 0:
                labels_3d = labels_3d.long()
                # Drop out-of-range labels. num_classes comes from the
                # dataset rather than a hardcoded 10, which would silently
                # admit spurious high labels on Waymo.
                num_classes = (3 if self._dataset_name(self.test_cfg)
                               == 'Waymo' else 10)
                valid_mask = (labels_3d >= 0) & (labels_3d < num_classes)
                if not valid_mask.all():
                    valid_idx = valid_mask.nonzero(as_tuple=True)[0]
                    bboxes_3d = bboxes_3d[valid_idx]
                    scores_3d = scores_3d[valid_idx]
                    labels_3d = labels_3d[valid_idx]
                pred_instances_3d.bboxes_3d = bboxes_3d
                pred_instances_3d.scores_3d = scores_3d
                pred_instances_3d.labels_3d = labels_3d
            else:
                # Empty prediction fallback
                pred_instances_3d.bboxes_3d = LiDARInstance3DBoxes(
                    torch.zeros((0, 9), dtype=torch.float32))
                pred_instances_3d.scores_3d = torch.zeros((0, ),
                                                          dtype=torch.float32)
                pred_instances_3d.labels_3d = torch.zeros((0, ),
                                                          dtype=torch.int64)

            data_sample.pred_instances_3d = pred_instances_3d

            # Add empty 2D predictions if not present (required by evaluator)
            if not hasattr(
                    data_sample,
                    'pred_instances') or data_sample.pred_instances is None:
                data_sample.pred_instances = InstanceData()

        return batch_data_samples
