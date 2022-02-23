# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from mmseg.core import add_prefix
from mmseg.models import SEGMENTORS
from ..builder import build_backbone, build_head, build_loss, build_neck
from .base import Base3DSegmentor


@SEGMENTORS.register_module()
class EncoderDecoder3D(Base3DSegmentor):
    """3D Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be thrown during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 loss_regularization=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(EncoderDecoder3D, self).__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)
        self._init_loss_regularization(loss_regularization)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        assert self.with_decode_head, \
            '3D EncoderDecoder Segmentor should have a decode_head'

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = build_head(decode_head)
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(build_head(head_cfg))
            else:
                self.auxiliary_head = build_head(auxiliary_head)

    def _init_loss_regularization(self, loss_regularization):
        """Initialize ``loss_regularization``"""
        if loss_regularization is not None:
            if isinstance(loss_regularization, list):
                self.loss_regularization = nn.ModuleList()
                for loss_cfg in loss_regularization:
                    self.loss_regularization.append(build_loss(loss_cfg))
            else:
                self.loss_regularization = build_loss(loss_regularization)

    def extract_feat(self, points):
        """Extract features from points."""
        x = self.backbone(points)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, points, img_metas):
        """Encode points with backbone and decode into a semantic segmentation
        map of the same size as input.

        Args:
            points (torch.Tensor): Input points of shape [B, N, 3+C].
            img_metas (list[dict]): Meta information of each sample.

        Returns:
            torch.Tensor: Segmentation logits of shape [B, num_classes, N].
        """
        x = self.extract_feat(points)
        out = self._decode_head_forward_test(x, img_metas)
        return out

    def _decode_head_forward_train(self, x, img_metas, pts_semantic_mask):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     pts_semantic_mask,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, pts_semantic_mask):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  pts_semantic_mask,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, pts_semantic_mask, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def _loss_regularization_forward_train(self):
        """Calculate regularization loss for model weight in training."""
        losses = dict()
        if isinstance(self.loss_regularization, nn.ModuleList):
            for idx, regularize_loss in enumerate(self.loss_regularization):
                loss_regularize = dict(
                    loss_regularize=regularize_loss(self.modules()))
                losses.update(add_prefix(loss_regularize, f'regularize_{idx}'))
        else:
            loss_regularize = dict(
                loss_regularize=self.loss_regularization(self.modules()))
            losses.update(add_prefix(loss_regularize, 'regularize'))

        return losses

    def forward_dummy(self, points):
        """Dummy forward function."""
        seg_logit = self.encode_decode(points, None)

        return seg_logit

    def forward_train(self, points, img_metas, pts_semantic_mask):
        """Forward function for training.

        Args:
            points (list[torch.Tensor]): List of points of shape [N, C].
            img_metas (list): Image metas.
            pts_semantic_mask (list[torch.Tensor]): List of point-wise semantic
                labels of shape [N].

        Returns:
            dict[str, Tensor]: Losses.
        """
        points_cat = torch.stack(points)
        pts_semantic_mask_cat = torch.stack(pts_semantic_mask)

        # extract features using backbone
        x = self.extract_feat(points_cat)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      pts_semantic_mask_cat)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, pts_semantic_mask_cat)
            losses.update(loss_aux)

        if self.with_regularization_loss:
            loss_regularize = self._loss_regularization_forward_train()
            losses.update(loss_regularize)

        return losses

    @staticmethod
    def _input_generation(coords,
                          patch_center,
                          coord_max,
                          feats,
                          use_normalized_coord=False):
        """Generating model input.

        Generate input by subtracting patch center and adding additional
            features. Currently support colors and normalized xyz as features.

        Args:
            coords (torch.Tensor): Sampled 3D point coordinate of shape [S, 3].
            patch_center (torch.Tensor): Center coordinate of the patch.
            coord_max (torch.Tensor): Max coordinate of all 3D points.
            feats (torch.Tensor): Features of sampled points of shape [S, C].
            use_normalized_coord (bool, optional): Whether to use normalized
                xyz as additional features. Defaults to False.

        Returns:
            torch.Tensor: The generated input data of shape [S, 3+C'].
        """
        # subtract patch center, the z dimension is not centered
        centered_coords = coords.clone()
        centered_coords[:, 0] -= patch_center[0]
        centered_coords[:, 1] -= patch_center[1]

        # normalized coordinates as extra features
        if use_normalized_coord:
            normalized_coord = coords / coord_max
            feats = torch.cat([feats, normalized_coord], dim=1)

        points = torch.cat([centered_coords, feats], dim=1)

        return points

    def _sliding_patch_generation(self,
                                  points,
                                  num_points,
                                  block_size,
                                  sample_rate=0.5,
                                  use_normalized_coord=False,
                                  eps=1e-3):
        """Sampling points in a sliding window fashion.

        First sample patches to cover all the input points.
        Then sample points in each patch to batch points of a certain number.

        Args:
            points (torch.Tensor): Input points of shape [N, 3+C].
            num_points (int): Number of points to be sampled in each patch.
            block_size (float, optional): Size of a patch to sample.
            sample_rate (float, optional): Stride used in sliding patch.
                Defaults to 0.5.
            use_normalized_coord (bool, optional): Whether to use normalized
                xyz as additional features. Defaults to False.
            eps (float, optional): A value added to patch boundary to guarantee
                points coverage. Defaults to 1e-3.

        Returns:
            np.ndarray | np.ndarray:

                - patch_points (torch.Tensor): Points of different patches of
                    shape [K, N, 3+C].
                - patch_idxs (torch.Tensor): Index of each point in
                    `patch_points`, of shape [K, N].
        """
        device = points.device
        # we assume the first three dims are points' 3D coordinates
        # and the rest dims are their per-point features
        coords = points[:, :3]
        feats = points[:, 3:]

        coord_max = coords.max(0)[0]
        coord_min = coords.min(0)[0]
        stride = block_size * sample_rate
        num_grid_x = int(
            torch.ceil((coord_max[0] - coord_min[0] - block_size) /
                       stride).item() + 1)
        num_grid_y = int(
            torch.ceil((coord_max[1] - coord_min[1] - block_size) /
                       stride).item() + 1)

        patch_points, patch_idxs = [], []
        for idx_y in range(num_grid_y):
            s_y = coord_min[1] + idx_y * stride
            e_y = torch.min(s_y + block_size, coord_max[1])
            s_y = e_y - block_size
            for idx_x in range(num_grid_x):
                s_x = coord_min[0] + idx_x * stride
                e_x = torch.min(s_x + block_size, coord_max[0])
                s_x = e_x - block_size

                # extract points within this patch
                cur_min = torch.tensor([s_x, s_y, coord_min[2]]).to(device)
                cur_max = torch.tensor([e_x, e_y, coord_max[2]]).to(device)
                cur_choice = ((coords >= cur_min - eps) &
                              (coords <= cur_max + eps)).all(dim=1)

                if not cur_choice.any():  # no points in this patch
                    continue

                # sample points in this patch to multiple batches
                cur_center = cur_min + block_size / 2.0
                point_idxs = torch.nonzero(cur_choice, as_tuple=True)[0]
                num_batch = int(np.ceil(point_idxs.shape[0] / num_points))
                point_size = int(num_batch * num_points)
                replace = point_size > 2 * point_idxs.shape[0]
                num_repeat = point_size - point_idxs.shape[0]
                if replace:  # duplicate
                    point_idxs_repeat = point_idxs[torch.randint(
                        0, point_idxs.shape[0],
                        size=(num_repeat, )).to(device)]
                else:
                    point_idxs_repeat = point_idxs[torch.randperm(
                        point_idxs.shape[0])[:num_repeat]]

                choices = torch.cat([point_idxs, point_idxs_repeat], dim=0)
                choices = choices[torch.randperm(choices.shape[0])]

                # construct model input
                point_batches = self._input_generation(
                    coords[choices],
                    cur_center,
                    coord_max,
                    feats[choices],
                    use_normalized_coord=use_normalized_coord)

                patch_points.append(point_batches)
                patch_idxs.append(choices)

        patch_points = torch.cat(patch_points, dim=0)
        patch_idxs = torch.cat(patch_idxs, dim=0)

        # make sure all points are sampled at least once
        assert torch.unique(patch_idxs).shape[0] == points.shape[0], \
            'some points are not sampled in sliding inference'

        return patch_points, patch_idxs

    def slide_inference(self, point, img_meta, rescale):
        """Inference by sliding-window with overlap.

        Args:
            point (torch.Tensor): Input points of shape [N, 3+C].
            img_meta (dict): Meta information of input sample.
            rescale (bool): Whether transform to original number of points.
                Will be used for voxelization based segmentors.

        Returns:
            Tensor: The output segmentation map of shape [num_classes, N].
        """
        num_points = self.test_cfg.num_points
        block_size = self.test_cfg.block_size
        sample_rate = self.test_cfg.sample_rate
        use_normalized_coord = self.test_cfg.use_normalized_coord
        batch_size = self.test_cfg.batch_size * num_points

        # patch_points is of shape [K*N, 3+C], patch_idxs is of shape [K*N]
        patch_points, patch_idxs = self._sliding_patch_generation(
            point, num_points, block_size, sample_rate, use_normalized_coord)
        feats_dim = patch_points.shape[1]
        seg_logits = []  # save patch predictions

        for batch_idx in range(0, patch_points.shape[0], batch_size):
            batch_points = patch_points[batch_idx:batch_idx + batch_size]
            batch_points = batch_points.view(-1, num_points, feats_dim)
            # batch_seg_logit is of shape [B, num_classes, N]
            batch_seg_logit = self.encode_decode(batch_points, img_meta)
            batch_seg_logit = batch_seg_logit.transpose(1, 2).contiguous()
            seg_logits.append(batch_seg_logit.view(-1, self.num_classes))

        # aggregate per-point logits by indexing sum and dividing count
        seg_logits = torch.cat(seg_logits, dim=0)  # [K*N, num_classes]
        expand_patch_idxs = patch_idxs.unsqueeze(1).repeat(1, self.num_classes)
        preds = point.new_zeros((point.shape[0], self.num_classes)).\
            scatter_add_(dim=0, index=expand_patch_idxs, src=seg_logits)
        count_mat = torch.bincount(patch_idxs)
        preds = preds / count_mat[:, None]

        # TODO: if rescale and voxelization segmentor

        return preds.transpose(0, 1)  # to [num_classes, K*N]

    def whole_inference(self, points, img_metas, rescale):
        """Inference with full scene (one forward pass without sliding)."""
        seg_logit = self.encode_decode(points, img_metas)
        # TODO: if rescale and voxelization segmentor
        return seg_logit

    def inference(self, points, img_metas, rescale):
        """Inference with slide/whole style.

        Args:
            points (torch.Tensor): Input points of shape [B, N, 3+C].
            img_metas (list[dict]): Meta information of each sample.
            rescale (bool): Whether transform to original number of points.
                Will be used for voxelization based segmentors.

        Returns:
            Tensor: The output segmentation map.
        """
        assert self.test_cfg.mode in ['slide', 'whole']
        if self.test_cfg.mode == 'slide':
            seg_logit = torch.stack([
                self.slide_inference(point, img_meta, rescale)
                for point, img_meta in zip(points, img_metas)
            ], 0)
        else:
            seg_logit = self.whole_inference(points, img_metas, rescale)
        output = F.softmax(seg_logit, dim=1)
        return output

    def simple_test(self, points, img_metas, rescale=True):
        """Simple test with single scene.

        Args:
            points (list[torch.Tensor]): List of points of shape [N, 3+C].
            img_metas (list[dict]): Meta information of each sample.
            rescale (bool): Whether transform to original number of points.
                Will be used for voxelization based segmentors.
                Defaults to True.

        Returns:
            list[dict]: The output prediction result with following keys:

                - semantic_mask (Tensor): Segmentation mask of shape [N].
        """
        # 3D segmentation requires per-point prediction, so it's impossible
        # to use down-sampling to get a batch of scenes with same num_points
        # therefore, we only support testing one scene every time
        seg_pred = []
        for point, img_meta in zip(points, img_metas):
            seg_prob = self.inference(point.unsqueeze(0), [img_meta],
                                      rescale)[0]
            seg_map = seg_prob.argmax(0)  # [N]
            # to cpu tensor for consistency with det3d
            seg_map = seg_map.cpu()
            seg_pred.append(seg_map)
        # warp in dict
        seg_pred = [dict(semantic_mask=seg_map) for seg_map in seg_pred]
        return seg_pred

    def aug_test(self, points, img_metas, rescale=True):
        """Test with augmentations.

        Args:
            points (list[torch.Tensor]): List of points of shape [B, N, 3+C].
            img_metas (list[list[dict]]): Meta information of each sample.
                Outer list are different samples while inner is different augs.
            rescale (bool): Whether transform to original number of points.
                Will be used for voxelization based segmentors.
                Defaults to True.

        Returns:
            list[dict]: The output prediction result with following keys:

                - semantic_mask (Tensor): Segmentation mask of shape [N].
        """
        # in aug_test, one scene going through different augmentations could
        # have the same number of points and are stacked as a batch
        # to save memory, we get augmented seg logit inplace
        seg_pred = []
        for point, img_meta in zip(points, img_metas):
            seg_prob = self.inference(point, img_meta, rescale)
            seg_prob = seg_prob.mean(0)  # [num_classes, N]
            seg_map = seg_prob.argmax(0)  # [N]
            # to cpu tensor for consistency with det3d
            seg_map = seg_map.cpu()
            seg_pred.append(seg_map)
        # warp in dict
        seg_pred = [dict(semantic_mask=seg_map) for seg_map in seg_pred]
        return seg_pred
