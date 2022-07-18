# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

try:
    import MinkowskiEngine as ME
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')

from mmseg.core import add_prefix
from ..builder import (
    SEGMENTORS, build_backbone,
    build_head, build_loss, build_neck)
from .base import Base3DSegmentor


@SEGMENTORS.register_module()
class SparseEncoderDecoder3D(Base3DSegmentor):
    r"""3D Sparse Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be thrown during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 voxel_size,
                 neck=None,
                 auxiliary_head=None,
                 loss_regularization=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SparseEncoderDecoder3D, self).__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)
        self._init_loss_regularization(loss_regularization)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.voxel_size = voxel_size
        assert self.with_decode_head, \
            '3D Sparse EncoderDecoder Segmentor should have a decode_head'

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

    def _collate(self, points):
        
        coordinates, features = ME.utils.batch_sparse_collate(
            data=[
                (p[:, :3] / self.voxel_size, p[:, 3:])
                for p in points
            ],
            dtype=points[0].dtype,
            device=points[0].device,
        )

        field = ME.TensorField(
            features=features,
            coordinates=coordinates,
            quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=points[0].device,
        )

        return field

    def extract_feat(self, x, img_metas):
        """Extract features from points."""
        x = self.backbone(x)
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
        points = [
            torch.cat([p, torch.unsqueeze(m, 1)], dim=1)
            for p, m in zip(points, pts_semantic_mask)
        ]

        field = self._collate(points)
        x = field.sparse()

        targets = ME.SparseTensor(
            x.features[:, 6:7],
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        ).features[:, 0].round().long()

        x = ME.SparseTensor(
            x.features[:, :3],
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )

        # extract features using backbone
        x = self.extract_feat(x, img_metas)

        losses = dict()

        loss_decode = self._decode_head_forward_train(
            x=x,
            img_metas=img_metas,
            pts_semantic_mask=targets,
        )
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x=x,
                img_metas=img_metas,
                pts_semantic_mask=targets,
            )
            losses.update(loss_aux)

        if self.with_regularization_loss:
            loss_regularize = self._loss_regularization_forward_train()
            losses.update(loss_regularize)

        return losses

    def simple_test(self, points, img_metas, rescale=False):
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
        field = self._collate(points=points)
        x = field.sparse()
        x = self.extract_feat(x, img_metas)
        masks = self.decode_head.forward_test(x, field, img_metas, self.test_cfg)
         
        out = [dict(semantic_mask=masks[0].argmax(1).cpu())]

        return out

    def aug_test(self, points, img_metas):
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
        field = self.collate([p[0] for p in points])
        x = field.sparse()
        x = self.extract_feat(x, img_metas)
        masks = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        masks = torch.mean(torch.sigmoid(torch.stack(masks)), dim=0)
        return [dict(semantic_mask=masks.argmax(1).cpu())]