# Copyright (c) OpenMMLab. All rights reserved.
try:
    import MinkowskiEngine as ME
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')

import torch
from torch import nn as nn

from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder3D


@SEGMENTORS.register_module()
class SparseEncoderDecoder3D(EncoderDecoder3D):
    r"""3D Sparse Encoder Decoder segmentors.
    Sparse version of `EncoderDecoder3D` class.

    Args:
        voxel_size (float): voxel size for point cloud processing
            Defaults to 0.05 (cm).
    """

    def __init__(self,
                 voxel_size,
                 *args,
                 **kwargs):
        super(SparseEncoderDecoder3D, self).__init__(
            *args, **kwargs)

        self.voxel_size = voxel_size

    def _collate(self, points):
        """Transform data from PyTorch-based to MinkowskiEngine-based.
        
        Args:
            points (list[torch.Tensor]): Input points of shape [B, N, 3+C].

        Returns:
            MinkowskiEngine.TensorField: TensorField data container.
        """
        
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
            x.features[:, -1, None],
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        ).features[:, 0].round().long()

        x = ME.SparseTensor(
            x.features[:, :self.backbone.in_channels],
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )

        # extract features using backbone
        x = self.extract_feat(x)

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
        field = self._collate(points)
        x = field.sparse()
        x = self.extract_feat(x)
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
