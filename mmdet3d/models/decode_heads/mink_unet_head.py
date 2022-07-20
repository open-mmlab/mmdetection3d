# Copyright (c) OpenMMLab. All rights reserved.
# Follow https://github.com/NVIDIA/MinkowskiEngine/blob/master/examples/minkunet.py # noqa
try:
    import MinkowskiEngine as ME
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')

from torch import nn as nn
from mmcv.runner import force_fp32

from ..builder import HEADS
from .decode_head import Base3DDecodeHead


@HEADS.register_module()
class MinkUNetHead(Base3DDecodeHead):
    r"""Sparse UNet head. See `4D Spatio-Temporal ConvNets
    <https://arxiv.org/abs/1904.08755>`_ for more details.

    Args:
        channels (int): dimension of features from a backbone.
            Defaults to 96.
        num_classes (int): number of classes (channels of final conv module).
            Defaults to 13.
        D (int): the spatial dimension of the space where all the inputs and the network are defined.
            For example, images are in a 2D space, meshes and 3D shapes are in a 3D space.
            Defaults to 3.
    """

    def __init__(self,
                 channels=96,
                 num_classes=13,
                 D: int=3,
                 **kwargs):
        super(MinkUNetHead, self).__init__(
            channels, num_classes, **kwargs)

        self.final = ME.MinkowskiConvolution(
            channels,
            num_classes,
            kernel_size=1,
            bias=True,
            dimension=D,
        )

    def _extract_input(self, feat_dict):
        """Extract inputs from features dictionary.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            list[MinkowskiEngine.SparseTensor]: Features of multiple levels of points.
        """
        features = feat_dict['features']

        return features

    def forward(self, feat_dict):
        """Forward pass.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            MinkowskiEngine.SparseTensor: Segmentation map of shape [B, num_classes, N].
        """
        features = self._extract_input(feat_dict)

        output = self.final(features)
        
        return output

    def forward_test(self, inputs, field, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (dict[MinkowskiEngine.SparseTensor]): Dict of multi-level point features.
            field (MinkowskiEngine.TensorField): TensorField to sample points
                equal to the output dimension (semantic segmentation mask).
            img_metas (list[dict]): Meta information of each sample.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        x = self.forward(inputs)
        x = x.slice(field)
        out = [
            x.features[permutation]
            for permutation in x.decomposition_permutations
        ]
        return out

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute semantic segmentation loss.

        Args:
            seg_logit (torch.Tensor): Predicted per-point segmentation logits
                of shape [B, num_classes, N].
            seg_label (torch.Tensor): Ground-truth segmentation label of
                shape [B, N].
        """
        loss = dict()
        loss['loss_sem_seg'] = self.loss_decode(
            seg_logit.features, seg_label, ignore_index=self.ignore_index)
        return loss
