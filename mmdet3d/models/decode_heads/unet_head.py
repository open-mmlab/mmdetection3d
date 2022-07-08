# Copyright (c) OpenMMLab. All rights reserved.
# Follow https://github.com/NVIDIA/MinkowskiEngine/blob/master/examples/minkunet.py # noqa
try:
    import MinkowskiEngine as ME
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')

from mmcv.cnn.bricks import ConvModule
from torch import nn as nn

from mmdet3d.ops import PointFPModule
from ..builder import HEADS
from .decode_head import Base3DDecodeHead


@HEADS.register_module()
class UNetHead(Base3DDecodeHead):
    r"""UNet head. See `4D Spatio-Temporal ConvNets
    <https://arxiv.org/abs/1904.08755>`_ for more details.

    Args:
        fp_channels (tuple[tuple[int]]): Tuple of mlp channels in FP modules.
        fp_norm_cfg (dict): Config of norm layers used in FP modules.
            Default: dict(type='BN2d').
    """

    def __init__(self,
                 in_channels: int=96*4,
                 out_channels: int=18,
                 kernel_size: int=1,
                 D: int=3,
                 **kwargs):
        super(Base3DDecodeHead, self).__init__(**kwargs)

        self.final = ME.MinkowskiConvolution(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=True,
            dimension=D,
        )

    def _extract_input(self, feat_dict):
        """Extract inputs from features dictionary.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            list[torch.Tensor]: Features of multiple levels of points.
        """
        features = feat_dict['features']
        #assert len(sa_xyz) == len(sa_features)

        return features

    def forward(self, feat_dict):
        """Forward pass.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            torch.Tensor: Segmentation map of shape [B, num_classes, N].
        """
        features = self._extract_input(feat_dict)

        output = self.final(features)
        
        return output
