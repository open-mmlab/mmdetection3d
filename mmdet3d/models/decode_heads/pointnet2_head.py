# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence, Tuple

from mmcv.cnn.bricks import ConvModule
from torch import Tensor
from torch import nn as nn

from mmdet3d.models.layers import PointFPModule
from mmdet3d.registry import MODELS
from mmdet3d.utils.typing_utils import ConfigType
from .decode_head import Base3DDecodeHead


@MODELS.register_module()
class PointNet2Head(Base3DDecodeHead):
    r"""PointNet2 decoder head.

    Decoder head used in `PointNet++ <https://arxiv.org/abs/1706.02413>`_.
    Refer to the `official code <https://github.com/charlesq34/pointnet2>`_.

    Args:
        fp_channels (Sequence[Sequence[int]]): Tuple of mlp channels in FP
            modules. Defaults to ((768, 256, 256), (384, 256, 256),
            (320, 256, 128), (128, 128, 128, 128)).
        fp_norm_cfg (dict or :obj:`ConfigDict`): Config of norm layers used
            in FP modules. Defaults to dict(type='BN2d').
    """

    def __init__(self,
                 fp_channels: Sequence[Sequence[int]] = ((768, 256, 256),
                                                         (384, 256, 256),
                                                         (320, 256, 128),
                                                         (128, 128, 128, 128)),
                 fp_norm_cfg: ConfigType = dict(type='BN2d'),
                 **kwargs) -> None:
        super(PointNet2Head, self).__init__(**kwargs)

        self.num_fp = len(fp_channels)
        self.FP_modules = nn.ModuleList()
        for cur_fp_mlps in fp_channels:
            self.FP_modules.append(
                PointFPModule(mlp_channels=cur_fp_mlps, norm_cfg=fp_norm_cfg))

        # https://github.com/charlesq34/pointnet2/blob/master/models/pointnet2_sem_seg.py#L40
        self.pre_seg_conv = ConvModule(
            fp_channels[-1][-1],
            self.channels,
            kernel_size=1,
            bias=True,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def _extract_input(self,
                       feat_dict: dict) -> Tuple[List[Tensor], List[Tensor]]:
        """Extract inputs from features dictionary.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            Tuple[List[Tensor], List[Tensor]]: Coordinates and features of
            multiple levels of points.
        """
        sa_xyz = feat_dict['sa_xyz']
        sa_features = feat_dict['sa_features']
        assert len(sa_xyz) == len(sa_features)

        return sa_xyz, sa_features

    def forward(self, feat_dict: dict) -> Tensor:
        """Forward pass.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            Tensor: Segmentation map of shape [B, num_classes, N].
        """
        sa_xyz, sa_features = self._extract_input(feat_dict)

        # https://github.com/charlesq34/pointnet2/blob/master/models/pointnet2_sem_seg.py#L24
        sa_features[0] = None

        fp_feature = sa_features[-1]

        for i in range(self.num_fp):
            # consume the points in a bottom-up manner
            fp_feature = self.FP_modules[i](sa_xyz[-(i + 2)], sa_xyz[-(i + 1)],
                                            sa_features[-(i + 2)], fp_feature)
        output = self.pre_seg_conv(fp_feature)
        output = self.cls_seg(output)

        return output
