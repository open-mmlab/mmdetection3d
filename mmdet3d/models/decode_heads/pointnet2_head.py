# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn.bricks import ConvModule
from torch import nn as nn

from mmdet3d.ops import PointFPModule
from ..builder import HEADS
from .decode_head import Base3DDecodeHead


@HEADS.register_module()
class PointNet2Head(Base3DDecodeHead):
    r"""PointNet2 decoder head.

    Decoder head used in `PointNet++ <https://arxiv.org/abs/1706.02413>`_.
    Refer to the `official code <https://github.com/charlesq34/pointnet2>`_.

    Args:
        fp_channels (tuple[tuple[int]]): Tuple of mlp channels in FP modules.
        fp_norm_cfg (dict): Config of norm layers used in FP modules.
            Default: dict(type='BN2d').
    """

    def __init__(self,
                 fp_channels=((768, 256, 256), (384, 256, 256),
                              (320, 256, 128), (128, 128, 128, 128)),
                 fp_norm_cfg=dict(type='BN2d'),
                 **kwargs):
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

    def _extract_input(self, feat_dict):
        """Extract inputs from features dictionary.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            list[torch.Tensor]: Coordinates of multiple levels of points.
            list[torch.Tensor]: Features of multiple levels of points.
        """
        sa_xyz = feat_dict['sa_xyz']
        sa_features = feat_dict['sa_features']
        assert len(sa_xyz) == len(sa_features)

        return sa_xyz, sa_features

    def forward(self, feat_dict):
        """Forward pass.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            torch.Tensor: Segmentation map of shape [B, num_classes, N].
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
