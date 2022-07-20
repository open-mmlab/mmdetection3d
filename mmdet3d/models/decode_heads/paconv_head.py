# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

from mmcv.cnn.bricks import ConvModule
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.utils.typing import ConfigType
from .pointnet2_head import PointNet2Head


@MODELS.register_module()
class PAConvHead(PointNet2Head):
    r"""PAConv decoder head.

    Decoder head used in `PAConv <https://arxiv.org/abs/2103.14635>`_.
    Refer to the `official code <https://github.com/CVMI-Lab/PAConv>`_.

    Args:
        fp_channels (tuple[tuple[int]]): Tuple of mlp channels in FP modules.
        fp_norm_cfg (dict): Config of norm layers used in FP modules.
            Default: dict(type='BN2d').
    """

    def __init__(self,
                 fp_channels: Tuple[Tuple[int]] = ((768, 256, 256),
                                                   (384, 256, 256), (320, 256,
                                                                     128),
                                                   (128 + 6, 128, 128, 128)),
                 fp_norm_cfg: ConfigType = dict(type='BN2d'),
                 **kwargs) -> None:
        super(PAConvHead, self).__init__(
            fp_channels=fp_channels, fp_norm_cfg=fp_norm_cfg, **kwargs)

        # https://github.com/CVMI-Lab/PAConv/blob/main/scene_seg/model/pointnet2/pointnet2_paconv_seg.py#L53
        # PointNet++'s decoder conv has bias while PAConv's doesn't have
        # so we need to rebuild it here
        self.pre_seg_conv = ConvModule(
            fp_channels[-1][-1],
            self.channels,
            kernel_size=1,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, feat_dict: dict) -> Tensor:
        """Forward pass.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            torch.Tensor: Segmentation map of shape [B, num_classes, N].
        """
        sa_xyz, sa_features = self._extract_input(feat_dict)

        # PointNet++ doesn't use the first level of `sa_features` as input
        # while PAConv inputs it through skip-connection
        fp_feature = sa_features[-1]

        for i in range(self.num_fp):
            # consume the points in a bottom-up manner
            fp_feature = self.FP_modules[i](sa_xyz[-(i + 2)], sa_xyz[-(i + 1)],
                                            sa_features[-(i + 2)], fp_feature)

        output = self.pre_seg_conv(fp_feature)
        output = self.cls_seg(output)

        return output
