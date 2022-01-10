# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn.bricks import ConvModule

from mmdet3d.ops import DGCNNFPModule
from mmdet.models import HEADS
from .decode_head import Base3DDecodeHead


@HEADS.register_module()
class DGCNNHead(Base3DDecodeHead):
    r"""DGCNN decoder head.

    Decoder head used in `DGCNN <https://arxiv.org/abs/1801.07829>`_.
    Refer to the
    `reimplementation code <https://github.com/AnTao97/dgcnn.pytorch>`_.

    Args:
        fp_channels (tuple[int], optional): Tuple of mlp channels in feature
            propagation (FP) modules. Defaults to (1216, 512).
    """

    def __init__(self, fp_channels=(1216, 512), **kwargs):
        super(DGCNNHead, self).__init__(**kwargs)

        self.FP_module = DGCNNFPModule(
            mlp_channels=fp_channels, act_cfg=self.act_cfg)

        # https://github.com/charlesq34/pointnet2/blob/master/models/pointnet2_sem_seg.py#L40
        self.pre_seg_conv = ConvModule(
            fp_channels[-1],
            self.channels,
            kernel_size=1,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def _extract_input(self, feat_dict):
        """Extract inputs from features dictionary.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            torch.Tensor: points for decoder.
        """
        fa_points = feat_dict['fa_points']

        return fa_points

    def forward(self, feat_dict):
        """Forward pass.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            torch.Tensor: Segmentation map of shape [B, num_classes, N].
        """
        fa_points = self._extract_input(feat_dict)

        fp_points = self.FP_module(fa_points)
        fp_points = fp_points.transpose(1, 2).contiguous()
        output = self.pre_seg_conv(fp_points)
        output = self.cls_seg(output)

        return output
