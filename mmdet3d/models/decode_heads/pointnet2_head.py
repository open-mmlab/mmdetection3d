from torch import nn as nn

from mmdet3d.ops import PointFPModule
from mmseg.models import HEADS
from .decode_head import Base3DDecodeHead


@HEADS.register_module()
class PointNet2Head(Base3DDecodeHead):
    """PointNet2 decoder head.

    Args:
        fp_channels (tuple[tuple[int]]): List of mlp channels in FP modules.
    """

    def __init__(self,
                 fp_channels=((768, 256, 256), (384, 256, 256),
                              (320, 256, 128), (128, 128, 128, 128)),
                 **kwargs):
        super(PointNet2Head, self).__init__(**kwargs)

        self.num_fp = len(fp_channels)
        self.FP_modules = nn.ModuleList()
        for cur_fp_mlps in fp_channels:
            self.FP_modules.append(PointFPModule(mlp_channels=cur_fp_mlps))

    def _extract_input(self, feat_dict):
        """Extract inputs from features dictionary.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            torch.Tensor: Coordinates of multiple levels of points.
            torch.Tensor: Features of multiple levels of points.
        """
        sa_xyz = feat_dict['sa_xyz']
        sa_features = feat_dict['sa_features']

        return sa_xyz, sa_features

    def forward(self, feat_dict):
        """Forward pass.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            torch.Tensor: Segmentation map of shape [B, num_classes, N].
        """
        sa_xyz, sa_features = self._extract_input(feat_dict)

        fp_features = sa_features[-1]

        for i in range(self.num_fp):
            # consume the points in a bottom-up manner
            fp_features = self.FP_modules[i](sa_xyz[-(i + 2)],
                                             sa_xyz[-(i + 1)],
                                             sa_features[-(i + 2)],
                                             fp_features)
        output = self.cls_seg(fp_features)

        return output
