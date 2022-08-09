# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model import BaseModule
from torch import nn as nn

from mmdet3d.models.layers.pointnet_modules import PointFPModule
from mmdet3d.registry import MODELS


@MODELS.register_module()
class PointNetFPNeck(BaseModule):
    r"""PointNet FP Module used in PointRCNN.

    Refer to the `official code <https://github.com/charlesq34/pointnet2>`_.

    .. code-block:: none

        sa_n ----------------------------------------
                                                     |
        ... ---------------------------------        |
                                             |       |
        sa_1 -------------                   |       |
                          |                  |       |
        sa_0 -> fp_0 -> fp_module ->fp_1 -> ... -> fp_module -> fp_n

    sa_n including sa_xyz (torch.Tensor) and sa_features (torch.Tensor)
    fp_n including fp_xyz (torch.Tensor) and fp_features (torch.Tensor)

    Args:
        fp_channels (tuple[tuple[int]]): Tuple of mlp channels in FP modules.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self, fp_channels, init_cfg=None):
        super(PointNetFPNeck, self).__init__(init_cfg=init_cfg)

        self.num_fp = len(fp_channels)
        self.FP_modules = nn.ModuleList()
        for cur_fp_mlps in fp_channels:
            self.FP_modules.append(PointFPModule(mlp_channels=cur_fp_mlps))

    def _extract_input(self, feat_dict):
        """Extract inputs from features dictionary.

        Args:
            feat_dict (dict): Feature dict from backbone, which may contain
                the following keys and values:

                - sa_xyz (list[torch.Tensor]): Points of each sa module
                    in shape (N, 3).
                - sa_features (list[torch.Tensor]): Output features of
                    each sa module in shape (N, M).

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
            dict[str, torch.Tensor]: Outputs of the Neck.

                - fp_xyz (torch.Tensor): The coordinates of fp features.
                - fp_features (torch.Tensor): The features from the last
                    feature propagation layers.
        """
        sa_xyz, sa_features = self._extract_input(feat_dict)

        fp_feature = sa_features[-1]
        fp_xyz = sa_xyz[-1]

        for i in range(self.num_fp):
            # consume the points in a bottom-up manner
            fp_feature = self.FP_modules[i](sa_xyz[-(i + 2)], sa_xyz[-(i + 1)],
                                            sa_features[-(i + 2)], fp_feature)
            fp_xyz = sa_xyz[-(i + 2)]

        ret = dict(fp_xyz=fp_xyz, fp_features=fp_feature)
        return ret
