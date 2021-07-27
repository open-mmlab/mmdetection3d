from mmdet3d.ops.dgcnn_modules.dgcnn_gf_module import DGCNNGFModule
import torch
from mmcv.runner import auto_fp16
from mmcv.runner import BaseModule
from torch import nn as nn

from mmdet3d.ops import (build_gf_module, build_fa_module, DGCNNFAModule,
                         DGCNNFPModule)
from mmdet.models import BACKBONES


@BACKBONES.register_module()
class DGCNNGF(BaseModule):
    """Backbone network for DGCNN.

    Args:
        in_channels (int): Input channels of point cloud.
        num_points (tuple[int]): The number of input points in each GF module.
        num_samples (tuple[int]): The number of samples for knn or ball query
            in each GF module.
        knn_mods (tuple[str]): If knn, mod of KNN of each GF module.
        radius (tuple[float]): Sampling radii of each GF module.
        gf_channels (tuple[tuple[int]]): Out channels of each mlp in GF module.
        fa_channels (tuple[int]): Out channels of each mlp in FA module.
        fp_channels (tuple[int]): Out channels of each mlp in FP module.
        act_cfg (dict): Config of activation layer.
        gf_cfg (dict): Config of graph feature module, which may contain the
            following keys and values:

            - pool_mod (str): Pool method ('max' or 'avg') for GF modules.
    """

    def __init__(self,
                 in_channels,
                 num_points=(4096, 4096, 4096),
                 num_samples=(20, 20, 20),
                 knn_mods=['D-KNN', 'F-KNN', 'F-KNN'],
                 radius=(None, None, None),
                 gf_channels=((64, 64), (64, 64), (64, )),
                 fa_channels=(1024, ),
                 fp_channels=(512, ),
                 act_cfg=dict(type='ReLU'),
                 gf_cfg=dict(type='DGCNNGFModule', pool_mod='max'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.num_gf = len(gf_channels)
        self.num_fp = len(fp_channels)

        assert len(num_points) == len(num_samples) == len(knn_mods) == len(
            radius) == len(gf_channels)

        self.GF_modules = nn.ModuleList()
        gf_in_channel = in_channels * 2
        skip_channel_list = [gf_in_channel]  # input channel list

        for gf_index in range(self.num_gf):
            cur_gf_mlps = list(gf_channels[gf_index])
            cur_gf_mlps = [gf_in_channel] + cur_gf_mlps
            gf_out_channel = cur_gf_mlps[-1]

            self.GF_modules.append(
                build_gf_module(
                    mlp_channels=cur_gf_mlps,
                    num_point=num_points[gf_index],
                    num_sample=num_samples[gf_index],
                    knn_mod=knn_mods[gf_index],
                    radius=radius[gf_index],
                    act_cfg=act_cfg,
                    cfg=gf_cfg))
            skip_channel_list.append(gf_out_channel)
            gf_in_channel = gf_out_channel * 2

        fa_in_channel = sum(skip_channel_list[1:])
        cur_fa_mlps = list(fa_channels)
        cur_fa_mlps = [fa_in_channel] + cur_fa_mlps

        self.FA_module = DGCNNFAModule(
            mlp_channels=cur_fa_mlps, act_cfg=act_cfg)

        fp_in_channel = fa_in_channel + fa_channels[-1]
        cur_fp_mlps = list(fp_channels)
        cur_fp_mlps = [fp_in_channel] + cur_fp_mlps

        self.FP_module = DGCNNFPModule(
            mlp_channels=cur_fp_mlps, act_cfg=act_cfg)

    @auto_fp16(apply_to=('points', ))
    def forward(self, points):
        """Forward pass.

        Args:
            points (torch.Tensor): point coordinates with features,
                with shape (B, N, in_channels).

        Returns:
            dict[str, list[torch.Tensor]]: Outputs after GF, FA and FP \
                modules.

                - gf_points (list[torch.Tensor]): Outputs after each GF module.
                - fa_points (torch.Tensor): Outputs after FA module.
                - fp_points (torch.Tensor): Outputs after FP module.
        """
        gf_points = [points]

        for i in range(self.num_gf):
            cur_points = self.GF_modules[i](gf_points[i])
            gf_points.append(cur_points)

        fa_points = self.FA_module(gf_points)

        fp_points = self.FP_module(fa_points)

        ret = dict(
            gf_points=gf_points, fa_points=fa_points, fp_points=fp_points)
        return ret
