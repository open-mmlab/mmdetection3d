import torch
from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint
from torch import nn as nn

from mmdet3d.ops import PointSAModuleMSG
from mmdet.models import BACKBONES


@BACKBONES.register_module()
class PointNet2SAMSG(nn.Module):
    """PointNet2 with Multi-scale grouping.

    Args:
        in_channels (int): Input channels of point cloud.
        num_points (tuple[int]): The number of points which each SA
            module samples.
        radius (tuple[float]): Sampling radii of each SA module.
        num_samples (tuple[int]): The number of samples for ball
            query in each SA module.
        sa_channels (tuple[tuple[int]]): Out channels of each mlp in SA module.
        aggregation_channels (tuple[int]): Out channels of aggregation
            multi-scale grouping features.
        fps_mods (tuple[int]): Mod of FPS for each SA module.
        fps_sample_range_lists (tuple[tuple[int]]): The number of sampling
            points which each SA module samples.
        norm_cfg (dict): Config of normalization layer.
        pool_mod (str): Pool method ('max' or 'avg') for SA modules.
        use_xyz (bool): Whether to use xyz as a part of features.
        normalize_xyz (bool): Whether to normalize xyz with radii in
            each SA module.
    """

    def __init__(self,
                 in_channels,
                 num_points=(2048, 1024, 512, 256),
                 radius=(0.2, 0.4, 0.8, 1.2),
                 num_samples=(64, 32, 16, 16),
                 sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                              (128, 128, 256)),
                 aggregation_channels=(64, 128, 256),
                 fps_mods=None,
                 fps_sample_range_lists=None,
                 norm_cfg=dict(type='BN2d'),
                 pool_mod='max',
                 use_xyz=True,
                 normalize_xyz=True):
        super().__init__()

        self.num_sa = len(sa_channels)

        assert len(num_points) == len(radius) == len(num_samples) == len(
            sa_channels) == len(aggregation_channels)
        assert pool_mod in ['max', 'avg']

        self.SA_modules = nn.ModuleList()
        self.aggregation_mlps = nn.ModuleList()
        sa_in_channel = in_channels - 3  # number of channels without xyz
        skip_channel_list = [sa_in_channel]

        for sa_index in range(self.num_sa):
            cur_sa_mlps = list(sa_channels[sa_index])
            sa_out_channel = 0
            for radius_index in range(len(radius[sa_index])):
                cur_sa_mlps[radius_index] = [sa_in_channel] + list(
                    cur_sa_mlps[radius_index])
                sa_out_channel += cur_sa_mlps[radius_index][-1]

            if isinstance(fps_mods[sa_index], tuple):
                cur_fps_mod = list(fps_mods[sa_index])
            else:
                cur_fps_mod = list([fps_mods[sa_index]])

            if isinstance(fps_sample_range_lists[sa_index], tuple):
                cur_fps_sample_range_list = list(
                    fps_sample_range_lists[sa_index])
            else:
                cur_fps_sample_range_list = list(
                    [fps_sample_range_lists[sa_index]])

            self.SA_modules.append(
                PointSAModuleMSG(
                    num_point=num_points[sa_index],
                    radii=radius[sa_index],
                    sample_nums=num_samples[sa_index],
                    mlp_channels=cur_sa_mlps,
                    fps_mod=cur_fps_mod,
                    fps_sample_range_list=cur_fps_sample_range_list,
                    norm_cfg=norm_cfg,
                    use_xyz=use_xyz,
                    pool_mod=pool_mod,
                    normalize_xyz=normalize_xyz,
                    bias=True))
            skip_channel_list.append(sa_out_channel)
            self.aggregation_mlps.append(
                ConvModule(
                    sa_out_channel,
                    aggregation_channels[sa_index],
                    conv_cfg=dict(type='Conv1d'),
                    norm_cfg=dict(type='BN1d'),
                    kernel_size=1,
                    bias=True))
            sa_in_channel = aggregation_channels[sa_index]

    def init_weights(self, pretrained=None):
        """Initialize the weights of PointNet backbone."""
        # Do not initialize the conv layers
        # to follow the original implementation
        if isinstance(pretrained, str):
            from mmdet3d.utils import get_root_logger
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    @staticmethod
    def _split_point_feats(points):
        """Split coordinates and features of input points.

        Args:
            points (torch.Tensor): Point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).

        Returns:
            torch.Tensor: Coordinates of input points.
            torch.Tensor: Features of input points.
        """
        xyz = points[..., 0:3].contiguous()
        if points.size(-1) > 3:
            features = points[..., 3:].transpose(1, 2).contiguous()
        else:
            features = None

        return xyz, features

    def forward(self, points):
        """Forward pass.

        Args:
            points (torch.Tensor): point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).

        Returns:
            dict[str, torch.Tensor]: Outputs of the last SA module.

                - sa_xyz (torch.Tensor): The coordinates of sa features.
                - sa_features (torch.Tensor): The features from the
                    last Set Aggregation Layers.
                - sa_indices (torch.Tensor): Indices of the \
                    input points.
        """
        xyz, features = self._split_point_feats(points)

        batch, num_points = xyz.shape[:2]
        indices = xyz.new_tensor(range(num_points)).unsqueeze(0).repeat(
            batch, 1).long()

        sa_xyz = [xyz]
        sa_features = [features]
        sa_indices = [indices]

        for i in range(self.num_sa):
            cur_xyz, cur_features, cur_indices = self.SA_modules[i](
                sa_xyz[i], sa_features[i])
            cur_features = self.aggregation_mlps[i](cur_features)
            sa_xyz.append(cur_xyz)
            sa_features.append(cur_features)
            sa_indices.append(
                torch.gather(sa_indices[-1], 1, cur_indices.long()))

        ret = dict(
            sa_xyz=sa_xyz[-1],
            sa_features=sa_features[-1],
            sa_indices=sa_indices[-1])
        return ret
