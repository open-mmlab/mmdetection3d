# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

from mmengine.registry import MODELS
from torch import Tensor, nn

from mmdet3d.models.layers.torchsparse import IS_TORCHSPARSE_AVAILABLE
from mmdet3d.utils import OptMultiConfig
from .minkunet_backbone import (MinkUNetBackbone, initial_voxelize,
                                point_to_voxel, voxel_to_point)

if IS_TORCHSPARSE_AVAILABLE:
    import torchsparse
    from torchsparse.tensor import PointTensor, SparseTensor
else:
    PointTensor = SparseTensor = None


@MODELS.register_module()
class SPVCNNBackbone(MinkUNetBackbone):
    """SPVCNN backbone with torchsparse backend.

    More details can be found in `paper <https://arxiv.org/abs/2007.16100>`_ .

    Args:
        in_channels (int): Number of input voxel feature channels.
            Defaults to 4.
        base_channels (int): The input channels for first encoder layer.
            Defaults to 32.
        encoder_channels (List[int]): Convolutional channels of each encode
            layer. Defaults to [32, 64, 128, 256].
        decoder_channels (List[int]): Convolutional channels of each decode
            layer. Defaults to [256, 128, 96, 96].
        num_stages (int): Number of stages in encoder and decoder.
            Defaults to 4.
        drop_ratio (float): Dropout ratio of voxel features. Defaults to 0.3.
        init_cfg (dict or :obj:`ConfigDict` or list[dict or :obj:`ConfigDict`]
            , optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 in_channels: int = 4,
                 base_channels: int = 32,
                 encoder_channels: Sequence[int] = [32, 64, 128, 256],
                 decoder_channels: Sequence[int] = [256, 128, 96, 96],
                 num_stages: int = 4,
                 drop_ratio: float = 0.3,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            num_stages=num_stages,
            init_cfg=init_cfg)

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(base_channels, encoder_channels[-1]),
                nn.BatchNorm1d(encoder_channels[-1]), nn.ReLU(True)),
            nn.Sequential(
                nn.Linear(encoder_channels[-1], decoder_channels[2]),
                nn.BatchNorm1d(decoder_channels[2]), nn.ReLU(True)),
            nn.Sequential(
                nn.Linear(decoder_channels[2], decoder_channels[4]),
                nn.BatchNorm1d(decoder_channels[4]), nn.ReLU(True))
        ])
        self.dropout = nn.Dropout(drop_ratio, True)

    def forward(self, voxel_features: Tensor, coors: Tensor) -> Tensor:
        """Forward function.

        Args:
            voxel_features (Tensor): Voxel features in shape (N, C).
            coors (Tensor): Coordinates in shape (N, 4),
                the columns in the order of (x_idx, y_idx, z_idx, batch_idx).

        Returns:
            PointTensor: Backbone features.
        """
        voxels = SparseTensor(voxel_features, coors)
        points = PointTensor(voxels.F, voxels.C.float())
        voxels = initial_voxelize(points)

        voxels = self.conv_input(voxels)
        points = voxel_to_point(voxels, points)
        voxels = point_to_voxel(voxels, points)
        laterals = [voxels]
        for encoder in self.encoder:
            voxels = encoder(voxels)
            laterals.append(voxels)
        laterals = laterals[:-1][::-1]

        points = voxel_to_point(voxels, points, self.point_transforms[0])
        voxels = point_to_voxel(voxels, points)
        voxels.F = self.dropout(voxels.F)

        decoder_outs = []
        for i, decoder in enumerate(self.decoder):
            voxels = decoder[0](voxels)
            voxels = torchsparse.cat((voxels, laterals[i]))
            voxels = decoder[1](voxels)
            decoder_outs.append(voxels)
            if i == 1:
                points = voxel_to_point(voxels, points,
                                        self.point_transforms[1])
                voxels = point_to_voxel(voxels, points)
                voxels.F = self.dropout(voxels.F)

        points = voxel_to_point(voxels, points, self.point_transforms[2])
        return points.F
