# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from mmengine.model import BaseModule
from mmengine.registry import MODELS
from torch import Tensor, nn

from mmdet3d.models.layers import (TorchSparseConvModule,
                                   TorchSparseResidualBlock)
from mmdet3d.models.layers.torchsparse import IS_TORCHSPARSE_AVAILABLE
from mmdet3d.utils import OptMultiConfig

if IS_TORCHSPARSE_AVAILABLE:
    import torchsparse
    from torchsparse.tensor import SparseTensor
else:
    SparseTensor = None


@MODELS.register_module()
class MinkUNetBackbone(BaseModule):
    r"""MinkUNet backbone with TorchSparse backend.

    Refer to `implementation code <https://github.com/mit-han-lab/spvnas>`_.

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
        init_cfg (dict or :obj:`ConfigDict` or List[dict or :obj:`ConfigDict`]
            , optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels: int = 4,
                 base_channels: int = 32,
                 encoder_channels: List[int] = [32, 64, 128, 256],
                 decoder_channels: List[int] = [256, 128, 96, 96],
                 num_stages: int = 4,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg)
        assert num_stages == len(encoder_channels) == len(decoder_channels)
        self.num_stages = num_stages
        self.conv_input = nn.Sequential(
            TorchSparseConvModule(in_channels, base_channels, kernel_size=3),
            TorchSparseConvModule(base_channels, base_channels, kernel_size=3))
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        encoder_channels.insert(0, base_channels)
        decoder_channels.insert(0, encoder_channels[-1])
        for i in range(num_stages):
            self.encoder.append(
                nn.Sequential(
                    TorchSparseConvModule(
                        encoder_channels[i],
                        encoder_channels[i],
                        kernel_size=2,
                        stride=2),
                    TorchSparseResidualBlock(
                        encoder_channels[i],
                        encoder_channels[i + 1],
                        kernel_size=3),
                    TorchSparseResidualBlock(
                        encoder_channels[i + 1],
                        encoder_channels[i + 1],
                        kernel_size=3)))

            self.decoder.append(
                nn.ModuleList([
                    TorchSparseConvModule(
                        decoder_channels[i],
                        decoder_channels[i + 1],
                        kernel_size=2,
                        stride=2,
                        transposed=True),
                    nn.Sequential(
                        TorchSparseResidualBlock(
                            decoder_channels[i + 1] + encoder_channels[-2 - i],
                            decoder_channels[i + 1],
                            kernel_size=3),
                        TorchSparseResidualBlock(
                            decoder_channels[i + 1],
                            decoder_channels[i + 1],
                            kernel_size=3))
                ]))

    def forward(self, voxel_features: Tensor, coors: Tensor) -> SparseTensor:
        """Forward function.

        Args:
            voxel_features (Tensor): Voxel features in shape (N, C).
            coors (Tensor): Coordinates in shape (N, 4),
                the columns in the order of (x_idx, y_idx, z_idx, batch_idx).

        Returns:
            SparseTensor: Backbone features.
        """
        x = torchsparse.SparseTensor(voxel_features, coors)
        x = self.conv_input(x)
        laterals = [x]
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            laterals.append(x)
        laterals = laterals[:-1][::-1]

        decoder_outs = []
        for i, decoder_layer in enumerate(self.decoder):
            x = decoder_layer[0](x)
            x = torchsparse.cat((x, laterals[i]))
            x = decoder_layer[1](x)
            decoder_outs.append(x)

        return decoder_outs[-1]
