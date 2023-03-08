# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

from mmengine.model import BaseModule
from mmengine.registry import MODELS
from torch import Tensor, nn

from mmdet3d.models.layers import (IS_TORCHSPARSE_AVAILABLE,
                                   TorchsparseConvModule,
                                   TorchsparseResidualBlock)
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
        in_channels (int): Number of input image channels. Default" 3.
        base_channels (int): Number of base channels of each stage.
            The output channels of the first stage. Defaults to 64.
        enc_channels (tuple[int]): Convolutional channels of each encode block.
        dec_channels (tuple[int]): Convolutional channels of each decode block.
        num_stages (int): Number of stages in encoder and decoder.
            Defaults to 4.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels: int = 4,
                 base_channels: int = 32,
                 encoder_channels: Sequence[int] = [32, 64, 128, 256],
                 decoder_channels: Sequence[int] = [256, 128, 96, 96],
                 num_stages: int = 4,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg)
        assert num_stages == len(encoder_channels) == len(decoder_channels)
        self.num_stages = num_stages
        self.conv_input = nn.Sequential(
            TorchsparseConvModule(in_channels, base_channels, kernel_size=3),
            TorchsparseConvModule(base_channels, base_channels, kernel_size=3))
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        encoder_channels.insert(0, base_channels)
        decoder_channels.insert(0, encoder_channels[-1])
        for i in range(num_stages):
            self.encoder.append(
                nn.Sequential(
                    TorchsparseConvModule(
                        encoder_channels[i],
                        encoder_channels[i],
                        kernel_size=2,
                        stride=2),
                    TorchsparseResidualBlock(
                        encoder_channels[i],
                        encoder_channels[i + 1],
                        kernel_size=3),
                    TorchsparseResidualBlock(
                        encoder_channels[i + 1],
                        encoder_channels[i + 1],
                        kernel_size=3)))

            self.decoder.append(
                nn.ModuleList([
                    TorchsparseConvModule(
                        decoder_channels[i],
                        decoder_channels[i + 1],
                        kernel_size=2,
                        stride=2,
                        transposed=True),
                    nn.Sequential(
                        TorchsparseResidualBlock(
                            decoder_channels[i + 1] + encoder_channels[-2 - i],
                            decoder_channels[i + 1],
                            kernel_size=3),
                        TorchsparseResidualBlock(
                            decoder_channels[i + 1],
                            decoder_channels[i + 1],
                            kernel_size=3))
                ]))

    def forward(self, voxel_features: Tensor, coors: Tensor) -> SparseTensor:
        """Forward function.

        Args:
            voxel_features (Tensor): Voxel features in shape (N, C).
            coors (Tensor): Coordinates in shape (N, 4),
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).

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
