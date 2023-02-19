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


@MODELS.register_module()
class MinkUNetBackbone(BaseModule):
    """MinkUNet backbone is the implementation of `4D Spatio-Temporal ConvNets.

    <https://arxiv.org/abs/1904.08755>` with torchsparse backend.

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
                 enc_channels: Sequence[int] = [32, 64, 128, 256],
                 dec_channels: Sequence[int] = [256, 128, 96, 96],
                 num_stages: int = 4,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg)
        assert num_stages == len(enc_channels) == len(dec_channels)
        self.num_stages = num_stages
        self.conv_input = nn.Sequential(
            TorchsparseConvModule(in_channels, base_channels, kernel_size=3),
            TorchsparseConvModule(base_channels, base_channels, kernel_size=3))
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        enc_channels.insert(0, base_channels)
        dec_channels.insert(0, enc_channels[-1])
        for i in range(num_stages):
            self.encoder.append(
                nn.Sequential(
                    TorchsparseConvModule(
                        enc_channels[i],
                        enc_channels[i],
                        kernel_size=2,
                        stride=2),
                    TorchsparseResidualBlock(
                        enc_channels[i], enc_channels[i + 1], kernel_size=3),
                    TorchsparseResidualBlock(
                        enc_channels[i + 1],
                        enc_channels[i + 1],
                        kernel_size=3)))

            self.decoder.append(
                nn.ModuleList([
                    TorchsparseConvModule(
                        dec_channels[i],
                        dec_channels[i + 1],
                        kernel_size=2,
                        stride=2,
                        transposed=True),
                    nn.Sequential(
                        TorchsparseResidualBlock(
                            dec_channels[i + 1] + enc_channels[-2 - i],
                            dec_channels[i + 1],
                            kernel_size=3),
                        TorchsparseResidualBlock(
                            dec_channels[i + 1],
                            dec_channels[i + 1],
                            kernel_size=3))
                ]))

    def forward(self, voxel_features: Tensor, coors: Tensor) -> SparseTensor:
        """Forward function.

        Args:
            voxel_features (Tensor): Voxel features in shape (N, C).
            coors (Tensor): Coordinates in shape (N, 4),
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).

        Returns:
            SparseTensor: backbone features.
        """
        x = torchsparse.SparseTensor(voxel_features, coors)
        x = self.conv_input(x)
        laterals = [x]
        for enc in self.encoder:
            x = enc(x)
            laterals.append(x)
        laterals = laterals[:-1][::-1]

        dec_outs = []
        for i, dec in enumerate(self.decoder):
            x = dec[0](x)
            x = torchsparse.cat((x, laterals[i]))
            x = dec[1](x)
            dec_outs.append(x)

        return dec_outs[-1]
