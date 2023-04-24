# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from mmengine.model import BaseModule
from mmengine.registry import MODELS
from torch import Tensor, nn

from mmdet3d.models.layers import (TorchSparseBasicBlock,
                                   TorchSparseBottleneck,
                                   TorchSparseConvModule)
from mmdet3d.models.layers.torchsparse import IS_TORCHSPARSE_AVAILABLE
from mmdet3d.utils import ConfigType, OptMultiConfig

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
        num_stages (int): Number of stages in encoder and decoder.
            Defaults to 4.
        block_type (str): Type of block in encoder and decoder.
        encoder_channels (List[int]): Convolutional channels of each encode
            layer. Defaults to [32, 64, 128, 256].
        encoder_blocks (List[int]): Number of blocks in each encode layer.
        decoder_channels (List[int]): Convolutional channels of each decode
            layer. Defaults to [256, 128, 96, 96].
        decoder_blocks (List[int]): Number of blocks in each decode layer.
        norm_cfg (dict or :obj:`ConfigDict`): Config of normalization.
        init_cfg (dict or :obj:`ConfigDict` or List[dict or :obj:`ConfigDict`]
            , optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels: int = 4,
                 base_channels: int = 32,
                 num_stages: int = 4,
                 block_type: str = 'basicblock',
                 encoder_channels: List[int] = [32, 64, 128, 256],
                 encoder_blocks: List[int] = [2, 2, 2, 2],
                 decoder_channels: List[int] = [256, 128, 96, 96],
                 decoder_blocks: List[int] = [2, 2, 2, 2],
                 norm_cfg: ConfigType = dict(type='TorchSparseBN'),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg)
        assert num_stages == len(encoder_channels) == len(decoder_channels)
        self.num_stages = num_stages
        self.conv_input = nn.Sequential(
            TorchSparseConvModule(
                in_channels, base_channels, kernel_size=3, norm_cfg=norm_cfg),
            TorchSparseConvModule(
                base_channels, base_channels, kernel_size=3,
                norm_cfg=norm_cfg))

        if block_type == 'basicblock':
            block = TorchSparseBasicBlock
        elif block_type == 'bottleneck':
            block = TorchSparseBottleneck
        else:
            raise NotImplementedError(f'Unsppported block type: {block_type}')

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
                        stride=2,
                        norm_cfg=norm_cfg),
                    block(
                        encoder_channels[i],
                        encoder_channels[i + 1],
                        kernel_size=3,
                        norm_cfg=norm_cfg), *[
                            block(
                                encoder_channels[i + 1],
                                encoder_channels[i + 1],
                                kernel_size=3,
                                norm_cfg=norm_cfg)
                        ] * (encoder_blocks[i] - 1)))

            self.decoder.append(
                nn.ModuleList([
                    TorchSparseConvModule(
                        decoder_channels[i],
                        decoder_channels[i + 1],
                        kernel_size=2,
                        stride=2,
                        transposed=True,
                        norm_cfg=norm_cfg),
                    nn.Sequential(
                        block(
                            decoder_channels[i + 1] + encoder_channels[-2 - i],
                            decoder_channels[i + 1],
                            kernel_size=3,
                            norm_cfg=norm_cfg), *[
                                block(
                                    decoder_channels[i + 1],
                                    decoder_channels[i + 1],
                                    kernel_size=3,
                                    norm_cfg=norm_cfg)
                            ] * (decoder_blocks[i] - 1))
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
