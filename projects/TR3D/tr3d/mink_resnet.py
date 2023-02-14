# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

try:
    import MinkowskiEngine as ME
except ImportError:
    # Please follow getting_started.md to install MinkowskiEngine.
    ME = SparseTensor = None
    pass

from mmdet3d.models.backbones import MinkResNet
from mmdet3d.registry import MODELS


@MODELS.register_module()
class TR3DMinkResNet(MinkResNet):
    r"""Minkowski ResNet backbone. See `4D Spatio-Temporal ConvNets
    <https://arxiv.org/abs/1904.08755>`_ for more details. The onle difference
    with MinkResNet is the `norm` and `num_planes` parameters. These classes
    should be merged in the future.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input channels, 3 for RGB.
        num_stages (int): Resnet stages. Defaults to 4.
        pool (bool): Whether to add max pooling after first conv.
            Defaults to True.
        norm (str): Norm type ('instance' or 'batch') for stem layer.
            Usually ResNet implies BatchNorm but for some reason
            original MinkResNet implies InstanceNorm. Defaults to 'instance'.
        num_planes (tuple[int]): Number of planes per block before
            block.expansion. Defaults to (64, 128, 256, 512).
    """

    def __init__(self,
                 depth: int,
                 in_channels: int,
                 num_stages: int = 4,
                 pool: bool = True,
                 norm: str = 'instance',
                 num_planes: Tuple[int] = (64, 128, 256, 512)):
        super(TR3DMinkResNet, self).__init__(depth, in_channels, num_stages,
                                             pool)
        block, stage_blocks = self.arch_settings[depth]
        self.inplanes = 64
        norm_layer = ME.MinkowskiInstanceNorm if norm == 'instance' else \
            ME.MinkowskiBatchNorm
        self.norm1 = norm_layer(self.inplanes)

        for i in range(len(stage_blocks)):
            setattr(
                self, f'layer{i + 1}',
                self._make_layer(
                    block, num_planes[i], stage_blocks[i], stride=2))
