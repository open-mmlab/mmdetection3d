# Copyright (c) OpenMMLab. All rights reserved.
r"""Modified from Cylinder3D.

Please refer to `Cylinder3D github page
<https://github.com/xinge008/Cylinder3D>`_ for details
"""

from typing import List, Optional

import numpy as np
import torch
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.ops import (SparseConv3d, SparseConvTensor, SparseInverseConv3d,
                      SubMConv3d)
from mmengine.model import BaseModule
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType


class AsymmResBlock(BaseModule):
    """Asymmetrical Residual Block.

    Args:
        in_channels (int): Input channels of the block.
        out_channels (int): Output channels of the block.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for
            normalization layer.
        act_cfg (:obj:`ConfigDict` or dict): Config dict of activation layers.
            Defaults to dict(type='LeakyReLU').
        indice_key (str, optional): Name of indice tables. Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_cfg: ConfigType,
                 act_cfg: ConfigType = dict(type='LeakyReLU'),
                 indice_key: Optional[str] = None):
        super().__init__()

        self.conv0_0 = SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + 'bef')
        self.act0_0 = build_activation_layer(act_cfg)
        self.bn0_0 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv0_1 = SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 1, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + 'bef')
        self.act0_1 = build_activation_layer(act_cfg)
        self.bn0_1 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv1_0 = SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 1, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + 'bef')
        self.act1_0 = build_activation_layer(act_cfg)
        self.bn1_0 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv1_1 = SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + 'bef')
        self.act1_1 = build_activation_layer(act_cfg)
        self.bn1_1 = build_norm_layer(norm_cfg, out_channels)[1]

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        """Forward pass."""
        shortcut = self.conv0_0(x)

        shortcut.features = self.act0_0(shortcut.features)
        shortcut.features = self.bn0_0(shortcut.features)

        shortcut = self.conv0_1(shortcut)
        shortcut.features = self.act0_1(shortcut.features)
        shortcut.features = self.bn0_1(shortcut.features)

        res = self.conv1_0(x)
        res.features = self.act1_0(res.features)
        res.features = self.bn1_0(res.features)

        res = self.conv1_1(res)
        res.features = self.act1_1(res.features)
        res.features = self.bn1_1(res.features)

        res.features = res.features + shortcut.features

        return res


class AsymmeDownBlock(BaseModule):
    """Asymmetrical DownSample Block.

    Args:
       in_channels (int): Input channels of the block.
       out_channels (int): Output channels of the block.
       norm_cfg (:obj:`ConfigDict` or dict): Config dict for
            normalization layer.
       act_cfg (:obj:`ConfigDict` or dict): Config dict of activation layers.
            Defaults to dict(type='LeakyReLU').
       pooling (bool): Whether pooling features at the end of
           block. Defaults: True.
       height_pooling (bool): Whether pooling features at
           the height dimension. Defaults: False.
       indice_key (str, optional): Name of indice tables. Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_cfg: ConfigType,
                 act_cfg: ConfigType = dict(type='LeakyReLU'),
                 pooling: bool = True,
                 height_pooling: bool = False,
                 indice_key: Optional[str] = None):
        super().__init__()
        self.pooling = pooling

        self.conv0_0 = SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 1, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + 'bef')
        self.act0_0 = build_activation_layer(act_cfg)
        self.bn0_0 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv0_1 = SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + 'bef')
        self.act0_1 = build_activation_layer(act_cfg)
        self.bn0_1 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv1_0 = SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + 'bef')
        self.act1_0 = build_activation_layer(act_cfg)
        self.bn1_0 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv1_1 = SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 1, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + 'bef')
        self.act1_1 = build_activation_layer(act_cfg)
        self.bn1_1 = build_norm_layer(norm_cfg, out_channels)[1]

        if pooling:
            if height_pooling:
                self.pool = SparseConv3d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    indice_key=indice_key,
                    bias=False)
            else:
                self.pool = SparseConv3d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=(2, 2, 1),
                    padding=1,
                    indice_key=indice_key,
                    bias=False)

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        """Forward pass."""
        shortcut = self.conv0_0(x)
        shortcut.features = self.act0_0(shortcut.features)
        shortcut.features = self.bn0_0(shortcut.features)

        shortcut = self.conv0_1(shortcut)
        shortcut.features = self.act0_1(shortcut.features)
        shortcut.features = self.bn0_1(shortcut.features)

        res = self.conv1_0(x)
        res.features = self.act1_0(res.features)
        res.features = self.bn1_0(res.features)

        res = self.conv1_1(res)
        res.features = self.act1_1(res.features)
        res.features = self.bn1_1(res.features)

        res.features = res.features + shortcut.features

        if self.pooling:
            pooled_res = self.pool(res)
            return pooled_res, res
        else:
            return res


class AsymmeUpBlock(BaseModule):
    """Asymmetrical UpSample Block.

    Args:
        in_channels (int): Input channels of the block.
        out_channels (int): Output channels of the block.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for
                normalization layer.
        act_cfg (:obj:`ConfigDict` or dict): Config dict of activation layers.
                Defaults to dict(type='LeakyReLU').
        indice_key (str, optional): Name of indice tables. Defaults to None.
        up_key (str, optional): Name of indice tables used in
            SparseInverseConv3d. Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_cfg: ConfigType,
                 act_cfg: ConfigType = dict(type='LeakyReLU'),
                 indice_key: Optional[str] = None,
                 up_key: Optional[str] = None):
        super().__init__()

        self.trans_conv = SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + 'new_up')
        self.trans_act = build_activation_layer(act_cfg)
        self.trans_bn = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv1 = SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            padding=1,
            bias=False,
            indice_key=indice_key)
        self.act1 = build_activation_layer(act_cfg)
        self.bn1 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv2 = SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 1, 3),
            padding=1,
            bias=False,
            indice_key=indice_key)
        self.act2 = build_activation_layer(act_cfg)
        self.bn2 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv3 = SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            padding=1,
            bias=False,
            indice_key=indice_key)
        self.act3 = build_activation_layer(act_cfg)
        self.bn3 = build_norm_layer(norm_cfg, out_channels)[1]

        self.up_subm = SparseInverseConv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            indice_key=up_key,
            bias=False)

    def forward(self, x: SparseConvTensor,
                skip: SparseConvTensor) -> SparseConvTensor:
        """Forward pass."""
        x_trans = self.trans_conv(x)
        x_trans.features = self.trans_act(x_trans.features)
        x_trans.features = self.trans_bn(x_trans.features)

        # upsample
        up = self.up_subm(x_trans)

        up.features = up.features + skip.features

        up = self.conv1(up)
        up.features = self.act1(up.features)
        up.features = self.bn1(up.features)

        up = self.conv2(up)
        up.features = self.act2(up.features)
        up.features = self.bn2(up.features)

        up = self.conv3(up)
        up.features = self.act3(up.features)
        up.features = self.bn3(up.features)

        return up


class DDCMBlock(BaseModule):
    """Dimension-Decomposition based Context Modeling.

    Args:
        in_channels (int): Input channels of the block.
        out_channels (int): Output channels of the block.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for
            normalization layer.
        act_cfg (:obj:`ConfigDict` or dict): Config dict of activation layers.
            Defaults to dict(type='Sigmoid').
        indice_key (str, optional): Name of indice tables. Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_cfg: ConfigType,
                 act_cfg: ConfigType = dict(type='Sigmoid'),
                 indice_key: Optional[str] = None):
        super().__init__()

        self.conv1 = SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 1, 1),
            padding=1,
            bias=False,
            indice_key=indice_key)
        self.bn1 = build_norm_layer(norm_cfg, out_channels)[1]
        self.act1 = build_activation_layer(act_cfg)

        self.conv2 = SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 1),
            padding=1,
            bias=False,
            indice_key=indice_key)
        self.bn2 = build_norm_layer(norm_cfg, out_channels)[1]
        self.act2 = build_activation_layer(act_cfg)

        self.conv3 = SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 1, 3),
            padding=1,
            bias=False,
            indice_key=indice_key)
        self.bn3 = build_norm_layer(norm_cfg, out_channels)[1]
        self.act3 = build_activation_layer(act_cfg)

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        """Forward pass."""
        shortcut = self.conv1(x)
        shortcut.features = self.bn1(shortcut.features)
        shortcut.features = self.act1(shortcut.features)

        shortcut2 = self.conv2(x)
        shortcut2.features = self.bn2(shortcut2.features)
        shortcut2.features = self.act2(shortcut2.features)

        shortcut3 = self.conv3(x)
        shortcut3.features = self.bn3(shortcut3.features)
        shortcut3.features = self.act3(shortcut3.features)
        shortcut.features = shortcut.features + \
            shortcut2.features + shortcut3.features

        shortcut.features = shortcut.features * x.features

        return shortcut


@MODELS.register_module()
class Asymm3DSpconv(BaseModule):
    """Asymmetrical 3D convolution networks.

    Args:
        grid_size (int): Size of voxel grids.
        input_channels (int): Input channels of the block.
        base_channels (int): Initial size of feature channels before
            feeding into Encoder-Decoder structure. Defaults to 16.
        backbone_depth (int): The depth of backbone. The backbone contains
            downblocks and upblocks with the number of backbone_depth.
        height_pooing (List[bool]): List indicating which downblocks perform
            height pooling.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01)).
        init_cfg (dict, optional): Initialization config.
            Defaults to None.
    """

    def __init__(self,
                 grid_size: int,
                 input_channels: int,
                 base_channels: int = 16,
                 backbone_depth: int = 4,
                 height_pooing: List[bool] = [True, True, False, False],
                 norm_cfg: ConfigType = dict(
                     type='BN1d', eps=1e-3, momentum=0.01),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.grid_size = grid_size
        self.backbone_depth = backbone_depth
        self.down_context = AsymmResBlock(
            input_channels, base_channels, indice_key='pre', norm_cfg=norm_cfg)

        self.down_block_list = torch.nn.ModuleList()
        self.up_block_list = torch.nn.ModuleList()
        for i in range(self.backbone_depth):
            self.down_block_list.append(
                AsymmeDownBlock(
                    2**i * base_channels,
                    2**(i + 1) * base_channels,
                    height_pooling=height_pooing[i],
                    indice_key='down' + str(i),
                    norm_cfg=norm_cfg))
            if i == self.backbone_depth - 1:
                self.up_block_list.append(
                    AsymmeUpBlock(
                        2**(i + 1) * base_channels,
                        2**(i + 1) * base_channels,
                        up_key='down' + str(i),
                        indice_key='up' + str(self.backbone_depth - 1 - i),
                        norm_cfg=norm_cfg))
            else:
                self.up_block_list.append(
                    AsymmeUpBlock(
                        2**(i + 2) * base_channels,
                        2**(i + 1) * base_channels,
                        up_key='down' + str(i),
                        indice_key='up' + str(self.backbone_depth - 1 - i),
                        norm_cfg=norm_cfg))

        self.ddcm = DDCMBlock(
            2 * base_channels,
            2 * base_channels,
            indice_key='ddcm',
            norm_cfg=norm_cfg)

    def forward(self, voxel_features: Tensor, coors: Tensor,
                batch_size: int) -> SparseConvTensor:
        """Forward pass."""
        coors = coors.int()
        ret = SparseConvTensor(voxel_features, coors, np.array(self.grid_size),
                               batch_size)
        ret = self.down_context(ret)

        down_skip_list = []
        down_pool = ret
        for i in range(self.backbone_depth):
            down_pool, down_skip = self.down_block_list[i](down_pool)
            down_skip_list.append(down_skip)

        up = down_pool
        for i in range(self.backbone_depth - 1, -1, -1):
            up = self.up_block_list[i](up, down_skip_list[i])

        ddcm = self.ddcm(up)
        ddcm.features = torch.cat((ddcm.features, up.features), 1)

        return ddcm
