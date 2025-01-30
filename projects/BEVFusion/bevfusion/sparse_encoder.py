# Copyright (c) OpenMMLab. All rights reserved.
from mmdet3d.models.layers import make_sparse_convmodule
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
from mmdet3d.models.middle_encoders import SparseEncoder
from mmdet3d.registry import MODELS
from projects.BEVFusion.bevfusion.sam_supervised_utils import *
if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor
else:
    from mmcv.ops import SparseConvTensor


@MODELS.register_module()
class BEVFusionSparseEncoder(SparseEncoder):
    r"""Sparse encoder for BEVFusion. The difference between this
    implementation and that of ``SparseEncoder`` is that the shape order of 3D
    conv is (H, W, D) in ``BEVFusionSparseEncoder`` rather than (D, H, W) in
    ``SparseEncoder``. This difference comes from the implementation of
    ``voxelization``.

    Args:
        in_channels (int): The number of input channels.
        sparse_shape (list[int]): The sparse shape of input tensor.
        order (list[str], optional): Order of conv module.
            Defaults to ('conv', 'norm', 'act').
        norm_cfg (dict, optional): Config of normalization layer. Defaults to
            dict(type='BN1d', eps=1e-3, momentum=0.01).
        base_channels (int, optional): Out channels for conv_input layer.
            Defaults to 16.
        output_channels (int, optional): Out channels for conv_out layer.
            Defaults to 128.
        encoder_channels (tuple[tuple[int]], optional):
            Convolutional channels of each encode block.
            Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
        encoder_paddings (tuple[tuple[int]], optional):
            Paddings of each encode block.
            Defaults to ((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)).
        block_type (str, optional): Type of the block to use.
            Defaults to 'conv_module'.
        return_middle_feats (bool): Whether output middle features.
            Default to False.
    """

    def __init__(self,
                 in_channels,
                 sparse_shape,
                 order=('conv', 'norm', 'act'),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 base_channels=16,
                 output_channels=128,
                 encoder_channels=((16, ), (32, 32, 32), (64, 64, 64), (64, 64,
                                                                        64)),
                 encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1,
                                                                 1)),
                 block_type='conv_module',
                 return_middle_feats=False):
        super(SparseEncoder, self).__init__()
        assert block_type in ['conv_module', 'basicblock']
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.stage_num = len(self.encoder_channels)
        self.fp16_enabled = False
        self.return_middle_feats = return_middle_feats
        # Spconv init all weight on its own

        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == {'conv', 'norm', 'act'}

        if self.order[0] != 'conv':  # pre activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d',
                order=('conv', ))
        else:  # post activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d')

        encoder_out_channels = self.make_encoder_layers(
            make_sparse_convmodule,
            norm_cfg,
            self.base_channels,
            block_type=block_type)

        self.conv_out = make_sparse_convmodule(
            encoder_out_channels,
            self.output_channels,
            kernel_size=(1, 1, 3),
            stride=(1, 1, 2),
            norm_cfg=norm_cfg,
            padding=0,
            indice_key='spconv_down2',
            conv_type='SparseConv3d')

    def forward(self, voxel_features, coors, batch_size):
        """Forward of SparseEncoder.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, C).
            coors (torch.Tensor): Coordinates in shape (N, 4),
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            torch.Tensor | tuple[torch.Tensor, list]: Return spatial features
                include:

            - spatial_features (torch.Tensor): Spatial features are out from
                the last layer.
            - encode_features (List[SparseConvTensor], optional): Middle layer
                output features. When self.return_middle_feats is True, the
                module returns middle features.
        """
        coors = coors.int()
        input_sp_tensor = SparseConvTensor(voxel_features, coors,
                                           self.sparse_shape, batch_size)
        x = self.conv_input(input_sp_tensor)

        x_temp = x
        
        # save_voxel_centers(x.indices[:,1:4], "/data/otherg/mmdetection3d/wztcode/save_ply/temp1.ply")
        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)

        # flag = check_indices_match(x_temp.indices, x.indices)
        # print("flag:", flag)
        # print(count_matching_rows(x_temp.indices, x.indices))

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]


        # save_voxel_centers(encode_features[-1].indices[:,1:4], "/data/otherg/mmdetection3d/wztcode/save_ply/temp2.ply")
        # return_feature = replace_indices_to_input_coordinates(x.indices, encode_features)
        out = self.conv_out(encode_features[-1])
        spatial_features = out.dense()  #wzt : dense() is used to convert the sparse tensor to dense tensor, Flatten

        N, C, H, W, D = spatial_features.shape
        spatial_features = spatial_features.permute(0, 1, 4, 2, 3).contiguous()
        spatial_features = spatial_features.view(N, C * D, H, W)

        if self.return_middle_feats:
            return spatial_features, encode_features
        else:
            return spatial_features, out
