import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import (TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from mmengine.model import BaseModule, constant_init, xavier_init

from mmdet3d.registry import MODELS


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@MODELS.register_module()
class Detr3DTransformer(BaseModule):
    """Implements the DETR3D transformer.

    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        num_cams (int): Number of cameras in the dataset.
            Default: 6 in NuScenes Det.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 decoder=None,
                 **kwargs):
        super(Detr3DTransformer, self).__init__(**kwargs)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.reference_points = nn.Linear(self.embed_dims, 3)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention) or isinstance(
                    m, Detr3DCrossAtten):
                m.init_weight()
        xavier_init(self.reference_points, distribution='uniform', bias=0.)

    def forward(self, mlvl_feats, query_embed, reg_branches=None, **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                (B, N, C, H_lvl, W_lvl).
            query_embed (Tensor): The query positional and semantic embedding
                for decoder, with shape [num_query, c+c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                [bs, N, embed_dims, h, w]. It is unused here.
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape
                      (num_dec_layers, bs, num_query, embed_dims), else has
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference
                    points in decoder, has shape
                    (num_dec_layers, bs, num_query, embed_dims)
        """
        assert query_embed is not None
        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)  # [bs,num_q,c]
        query = query.unsqueeze(0).expand(bs, -1, -1)  # [bs,num_q,c]
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        # decoder
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=mlvl_feats,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            **kwargs)

        inter_references_out = inter_references
        return inter_states, init_reference_out, inter_references_out


@MODELS.register_module()
class Detr3DTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.

    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default:
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(Detr3DTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

    def forward(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                **kwargs):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape self.reference_points =
                                        nn.Linear(self.embed_dims, 3)
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):  # iterative refinement
            reference_points_input = reference_points
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                **kwargs)
            output = output.permute(1, 0, 2)
            if reg_branches is not None:
                tmp = reg_branches[lid](output)

                assert reference_points.shape[-1] == 3

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(
                    reference_points[..., :2])
                new_reference_points[...,
                                     2:3] = tmp[..., 4:5] + inverse_sigmoid(
                                         reference_points[..., 2:3])
                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


@MODELS.register_module()
class Detr3DCrossAtten(BaseModule):
    """An attention module used in Detr3d.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=5,
        num_cams=6,
        im2col_step=64,
        pc_range=None,
        dropout=0.1,
        norm_cfg=None,
        init_cfg=None,
        batch_first=False,
    ):
        super(Detr3DCrossAtten, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_cams = num_cams
        self.attention_weights = nn.Linear(embed_dims,
                                           num_cams * num_levels * num_points)

        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.batch_first = batch_first
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                reference_points=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (List[Tensor]): Image features from
                different level. Each element has shape
                (B, N, C, H_lvl, W_lvl).
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            reference_points (Tensor): The normalized 3D reference
                points with shape (bs, num_query, 3)
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        query = query.permute(1, 0, 2)

        bs, num_query, _ = query.size()

        attention_weights = self.attention_weights(query).view(
            bs, 1, num_query, self.num_cams, self.num_points, self.num_levels)
        reference_points_3d, output, mask = feature_sampling(
            value, reference_points, self.pc_range, kwargs['img_metas'])
        output = torch.nan_to_num(output)
        mask = torch.nan_to_num(mask)
        attention_weights = attention_weights.sigmoid() * mask
        output = output * attention_weights
        output = output.sum(-1).sum(-1).sum(-1)
        output = output.permute(2, 0, 1)
        # (num_query, bs, embed_dims)
        output = self.output_proj(output)
        pos_feat = self.position_encoder(
            inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)
        return self.dropout(output) + inp_residual + pos_feat


def feature_sampling(mlvl_feats,
                     ref_pt,
                     pc_range,
                     img_metas,
                     no_sampling=False):
    """ sample multi-level features by projecting 3D reference points
            to 2D image
        Args:
            mlvl_feats (List[Tensor]): Image features from
                different level. Each element has shape
                (B, N, C, H_lvl, W_lvl).
            ref_pt (Tensor): The normalized 3D reference
                points with shape (bs, num_query, 3)
            pc_range: perception range of the detector
            img_metas (list[dict]): Meta information of multiple inputs
                in a batch, containing `lidar2img`.
            no_sampling (bool): If set 'True', the function will return
                2D projected points and mask only.
        Returns:
            ref_pt_3d (Tensor): A copy of original ref_pt
            sampled_feats (Tensor): sampled features with shape \
                (B C num_q N 1 fpn_lvl)
            mask (Tensor): Determine whether the reference point \
                has projected outsied of images, with shape \
                (B 1 num_q N 1 1)
    """
    lidar2img = [meta['lidar2img'] for meta in img_metas]
    lidar2img = np.asarray(lidar2img)
    lidar2img = ref_pt.new_tensor(lidar2img)
    ref_pt = ref_pt.clone()
    ref_pt_3d = ref_pt.clone()

    B, num_query = ref_pt.size()[:2]
    num_cam = lidar2img.size(1)
    eps = 1e-5

    ref_pt[..., 0:1] = \
        ref_pt[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]  # x
    ref_pt[..., 1:2] = \
        ref_pt[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]  # y
    ref_pt[..., 2:3] = \
        ref_pt[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]  # z

    # (B num_q 3) -> (B num_q 4) -> (B 1 num_q 4) -> (B num_cam num_q 4 1)
    ref_pt = torch.cat((ref_pt, torch.ones_like(ref_pt[..., :1])), -1)
    ref_pt = ref_pt.view(B, 1, num_query, 4)
    ref_pt = ref_pt.repeat(1, num_cam, 1, 1).unsqueeze(-1)
    # (B num_cam 4 4) -> (B num_cam num_q 4 4)
    lidar2img = lidar2img.view(B, num_cam, 1, 4, 4)\
                         .repeat(1, 1, num_query, 1, 1)
    # (... 4 4) * (... 4 1) -> (B num_cam num_q 4)
    pt_cam = torch.matmul(lidar2img, ref_pt).squeeze(-1)

    # (B num_cam num_q)
    z = pt_cam[..., 2:3]
    eps = eps * torch.ones_like(z)
    mask = (z > eps)
    pt_cam = pt_cam[..., 0:2] / torch.maximum(z, eps)  # prevent zero-division
    # padded nuscene image: 928*1600
    (h, w) = img_metas[0]['pad_shape']
    pt_cam[..., 0] /= w
    pt_cam[..., 1] /= h
    # else:
    # (h,w,_) = img_metas[0]['ori_shape'][0]          # waymo image
    # pt_cam[..., 0] /= w # cam0~2: 1280*1920
    # pt_cam[..., 1] /= h # cam3~4: 886 *1920 padded to 1280*1920
    # mask[:, 3:5, :] &= (pt_cam[:, 3:5, :, 1:2] < 0.7) # filter pt_cam_y > 886

    mask = (
        mask & (pt_cam[..., 0:1] > 0.0)
        & (pt_cam[..., 0:1] < 1.0)
        & (pt_cam[..., 1:2] > 0.0)
        & (pt_cam[..., 1:2] < 1.0))

    if no_sampling:
        return pt_cam, mask

    # (B num_cam num_q) -> (B 1 num_q num_cam 1 1)
    mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)
    mask = torch.nan_to_num(mask)

    pt_cam = (pt_cam - 0.5) * 2  # [0,1] to [-1,1] to do grid_sample
    sampled_feats = []
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size()
        feat = feat.view(B * N, C, H, W)
        pt_cam_lvl = pt_cam.view(B * N, num_query, 1, 2)
        sampled_feat = F.grid_sample(feat, pt_cam_lvl)
        # (B num_cam C num_query 1) -> List of (B C num_q num_cam 1)
        sampled_feat = sampled_feat.view(B, N, C, num_query, 1)
        sampled_feat = sampled_feat.permute(0, 2, 3, 1, 4)
        sampled_feats.append(sampled_feat)

    sampled_feats = torch.stack(sampled_feats, -1)
    # (B C num_q num_cam fpn_lvl)
    sampled_feats = \
        sampled_feats.view(B, C, num_query, num_cam, 1, len(mlvl_feats))
    return ref_pt_3d, sampled_feats, mask
