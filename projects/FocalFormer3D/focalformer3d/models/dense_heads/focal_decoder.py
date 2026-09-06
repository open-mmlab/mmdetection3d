# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates.
# All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FocalFormer3D/blob/main/LICENSE

import copy

# Importing mmdet.models registers FocalLoss, L1Loss and the other losses the
# config refers to with the 'mmdet.' scope prefix.
import mmdet.models  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmdet.models.layers import (DeformableDetrTransformerDecoder,
                                 DeformableDetrTransformerDecoderLayer,
                                 DetrTransformerDecoderLayer)
from mmdet.models.task_modules.assigners import AssignResult
from mmdet.models.task_modules.samplers import PseudoSampler
from mmdet.models.utils import multi_apply
from torch import nn

from mmdet3d.models.layers import circle_nms, nms_bev
from mmdet3d.models.utils import clip_sigmoid
from mmdet3d.models.utils.gaussian import (draw_heatmap_gaussian,
                                           gaussian_radius)
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures import xywhr2xyxyr
from mmdet3d.structures.bbox_3d.utils import rotation_3d_in_axis
from ..utils.decoder_utils import FFN
from ..utils.utils import MLP, gen_sineembed_for_position

# mmdet's DeformableDetrTransformerDecoder and DetrTransformerDecoderLayer are
# not decorated with @MODELS.register_module(), so build_transformer_layer_
# sequence cannot resolve them from a config type string. Register them here.
for _cls in [
        DeformableDetrTransformerDecoder,
        DeformableDetrTransformerDecoderLayer,
        DetrTransformerDecoderLayer,
]:
    if MODELS.get(_cls.__name__) is None:
        MODELS.register_module(module=_cls)


@MODELS.register_module()
class FocalDecoder(nn.Module):

    def __init__(self,
                 num_proposals=128,
                 hidden_channel=128,
                 hidden_channel_roi=512,
                 num_classes=4,
                 num_decoder_layers=1,
                 num_heads=8,
                 initialize_by_heatmap=False,
                 nms_kernel_size=1,
                 bn_momentum=0.1,
                 activation='relu',
                 classaware_reg=False,
                 common_heads=dict(),
                 num_heatmap_convs=2,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 bias='auto',
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_bbox=dict(type='L1Loss', reduction='mean'),
                 loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_weight_heatmap=1.,
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder=None,
                 num_stage_proposals=None,
                 multiscale=False,
                 multistage_heatmap=False,
                 reuse_first_heatmap=False,
                 extra_feat=False,
                 heatmap_box=False,
                 thin_heatmap_box=False,
                 loss_weight_separate_heatmap=0.2,
                 loss_weight_separate_bbox=0.5,
                 boxpos=None,
                 add_gt_groups=0,
                 add_gt_groups_noise='rect,1',
                 add_gt_groups_noise_box='gt',
                 gt_center_limit=None,
                 add_gt_pos_thresh=100.,
                 add_gt_pos_boxnoise_thresh=2.,
                 gt_query_loss_weight=1.,
                 bevpos=False,
                 input_img=True,
                 iterbev_wo_img=False,
                 mask_heatmap_mode='poscls',
                 roi_feats=0,
                 roi_dropout_rate=0.,
                 roi_expand_ratio=1.,
                 roi_based_reg=False,
                 decoder_cfg=dict(
                     type='DeformableDetrTransformerDecoder',
                     num_layers=6,
                     return_intermediate=False,
                     transformerlayers=dict(
                         type='DetrTransformerDecoderLayer',
                         attn_cfgs=[
                             dict(
                                 type='MultiheadAttention',
                                 embed_dims=128,
                                 num_heads=8,
                                 dropout=0.1),
                             dict(
                                 type='MultiScaleDeformableAttention',
                                 embed_dims=128,
                                 num_levels=1,
                                 num_points=6,
                                 num_heads=8,
                                 renorm_z=5.)
                         ],
                         feedforward_channels=1024,
                         ffn_dropout=0.1,
                         ffn_cfgs=dict(
                             type='FFN',
                             embed_dims=128,
                             num_fcs=2,
                             act_cfg=dict(type='ReLU', inplace=True)),
                         operation_order=('self_attn', 'norm', 'cross_attn',
                                          'norm', 'ffn', 'norm')))):
        super(FocalDecoder, self).__init__()

        self.num_classes = num_classes
        self.num_proposals_ori = self.num_proposals = num_proposals
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.bn_momentum = bn_momentum
        self.initialize_by_heatmap = initialize_by_heatmap
        self.nms_kernel_size = nms_kernel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if num_stage_proposals is None:
            self.num_stage_proposals = [self.num_proposals
                                        ] * self.num_decoder_layers

        else:
            self.num_stage_proposals = num_stage_proposals
        self.cumsum_proposals = np.asarray(
            [0] + list(np.cumsum(self.num_stage_proposals)))
        self.multiscale = multiscale
        self.multistage_heatmap = multistage_heatmap
        self.reuse_first_heatmap = reuse_first_heatmap
        self.extra_feat = extra_feat

        if self.reuse_first_heatmap:
            self.multistage_heatmap += 1
        self.boxpos = boxpos
        self.gt_query_loss_weight = gt_query_loss_weight
        self.bevpos = bevpos
        self.input_img = input_img
        self.iterbev_wo_img = iterbev_wo_img
        self.heatmap_box = heatmap_box
        self.thin_heatmap_box = thin_heatmap_box
        self.loss_weight_heatmap = loss_weight_heatmap
        self.loss_weight_separate_heatmap = loss_weight_separate_heatmap
        self.loss_weight_separate_bbox = loss_weight_separate_bbox

        if self.multiscale:
            self.dconv = ConvModule(
                hidden_channel,
                hidden_channel,
                stride=2,
                kernel_size=3,
                padding=1,
                bias=bias,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'))
            self.dconv2 = ConvModule(
                hidden_channel,
                hidden_channel,
                stride=2,
                kernel_size=3,
                padding=1,
                bias=bias,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'))

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)

        if not self.use_sigmoid_cls:
            self.num_classes += 1

        # build_loss → MODELS.build
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_heatmap = MODELS.build(loss_heatmap)
        self.gt_center_limit = gt_center_limit
        self.add_gt_pos_thresh = add_gt_pos_thresh
        self.add_gt_pos_boxnoise_thresh = add_gt_pos_boxnoise_thresh

        # build_bbox_coder → TASK_UTILS.build
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.mask_heatmap_mode = mask_heatmap_mode

        self.roi_feats = roi_feats

        if not roi_feats:

            assert not roi_based_reg

        self.roi_based_reg = roi_based_reg

        if isinstance(roi_expand_ratio, float):
            self.roi_expand_ratio = [roi_expand_ratio
                                     ] * self.num_decoder_layers

        else:
            self.roi_expand_ratio = roi_expand_ratio

        if self.roi_feats:
            fc_list = []
            pre_channel = self.roi_feats**2 * hidden_channel * (
                3 if self.multiscale else 1)
            num_roi_layers = 3

            for i in range(num_roi_layers):
                chl = (
                    hidden_channel_roi
                    if i < num_roi_layers - 1 else hidden_channel)
                fc_list.extend([
                    nn.Linear(pre_channel, chl, bias=False),
                    nn.BatchNorm1d(chl),
                    nn.ReLU(inplace=True)
                ])
                if roi_dropout_rate > 1e-4 and i != -1:
                    fc_list.append(nn.Dropout(roi_dropout_rate))
                pre_channel = chl
            self.roi_mlp = nn.Sequential(*fc_list)

        if self.initialize_by_heatmap:
            layers = []
            layers.append(
                ConvModule(
                    hidden_channel,
                    hidden_channel,
                    kernel_size=3,
                    padding=1,
                    bias=bias,
                    conv_cfg=dict(type='Conv2d'),
                    norm_cfg=dict(type='BN2d')))
            layers.append(
                build_conv_layer(
                    dict(type='Conv2d'),
                    hidden_channel,
                    num_classes,
                    kernel_size=3,
                    padding=1,
                    bias=bias))
            self.heatmap_head = nn.Sequential(*layers)

            if self.input_img or self.iterbev_wo_img:
                if self.multistage_heatmap:
                    self.heatmap_head_img = nn.ModuleList()
                    for i in range(self.multistage_heatmap):
                        if i == 0 and self.reuse_first_heatmap:
                            self.heatmap_head_img.append(None)
                        else:
                            self.heatmap_head_img.append(
                                copy.deepcopy(self.heatmap_head))

                    if self.heatmap_box:
                        self.multi_stage_task_heads = nn.ModuleList()
                        self.heatmap_tasks = [
                            dict(num_class=1, class_names=['car']),
                            dict(
                                num_class=2,
                                class_names=['truck', 'construction_vehicle']),
                            dict(num_class=2, class_names=['bus', 'trailer']),
                            dict(num_class=1, class_names=['barrier']),
                            dict(
                                num_class=2,
                                class_names=['motorcycle', 'bicycle']),
                            dict(
                                num_class=2,
                                class_names=['pedestrian', 'traffic_cone']),
                        ]
                        self.class_names = [
                            i['class_names'] for i in self.heatmap_tasks
                        ]
                        if self.test_cfg['dataset'] == 'Waymo':
                            heatmap_common_heads = {
                                'reg': (2, 2),
                                'height': (1, 2),
                                'dim': (3, 2),
                                'rot': (2, 2)
                            }
                        else:
                            heatmap_common_heads = {
                                'reg': (2, 2),
                                'height': (1, 2),
                                'dim': (3, 2),
                                'rot': (2, 2),
                                'vel': (2, 2)
                            }
                        separate_num_classes = [1, 2, 2, 1, 2, 2]
                        separate_head = {
                            'type': 'DCNSeparateHead',
                            'init_bias': -2.19,
                            'final_kernel': 3,
                            'dcn_config': {
                                'type': 'DCN',
                                'in_channels': hidden_channel,
                                'out_channels': hidden_channel,
                                'kernel_size': 3,
                                'padding': 1,
                                'groups': 4
                            },
                            'in_channels': 64,
                            'heads': heatmap_common_heads,
                            'num_cls': 2
                        }
                        if self.train_cfg is not None:
                            self.train_cfg['max_objs'] = 500
                            self.train_cfg['dense_reg'] = 1
                        self.norm_bbox = True
                        # build_loss → MODELS.build
                        self.heatmap_loss_cls = MODELS.build({
                            'type':
                            'mmdet.GaussianFocalLoss',
                            'reduction':
                            'mean'
                        })
                        self.heatmap_loss_bbox = MODELS.build({
                            'type':
                            'mmdet.L1Loss',
                            'reduction':
                            'mean',
                            'loss_weight':
                            0.25
                        })
                        for i in range(self.multistage_heatmap):
                            task_heads = nn.ModuleList()
                            if self.thin_heatmap_box:
                                layers = []
                                layers.append(
                                    ConvModule(
                                        hidden_channel,
                                        hidden_channel,
                                        kernel_size=3,
                                        padding=1,
                                        bias=bias,
                                        conv_cfg=dict(type='Conv2d'),
                                        norm_cfg=dict(type='BN2d')))
                                layers.append(
                                    build_conv_layer(
                                        dict(type='Conv2d'),
                                        hidden_channel,
                                        len(separate_num_classes) * 10,
                                        kernel_size=3,
                                        padding=1,
                                        bias=bias))
                                task_heads = nn.Sequential(*layers)
                            else:
                                for num_cls in separate_num_classes:
                                    heads = copy.deepcopy(heatmap_common_heads)
                                    heads.update(
                                        dict(
                                            heatmap=(num_cls,
                                                     num_heatmap_convs)))
                                    separate_head.update(
                                        in_channels=hidden_channel,
                                        heads=heads,
                                        num_cls=num_cls)
                                    # builder.build_head → MODELS.build
                                    task_heads.append(
                                        MODELS.build(separate_head))
                            self.multi_stage_task_heads.append(task_heads)
                else:
                    self.heatmap_head_img = copy.deepcopy(self.heatmap_head)
            self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)

        else:
            self.query_feat = nn.Parameter(
                torch.randn(1, hidden_channel, self.num_proposals))
            self.query_pos = nn.Parameter(
                torch.rand([1, self.num_proposals, 2]), requires_grad=False)

        # transformer decoder layers
        self.decoder = nn.ModuleList()
        self.decoder_cfg = decoder_cfg
        self.pos_embed_learned = nn.ModuleList()
        self.box_pos_embed_learned = nn.ModuleList()
        self.inter_reference_reg_branches = nn.ModuleList()

        for i in range(self.num_decoder_layers):
            self.decoder.append(
                build_transformer_layer_sequence(self.decoder_cfg))
            self.pos_embed_learned.append(
                MLP(256, hidden_channel, hidden_channel, 2))

            if self.boxpos == 'xywlr':
                if i > 0:
                    self.box_pos_embed_learned.append(
                        MLP(128 * 5, hidden_channel, hidden_channel, 2))
                else:
                    self.box_pos_embed_learned.append(
                        None if not self.heatmap_box else MLP(
                            128 * 5, hidden_channel, hidden_channel, 2))

        self.classaware_reg = classaware_reg
        self.prediction_heads = nn.ModuleList()

        for i in range(self.num_decoder_layers):
            heads = copy.deepcopy(common_heads)

            if self.classaware_reg:
                for k, v in heads.items():
                    heads[k] = [v[0] * self.num_classes, v[1]]
            heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
            self.prediction_heads.append(
                FFN(hidden_channel,
                    heads,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    bias=bias))

        x_size = self.test_cfg['grid_size'][0] // self.test_cfg[
            'out_size_factor']

        y_size = self.test_cfg['grid_size'][1] // self.test_cfg[
            'out_size_factor']

        self.bev_pos = self.create_2D_grid(x_size, y_size)
        self.img_feat_pos = None
        self.img_feat_collapsed_pos = None
        self.add_gt_groups = add_gt_groups
        self.add_gt_groups_noise = add_gt_groups_noise
        self.add_gt_groups_noise_box = add_gt_groups_noise_box
        self.init_weights()
        self._init_assigner_sampler()

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]

        batch_y, batch_x = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])

        batch_x = batch_x + 0.5

        batch_y = batch_y + 0.5

        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)

        return coord_base

    def init_weights(self):

        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)

        if hasattr(self, 'query'):
            nn.init.xavier_normal_(self.query)

        if hasattr(self,
                   'multi_stage_task_heads') and not self.thin_heatmap_box:
            for s in self.multi_stage_task_heads:
                for t in s:
                    t.init_weights()

        self.init_bn_momentum()

    def init_bn_momentum(self):

        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def _init_assigner_sampler(self):
        if self.train_cfg is None:
            return

        self.bbox_sampler = PseudoSampler()

        # build_assigner → TASK_UTILS.build
        if isinstance(self.train_cfg.assigner, dict):
            self.bbox_assigner = TASK_UTILS.build(self.train_cfg.assigner)

        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                TASK_UTILS.build(res) for res in self.train_cfg.assigner
            ]

    # generate_gt_groups is algorithmically unchanged from original
    # Only change: uses self.bbox_coder (already built via TASK_UTILS.build)

    def generate_gt_groups(self,
                           query_feat,
                           query_pos,
                           query_heatmap_score,
                           lidar_feat,
                           lidar_feat_flatten,
                           bev_pos,
                           heatmap,
                           gt_bboxes_3d,
                           gt_labels_3d,
                           dense_heatmap_boxes=None,
                           query_box=None):
        batch_size = len(gt_bboxes_3d)

        point_cloud_range = torch.as_tensor(
            self.train_cfg['point_cloud_range'], device='cuda')

        batch_valid_gt_mask = torch.zeros(
            (batch_size, self.max_num_gts * self.add_gt_groups),
            dtype=bool,
            device=query_pos.device)

        for batch_idx in range(batch_size):
            for group_i in range(self.add_gt_groups):
                batch_valid_gt_mask[batch_idx,
                                    group_i * self.max_num_gts:group_i *
                                    self.max_num_gts +
                                    self.num_gts[batch_idx]] = 1

        batch_gt_pos = []

        batch_gt_query_labels = []

        batch_gt_bboxes_3d = []

        for batch_idx in range(batch_size):
            gs_box = gt_bboxes_3d[batch_idx].tensor.cuda()
            gs = F.pad(
                gs_box[:, :2],
                pad=(0, 0, 0, self.max_num_gts - self.num_gts[batch_idx]))
            gs_labels = F.pad(
                gt_labels_3d[batch_idx],
                pad=(0, self.max_num_gts - self.num_gts[batch_idx]),
                value=self.num_classes)
            gs_bev_corners = F.pad(
                gt_bboxes_3d[batch_idx].corners.cuda().reshape(-1, 4, 2,
                                                               3)[:, :4,
                                                                  0, :2],
                pad=(0, 0, 0, 0, 0,
                     self.max_num_gts - self.num_gts[batch_idx]))
            gs = gs.repeat(self.add_gt_groups, 1)
            gs_labels = gs_labels.repeat(self.add_gt_groups)
            gs_bev_corners = gs_bev_corners.repeat(self.add_gt_groups, 1, 1)
            gs_box = self.bbox_coder.encode(gs_box)
            gs_box = F.pad(
                gs_box,
                pad=(0, 0, 0, self.max_num_gts - self.num_gts[batch_idx]))
            gs_box = gs_box.repeat(self.add_gt_groups, 1)
            meta_noise = (
                torch.rand((len(gs_bev_corners), 2), device='cuda') * 2 - 1)

            if self.add_gt_groups_noise.startswith('rect'):
                gs_bev_corners_rect = torch.cat([
                    gs_bev_corners.min(dim=1)[0],
                    gs_bev_corners.max(dim=1)[0]
                ],
                                                dim=1)
                noise = float(
                    self.add_gt_groups_noise.split(',')[1]) * meta_noise
                box_center_noise = (gs_bev_corners_rect[:, 2:4] -
                                    gs_bev_corners_rect[:, 0:2]) / 2. * noise
                gs = gs + box_center_noise

            elif self.add_gt_groups_noise.startswith('cam'):
                gs_vec = gs / (gs.norm(dim=1) + 1e-6)[:, None]
                gs_vec_orth = torch.stack([gs_vec[:, 1], -gs_vec[:, 0]], dim=1)
                gs_mat = torch.stack([gs_vec, gs_vec_orth], dim=-1)
                gs_bev_corners_trans = gs_bev_corners.matmul(gs_mat)
                gs_bev_corners_rect = torch.cat([
                    gs_bev_corners_trans.min(dim=1)[0],
                    gs_bev_corners_trans.max(dim=1)[0]
                ],
                                                dim=1)
                scale = self.add_gt_groups_noise.split(',')[1]
                if '-' in scale:
                    scale = torch.as_tensor(
                        list(map(float, scale.split('-'))),
                        device='cuda')[None]
                else:
                    scale = float(scale)
                noise = scale * meta_noise
                box_center_noise = (gs_bev_corners_rect[:, 2:4] -
                                    gs_bev_corners_rect[:, 0:2]) / 2. * noise
                gs = gs + box_center_noise[:, None].matmul(gs_mat)[:, 0]

            elif self.add_gt_groups_noise.startswith('box'):
                pseudo_w = gs_bev_corners[:, 2] - gs_bev_corners[:, 0]
                pseudo_h = gs_bev_corners[:, 1] - gs_bev_corners[:, 0]
                noise = float(
                    self.add_gt_groups_noise.split(',')[1]) * meta_noise
                box_center_noise = (pseudo_w / 2. * noise[:, 0:1]) + (
                    pseudo_h / 2. * noise[:, 1:2])
                gs = gs + box_center_noise

            else:
                raise NotImplementedError
            gs[:, 0] = gs[:, 0].clip(
                min=point_cloud_range[0] + 1e-6,
                max=point_cloud_range[3] - 1e-5)
            gs[:, 1] = gs[:, 1].clip(
                min=point_cloud_range[1] + 1e-6,
                max=point_cloud_range[4] - 1e-5)
            gs = (gs - point_cloud_range[None, :2]) / (
                point_cloud_range[None, 3:5] - point_cloud_range[None, :2])

            assert self.initialize_by_heatmap
            W, H = lidar_feat.shape[-2:]
            gs = gs * torch.as_tensor([H, W], dtype=gs.dtype, device='cuda')
            gs[:, 0] = gs[:, 0].clip(max=H - 1, min=0)
            gs[:, 1] = gs[:, 1].clip(max=W - 1, min=0)
            gs = gs.to(torch.int64)
            positive_mask = (
                box_center_noise.norm(dim=1) < self.add_gt_pos_thresh)
            gs_labels[torch.logical_not(positive_mask)] = self.num_classes

            if self.add_gt_groups_noise.startswith('box'):
                positive_mask = noise.norm(
                    dim=1) < self.add_gt_pos_boxnoise_thresh
                gs_labels[torch.logical_not(positive_mask)] = self.num_classes
            batch_gt_pos.append(gs)
            batch_gt_query_labels.append(gs_labels)
            batch_gt_bboxes_3d.append(gs_box)
        batch_gt_pos = torch.stack(batch_gt_pos, dim=0)

        batch_gt_query_labels = torch.stack(batch_gt_query_labels, dim=0)

        batch_gt_bboxes_3d = torch.stack(batch_gt_bboxes_3d, dim=0)

        batch_gt_bev_pos = bev_pos.gather(
            index=(batch_gt_pos[:, :, 1] * H +
                   batch_gt_pos[:, :, 0])[...,
                                          None].expand(-1, -1,
                                                       bev_pos.shape[-1]),
            dim=1)

        batch_gt_heatmap_score = heatmap.gather(
            index=(batch_gt_pos[:, :, 1] * H +
                   batch_gt_pos[:, :, 0])[:, None, :].expand(
                       -1, self.num_classes, -1),
            dim=-1)

        batch_gt_query_feat = lidar_feat_flatten.gather(
            index=(batch_gt_pos[:, None, :, 1] * H +
                   batch_gt_pos[:, None, :, 0]).expand(
                       -1, lidar_feat_flatten.shape[1], -1),
            dim=-1)

        noise_split = self.add_gt_groups_noise.split(',')

        if len(noise_split) > 2 and noise_split[2] == 'heatmap':
            batch_gt_heatmap_class = batch_gt_heatmap_score.argmax(dim=1)
            batch_gt_one_hot = F.one_hot(
                batch_gt_heatmap_class,
                num_classes=self.num_classes + 1).permute(0, 2, 1)

        elif len(noise_split) > 2 and noise_split[2] == 'heatmapcls':
            batch_gt_one_hot = batch_gt_heatmap_score

        else:
            batch_gt_one_hot = F.one_hot(
                batch_gt_query_labels,
                num_classes=self.num_classes + 1).permute(0, 2, 1)
        batch_gt_one_hot = batch_gt_one_hot[:, :self.num_classes]

        batch_gt_query_cat_encoding = self.class_encoding(
            batch_gt_one_hot.float())

        batch_gt_query_feat = batch_gt_query_feat + batch_gt_query_cat_encoding

        query_pos = torch.cat([
            query_pos,
            batch_gt_bev_pos * batch_valid_gt_mask[..., None].float()
        ],
                              dim=1)

        query_feat = torch.cat([
            query_feat,
            batch_gt_query_feat * batch_valid_gt_mask[:, None, :].float()
        ],
                               dim=2)

        query_heatmap_score = torch.cat([
            query_heatmap_score,
            batch_gt_heatmap_score * batch_valid_gt_mask[:, None, :].float()
        ],
                                        dim=2)

        if dense_heatmap_boxes is not None:
            if self.add_gt_groups_noise_box == 'pred':
                batch_gt_heatmap_box = dense_heatmap_boxes.gather(
                    index=(
                        batch_gt_query_labels.clip(max=self.num_classes - 1) *
                        H * W + batch_gt_pos[:, :, 1] * H +
                        batch_gt_pos[:, :, 0])[:, None].expand(
                            -1, dense_heatmap_boxes.shape[1], -1),
                    dim=-1)

            elif self.add_gt_groups_noise_box == 'gt':
                batch_gt_heatmap_box = batch_gt_bboxes_3d.transpose(1, 2)
                batch_gt_heatmap_box[:, :2, :] = (
                    batch_gt_bev_pos *
                    batch_valid_gt_mask[..., None].float()).transpose(1, 2)

            elif self.add_gt_groups_noise_box == 'gtnoise':
                batch_gt_heatmap_box = batch_gt_bboxes_3d.transpose(1, 2)
                meta_noise = (
                    torch.rand(
                        batch_gt_heatmap_box[:, 2:, :].shape, device='cuda') *
                    2 - 1)
                batch_gt_heatmap_box[:, :2, :] = (
                    batch_gt_bev_pos *
                    batch_valid_gt_mask[..., None].float()).transpose(1, 2)
                batch_gt_heatmap_box[:, [2], :] = batch_gt_heatmap_box[:, [
                    2
                ], :] + meta_noise[:, [0]] * batch_gt_heatmap_box[:,
                                                                  [5]].exp()
                batch_gt_heatmap_box[:, [3, 4, 5], :] = torch.log(
                    batch_gt_heatmap_box[:, [3, 4, 5], :].exp() *
                    (1 + meta_noise[:, [1, 2, 3]]).clip(min=0.1, max=3.) +
                    1e-6)
                batch_gt_heatmap_box[:, 6:7, :] = torch.sin(
                    meta_noise[:, 4:5] * np.pi)
                batch_gt_heatmap_box[:, 7:8, :] = torch.cos(
                    meta_noise[:, 4:5] * np.pi)
                if self.bbox_coder.code_size == 10:
                    batch_gt_heatmap_box[:, 8:10] = (
                        batch_gt_heatmap_box[:, 8:10] *
                        (1 + meta_noise[:, 6:8]))
            batch_gt_heatmap_box *= (batch_gt_query_labels !=
                                     self.num_classes).float()[:, None]
            query_box = torch.cat([
                query_box,
                batch_gt_heatmap_box * batch_valid_gt_mask[:, None, :].float()
            ],
                                  dim=2)
            return (query_feat, query_pos, query_heatmap_score,
                    batch_valid_gt_mask, batch_gt_query_labels, query_box)

        else:
            return (query_feat, query_pos, query_heatmap_score,
                    batch_valid_gt_mask, batch_gt_query_labels)

    # forward() is algorithmically identical to original
    # Only structural change: uses already-refactored builders

    def forward(self,
                pts_inputs,
                img_inputs,
                img_metas,
                gt_bboxes_3d=None,
                gt_labels_3d=None,
                **input_kwargs):
        self.num_proposals = self.num_proposals_ori

        lidar_feat = pts_inputs[0]

        if self.extra_feat:
            extra_feats = pts_inputs[1][-1]
            pts_inputs[1].pop(-1)
        batch_size = lidar_feat.shape[0]

        lidar_feat_flatten = lidar_feat.view(batch_size, lidar_feat.shape[1],
                                             -1)

        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)

        if self.multiscale:
            bev_pos_2 = self.create_2D_grid(
                lidar_feat.shape[2] // 2, lidar_feat.shape[2] // 2).repeat(
                    batch_size, 1, 1).to(lidar_feat.device) * 2
            bev_pos_4 = self.create_2D_grid(
                lidar_feat.shape[2] // 4, lidar_feat.shape[2] // 4).repeat(
                    batch_size, 1, 1).to(lidar_feat.device) * 4

        dense_heatmap_boxes = None

        query_box = None

        if not self.multistage_heatmap:
            dense_heatmap = self.heatmap_head(lidar_feat)

            if self.input_img or self.iterbev_wo_img:
                if isinstance(pts_inputs[1], (list, tuple)):
                    new_lidar_feat = pts_inputs[1][-1]
                else:
                    new_lidar_feat = pts_inputs[1]
                lidar_feat_flatten = new_lidar_feat.view(
                    *lidar_feat_flatten.shape)
                dense_heatmap_img = self.heatmap_head_img(
                    new_lidar_feat.view(lidar_feat.shape))
                heatmap = (dense_heatmap.detach().sigmoid() +
                           dense_heatmap_img.detach().sigmoid()) / 2

            else:
                heatmap = dense_heatmap.detach().sigmoid()
                new_lidar_feat = lidar_feat

            if self.input_img or self.iterbev_wo_img:
                heatmap_train = [dense_heatmap, dense_heatmap_img]

            else:
                heatmap_train = dense_heatmap
            padding = self.nms_kernel_size // 2
            local_max = torch.zeros_like(heatmap)
            local_max_inner = F.max_pool2d(
                heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
            local_max[:, :, padding:(-padding),
                      padding:(-padding)] = local_max_inner

            if self.test_cfg['dataset'] == 'nuScenes':
                local_max[:, 8, ] = F.max_pool2d(
                    heatmap[:, 8], kernel_size=1, stride=1, padding=0)
                local_max[:, 9, ] = F.max_pool2d(
                    heatmap[:, 9], kernel_size=1, stride=1, padding=0)

            elif self.test_cfg['dataset'] == 'Waymo':
                local_max[:, 1, ] = F.max_pool2d(
                    heatmap[:, 1], kernel_size=1, stride=1, padding=0)
                local_max[:, 2, ] = F.max_pool2d(
                    heatmap[:, 2], kernel_size=1, stride=1, padding=0)
            heatmap = heatmap * (heatmap == local_max)
            heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)
            top_proposals = heatmap.view(batch_size, -1).argsort(
                dim=-1, descending=True)[..., :self.num_proposals]
            top_proposals_class = top_proposals // heatmap.shape[-1]
            top_proposals_index = top_proposals % heatmap.shape[-1]
            query_feat = lidar_feat_flatten.gather(
                index=top_proposals_index[:, None, :].expand(
                    -1, lidar_feat_flatten.shape[1], -1),
                dim=-1)
            self.query_labels = top_proposals_class
            one_hot = F.one_hot(
                top_proposals_class,
                num_classes=self.num_classes).permute(0, 2, 1)
            query_cat_encoding = self.class_encoding(one_hot.float())
            query_feat = query_feat + query_cat_encoding
            query_pos = bev_pos.gather(
                index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(
                    -1, -1, bev_pos.shape[-1]),
                dim=1)
            query_heatmap_score = heatmap.gather(
                index=top_proposals_index[:, None, :].expand(
                    -1, self.num_classes, -1),
                dim=-1)

        else:

            # Multistage heatmap path. Identical logic to the original,
            # it just uses the refactored builders.
            dense_heatmap = self.heatmap_head(lidar_feat)
            multistage_feats = pts_inputs[1]

            if self.reuse_first_heatmap:
                multistage_feats.insert(0, lidar_feat)
            query_labels = []
            query_feats = []
            query_boxes = []
            query_poses = []
            query_heatmap_scores = []
            acc_masks = torch.ones_like(dense_heatmap).view(batch_size, -1)
            multistage_masks = []
            heatmap_train = []
            multistage_bev_preds = []

            for i in range(self.multistage_heatmap):
                if i == 0 and self.reuse_first_heatmap:
                    if self.heatmap_box:

                        assert self.test_cfg['dataset'] == 'nuScenes'
                        shared_feat = multistage_feats[i]
                        dense_preds = []
                        dense_heatmap_boxes = []
                        if not self.thin_heatmap_box:
                            for task_id, task in enumerate(
                                    self.multi_stage_task_heads[i]):
                                dense_preds.append(task(shared_feat))
                                dense_pred = dense_preds[-1]
                                if 'vel' in dense_pred:
                                    dense_pred = (dense_pred['reg'],
                                                  dense_pred['height'],
                                                  dense_pred['dim'],
                                                  dense_pred['rot'],
                                                  dense_pred['vel'])
                                else:
                                    dense_pred = (dense_pred['reg'],
                                                  dense_pred['height'],
                                                  dense_pred['dim'],
                                                  dense_pred['rot'])
                                dense_pred = torch.cat(
                                    dense_pred, dim=1)[:, :, None].expand(
                                        -1, -1, self.heatmap_tasks[task_id]
                                        ['num_class'], -1, -1)
                                dense_heatmap_boxes.append(dense_pred)
                        else:
                            head_i = self.multi_stage_task_heads[i]
                            dense_heatmap_boxes_raw = head_i(shared_feat)
                            dense_preds_raw = torch.split(
                                dense_heatmap_boxes_raw, [10] * 6, dim=1)
                            for task_id in range(len(self.heatmap_tasks)):
                                dense_pred = torch.split(
                                    dense_preds_raw[task_id], [2, 1, 3, 2, 2],
                                    dim=1)
                                dense_preds.append(
                                    dict(
                                        reg=dense_pred[0],
                                        height=dense_pred[1],
                                        dim=dense_pred[2],
                                        rot=dense_pred[3],
                                        vel=dense_pred[4]))
                                dense_heatmap_boxes.append(
                                    dense_preds_raw[task_id]
                                    [:, :, None].expand(
                                        -1, -1, self.heatmap_tasks[task_id]
                                        ['num_class'], -1, -1))
                        multistage_bev_preds.append(dense_preds)
                        dense_heatmap_boxes = torch.cat(
                            dense_heatmap_boxes, dim=2)
                    heatmap = dense_heatmap.detach().sigmoid()
                    heatmap_train.append(dense_heatmap)
                    multistage_masks.append(
                        acc_masks.view(*heatmap.shape).clone())
                    heatmap = heatmap * acc_masks.view(*heatmap.shape)
                else:
                    if not self.heatmap_box:
                        dense_heatmap_img = self.heatmap_head_img[i](
                            multistage_feats[i])
                    else:

                        assert self.test_cfg['dataset'] == 'nuScenes'
                        shared_feat = multistage_feats[i]
                        dense_preds = []
                        dense_heatmap_boxes = []
                        if not self.thin_heatmap_box:
                            for task_id, task in enumerate(
                                    self.multi_stage_task_heads[i]):
                                dense_preds.append(task(shared_feat))
                                dense_pred = dense_preds[-1]
                                dense_pred = torch.cat(
                                    (dense_pred['reg'], dense_pred['height'],
                                     dense_pred['dim'], dense_pred['rot'],
                                     dense_pred['vel']),
                                    dim=1)[:, :, None].expand(
                                        -1, -1, self.heatmap_tasks[task_id]
                                        ['num_class'], -1, -1)
                                dense_heatmap_boxes.append(dense_pred)
                            dense_heatmap_img = torch.cat(
                                [p['heatmap'] for p in dense_preds], dim=1)
                        else:
                            head_i = self.multi_stage_task_heads[i]
                            dense_heatmap_boxes_raw = head_i(shared_feat)
                            dense_preds_raw = torch.split(
                                dense_heatmap_boxes_raw, [10] * 6, dim=1)
                            for task_id in range(len(self.heatmap_tasks)):
                                dense_pred = torch.split(
                                    dense_preds_raw[task_id], [2, 1, 3, 2, 2],
                                    dim=1)
                                dense_preds.append(
                                    dict(
                                        reg=dense_pred[0],
                                        height=dense_pred[1],
                                        dim=dense_pred[2],
                                        rot=dense_pred[3],
                                        vel=dense_pred[4]))
                                dense_heatmap_boxes.append(
                                    dense_preds_raw[task_id]
                                    [:, :, None].expand(
                                        -1, -1, self.heatmap_tasks[task_id]
                                        ['num_class'], -1, -1))
                            dense_heatmap_img = self.heatmap_head_img[i](
                                multistage_feats[i])
                        multistage_bev_preds.append(dense_preds)
                        dense_heatmap_boxes = torch.cat(
                            dense_heatmap_boxes, dim=2)
                    heatmap = dense_heatmap_img.detach().sigmoid()
                    if i == 0:
                        heatmap_train.append(dense_heatmap)
                        multistage_masks.append(
                            acc_masks.view(*heatmap.shape).clone())
                    heatmap = heatmap * acc_masks.view(*heatmap.shape)
                    heatmap_train.append(dense_heatmap_img)
                    multistage_masks.append(
                        acc_masks.view(*heatmap.shape).clone())
                lidar_feat_flatten = multistage_feats[i].view(
                    *lidar_feat_flatten.shape)
                padding = self.nms_kernel_size // 2
                local_max = torch.zeros_like(heatmap)
                local_max_inner = F.max_pool2d(
                    heatmap,
                    kernel_size=self.nms_kernel_size,
                    stride=1,
                    padding=0)
                local_max[:, :, padding:(-padding),
                          padding:(-padding)] = local_max_inner
                if self.test_cfg['dataset'] == 'nuScenes':
                    local_max[:, 8, ] = F.max_pool2d(
                        heatmap[:, 8], kernel_size=1, stride=1, padding=0)
                    local_max[:, 9, ] = F.max_pool2d(
                        heatmap[:, 9], kernel_size=1, stride=1, padding=0)
                elif self.test_cfg['dataset'] == 'Waymo':
                    local_max[:, 1, ] = F.max_pool2d(
                        heatmap[:, 1], kernel_size=1, stride=1, padding=0)
                    local_max[:, 2, ] = F.max_pool2d(
                        heatmap[:, 2], kernel_size=1, stride=1, padding=0)
                heatmap = heatmap * (heatmap == local_max)
                heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)
                top_proposals = torch.topk(
                    heatmap.view(batch_size, -1),
                    k=self.num_proposals,
                    dim=-1,
                    largest=True,
                    sorted=False).indices
                top_proposals_class = top_proposals // heatmap.shape[-1]
                top_proposals_index = top_proposals % heatmap.shape[-1]
                query_feat = lidar_feat_flatten.gather(
                    index=top_proposals_index[:, None, :].expand(
                        -1, lidar_feat_flatten.shape[1], -1),
                    dim=-1)
                query_labels.append(top_proposals_class)
                one_hot = F.one_hot(
                    top_proposals_class,
                    num_classes=self.num_classes).permute(0, 2, 1)
                query_cat_encoding = self.class_encoding(one_hot.float())
                query_feat = query_feat + query_cat_encoding
                query_pos = bev_pos.gather(
                    index=top_proposals_index[:,
                                              None, :].permute(0, 2, 1).expand(
                                                  -1, -1, bev_pos.shape[-1]),
                    dim=1)
                query_heatmap_score = heatmap.gather(
                    index=top_proposals_index[:, None, :].expand(
                        -1, self.num_classes, -1),
                    dim=-1)
                query_feats.append(query_feat)
                query_poses.append(query_pos)
                query_heatmap_scores.append(query_heatmap_score)
                if self.heatmap_box:
                    box_dim = dense_heatmap_boxes.shape[1]
                    dense_heatmap_boxes = dense_heatmap_boxes.detach().view(
                        batch_size, box_dim, self.num_classes,
                        heatmap.shape[-1])

                    assert self.test_cfg['dataset'] == 'nuScenes'
                    dense_heatmap_boxes[:, :2, :, :] += bev_pos.int().float(
                    ).transpose(1, 2)[:, :, None].expand_as(
                        dense_heatmap_boxes[:, :2, :, :])
                    # Trailing ":, :" slices are implicit; hb aliases the same
                    # tensor so these clips stay in place.
                    hb = dense_heatmap_boxes
                    hb[:, 2:3] = hb[:, 2:3].clip(min=-5., max=3.)
                    hb[:, 3:6] = hb[:, 3:6].clip(
                        min=np.log(0.5), max=np.log(15))
                    hb[:, 6:8] = hb[:, 6:8].clip(min=-1., max=1.)
                    hb[:, 8:10] = hb[:, 8:10].clip(min=-15., max=15.)
                    dense_heatmap_boxes = dense_heatmap_boxes.view(
                        batch_size, box_dim,
                        self.num_classes * heatmap.shape[-1])
                    query_box = dense_heatmap_boxes.gather(
                        index=top_proposals[:,
                                            None, :].expand(-1, box_dim, -1),
                        dim=-1)
                    query_boxes.append(query_box)
                # mask heatmap mode selection - identical to original
                if self.mask_heatmap_mode == 'pos':
                    selected_mask = acc_masks.new_zeros(
                        batch_size, self.num_classes, heatmap.shape[-1])
                    selected_mask.scatter_(
                        index=top_proposals_index[:, None, :].expand(
                            -1, self.num_classes, -1),
                        dim=2,
                        src=acc_masks.new_ones(
                            (batch_size, self.num_classes, heatmap.shape[-1])))
                elif self.mask_heatmap_mode == 'poscls':
                    selected_mask = acc_masks.new_zeros(
                        batch_size, self.num_classes * heatmap.shape[-1])
                    selected_mask.scatter_(
                        index=top_proposals,
                        dim=1,
                        src=torch.ones_like(
                            top_proposals, dtype=acc_masks.dtype))
                elif self.mask_heatmap_mode == 'boxcls':
                    boxmask_margin = 1.
                    boxmask_margin_ratio = None

                    assert self.test_cfg['dataset'] == 'nuScenes'
                    selected_mask = acc_masks.new_zeros(
                        batch_size, self.num_classes * heatmap.shape[-1])
                    selected_mask.scatter_(
                        index=top_proposals,
                        dim=1,
                        src=torch.ones_like(
                            top_proposals, dtype=acc_masks.dtype))

                    def pos_inside_boxes(query_box,
                                         bev_pos,
                                         margin,
                                         min_bev_dim,
                                         margin_ratio=None):

                        assert query_box.shape[1] >= 9
                        from mmdet3d.ops.roiaware_pool3d import \
                            points_in_boxes_gpu
                        rot = query_box[:, 6:8]
                        dim = query_box[:, 3:6]
                        center = query_box[:, 0:2]
                        height = query_box[:, 2:3]
                        vel = query_box[:, 8:]
                        query_boxes_std = self.bbox_coder.decode_box(
                            rot.clone(), dim.clone(), center.clone(),
                            height.clone(), vel.clone())
                        pc_range = torch.as_tensor(
                            [-54, -54, -5.0, 54, 54, 3.0], device='cuda')
                        query_boxes_std[..., [
                            0,
                        ]] = query_boxes_std[..., [
                            0,
                        ]].clip(
                            min=pc_range[0], max=pc_range[3])
                        query_boxes_std[..., [
                            1,
                        ]] = query_boxes_std[..., [
                            1,
                        ]].clip(
                            min=pc_range[1], max=pc_range[4])
                        if margin_ratio is not None and margin_ratio > 0.:
                            query_boxes_std[..., [3, 4]] *= (1. - margin_ratio)
                        else:
                            query_boxes_std[..., [3, 4]] -= margin
                        query_boxes_std[..., [3, 4]] = query_boxes_std[
                            ..., [3, 4]].clip(
                                min=min_bev_dim, max=10.)
                        query_boxes_std[..., 5] = 1000
                        query_boxes_std[..., 2] = -100.
                        temp_bev_pos = self.bbox_coder.decode_center(
                            bev_pos.transpose(1, 2))
                        temp_bev_pos = torch.cat([
                            temp_bev_pos,
                            temp_bev_pos.new_zeros(batch_size, 1,
                                                   bev_pos.shape[1])
                        ],
                                                 dim=1).transpose(1, 2)
                        inside_boxes = points_in_boxes_gpu(
                            temp_bev_pos, query_boxes_std[:, :, :7])
                        return inside_boxes

                    inside_boxes = pos_inside_boxes(
                        query_box,
                        bev_pos,
                        margin=boxmask_margin,
                        min_bev_dim=0.7,
                        margin_ratio=boxmask_margin_ratio)
                    bev_pos_class = top_proposals_class.gather(
                        index=inside_boxes.clip(min=0).long(), dim=1)
                    bev_pos_class[inside_boxes == -1] = self.num_classes
                    selected_mask_box = acc_masks.new_zeros(
                        batch_size, self.num_classes + 1, heatmap.shape[-1])
                    selected_mask_box.scatter_(
                        index=bev_pos_class[:, None],
                        dim=1,
                        src=torch.ones_like(
                            bev_pos_class[:, None], dtype=acc_masks.dtype))
                    selected_mask_box = selected_mask_box[:, :self.
                                                          num_classes].reshape(
                                                              batch_size,
                                                              self.num_classes
                                                              *
                                                              heatmap.shape[-1]
                                                          )
                    selected_mask = (selected_mask + selected_mask_box >
                                     0.1).float()
                else:
                    selected_mask = acc_masks.new_zeros(
                        batch_size, self.num_classes * heatmap.shape[-1])
                selected_mask = selected_mask.reshape(*dense_heatmap.shape)
                selected_mask_kernel = F.max_pool2d(
                    selected_mask,
                    kernel_size=self.nms_kernel_size,
                    stride=1,
                    padding=self.nms_kernel_size // 2)
                if self.test_cfg['dataset'] == 'nuScenes':
                    selected_mask_kernel[:, 8:10] = F.max_pool2d(
                        selected_mask[:, 8:10],
                        kernel_size=1,
                        stride=1,
                        padding=0)
                elif self.test_cfg['dataset'] == 'Waymo':
                    selected_mask_kernel[:, 1:3] = F.max_pool2d(
                        selected_mask[:, 1:3],
                        kernel_size=1,
                        stride=1,
                        padding=0)
                acc_masks = acc_masks * (1. - selected_mask_kernel).view(
                    *acc_masks.shape)
            self.query_labels = torch.cat(query_labels, dim=1)
            query_feat = torch.cat(query_feats, dim=2)
            query_pos = torch.cat(query_poses, dim=1)
            query_heatmap_score = torch.cat(query_heatmap_scores, dim=2)

            if self.heatmap_box:
                query_box = torch.cat(query_boxes, dim=2)
            self.num_proposals = (
                self.num_proposals_ori * self.multistage_heatmap)

        if self.training:
            self.num_gts = [i.shape[0] for i in gt_labels_3d]
            self.max_num_gts = max(self.num_gts)

        query_labels = self.query_labels

        if self.training and self.add_gt_groups > 0:
            updated_gt_queries = self.generate_gt_groups(
                query_feat,
                query_pos,
                query_heatmap_score,
                lidar_feat,
                lidar_feat_flatten,
                bev_pos,
                heatmap,
                gt_bboxes_3d,
                gt_labels_3d,
                dense_heatmap_boxes=dense_heatmap_boxes,
                query_box=query_box)

            if dense_heatmap_boxes is None:
                (query_feat, query_pos, query_heatmap_score,
                 batch_valid_gt_mask,
                 batch_gt_query_labels) = updated_gt_queries

            else:
                (query_feat, query_pos, query_heatmap_score,
                 batch_valid_gt_mask, batch_gt_query_labels,
                 query_box) = updated_gt_queries
            query_labels = torch.cat([query_labels, batch_gt_query_labels],
                                     dim=1)

        if self.multiscale:
            if not self.multistage_heatmap:
                lidar_feat = new_lidar_feat

            else:
                if self.extra_feat:
                    lidar_feat = extra_feats
                else:
                    lidar_feat = multistage_feats[-1]
            multiscale_inputs = [lidar_feat]

            if self.multiscale:
                multiscale_inputs.append(self.dconv(multiscale_inputs[-1]))
                multiscale_inputs.append(self.dconv2(multiscale_inputs[-1]))
            multiscale_inputs_flatten = torch.cat(
                [i.flatten(2, 3) for i in multiscale_inputs], dim=-1)

        ret_dicts = []

        for i in range(self.num_decoder_layers):
            if self.training:
                num_proposals_new = (
                    self.num_proposals + self.max_num_gts * self.add_gt_groups)

            else:
                num_proposals_new = self.num_proposals

            if not self.multiscale:
                W, H = lidar_feat.shape[-2:]
                spatial_shapes = torch.as_tensor(
                    [[H, W]], dtype=torch.long,
                    device='cuda')  # NOTE: mmdet 3.x expects (h, w)
                level_start_index = torch.as_tensor([
                    0,
                ],
                                                    dtype=torch.long,
                                                    device='cuda')

            else:
                spatial_shapes = torch.as_tensor(
                    [i.shape[2:] for i in multiscale_inputs],
                    dtype=torch.long,
                    device='cuda')
                level_start_index = torch.as_tensor([
                    0, *(torch.cumsum(
                        torch.prod(spatial_shapes, dim=1), dim=0)[:-1])
                ],
                                                    dtype=torch.long,
                                                    device='cuda')
                lidar_feat_flatten = multiscale_inputs_flatten
                if self.bevpos and i == 0:
                    bev_pos = torch.cat([bev_pos, bev_pos_2, bev_pos_4], dim=1)

            if self.training and self.add_gt_groups > 0:
                attn_masks = torch.ones(
                    (batch_size, num_proposals_new, num_proposals_new),
                    dtype=bool,
                    device='cuda')
                attn_masks[:, :, :self.num_proposals] = 0
                attn_masks[:, self.num_proposals:,
                           self.num_proposals:] = torch.logical_not(
                               batch_valid_gt_mask[:, None]
                               & batch_valid_gt_mask[:, :, None])
                attn_masks = attn_masks[:, None, :, :].repeat(
                    1, self.num_heads, 1, 1).flatten(0, 1)

            else:
                attn_masks = None
            kwargs = dict(
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=torch.ones((batch_size, 1, 2), device='cuda'),
                key_padding_mask=None,
                self_attn_mask=attn_masks)
            reference_points = query_pos / torch.flip(
                spatial_shapes[:1], dims=(1, ))[:, None]
            query_sine_pos = gen_sineembed_for_position(
                reference_points[:, :, :2])
            query_pos_embed = self.pos_embed_learned[i](query_sine_pos)

            if self.boxpos is not None and query_box is not None:
                query_box_embed = self.box_pos_embed_learned[i]
                query_pos_embed = query_pos_embed + query_box_embed

            if self.bevpos:
                bev_reference_points = bev_pos / torch.flip(
                    spatial_shapes[:1], dims=(1, ))[:, None]
                bev_sine_pos = gen_sineembed_for_position(
                    bev_reference_points[:, :, :2])
                bev_pos_embed = self.pos_embed_learned[i](bev_sine_pos)
                pos_lidar_feat_flatten = (
                    lidar_feat_flatten + bev_pos_embed.transpose(1, 2))

            else:
                pos_lidar_feat_flatten = lidar_feat_flatten

            if self.roi_feats and query_box is not None:
                rot = query_box[:, 6:8]
                dim = query_box[:, 3:6]
                center = query_box[:, 0:2]
                height = query_box[:, 2:3]
                vel = query_box[:, 8:]
                std_boxes = self.bbox_coder.decode_box(
                    rot.clone(),
                    dim.clone() * self.roi_expand_ratio[i], center.clone(),
                    height.clone(), vel.clone())
                # orig
                std_boxes = std_boxes.reshape(batch_size * num_proposals_new,
                                              std_boxes.shape[-1])
                grid_points = self.get_dense_grid_points(
                    std_boxes, batch_size * num_proposals_new, self.roi_feats)
                grid_points = torch.cat([
                    grid_points,
                    grid_points.new_ones(*grid_points.shape[:2], 1)
                ],
                                        dim=-1)
                grid_points = rotation_3d_in_axis(
                    grid_points, std_boxes[:, 6], axis=2)
                grid_points = grid_points[..., :2]
                grid_points = grid_points + std_boxes[:, None, :2]
                grid_points = grid_points.view(batch_size, num_proposals_new,
                                               self.roi_feats**2, 2)
                if self.test_cfg['dataset'] == 'nuScenes':
                    pc_range = torch.as_tensor([-54, -54, -5.0, 54, 54, 3.0],
                                               device='cuda')
                else:
                    pc_range = torch.as_tensor(
                        [-75.2, -75.2, -2, 75.2, 75.2, 4], device='cuda')
                grid_points = (grid_points - pc_range[:2]) / (
                    pc_range[3:5] - pc_range[:2])
                grid_points = grid_points * 2. - 1.
                grid_points = grid_points.clip(min=-2., max=2.)
                if not self.multiscale:
                    roi_feat = F.grid_sample(
                        lidar_feat_flatten.view(batch_size, -1,
                                                *lidar_feat.shape[-2:]),
                        grid_points)
                else:
                    ms_roi_feat = []
                    for feat in multiscale_inputs:
                        roi_feat = F.grid_sample(
                            feat, grid_points, mode='bilinear')
                        ms_roi_feat.append(roi_feat)
                    roi_feat = torch.cat(ms_roi_feat, dim=1)
                roi_feat = roi_feat.permute(0, 2, 1, 3).reshape(
                    batch_size * num_proposals_new,
                    lidar_feat_flatten.shape[1] *
                    (3 if self.multiscale else 1) * self.roi_feats**2)
                roi_feat = self.roi_mlp(roi_feat)
                roi_feat = roi_feat.view(
                    batch_size, num_proposals_new,
                    lidar_feat_flatten.shape[1]).transpose(1, 2)
                query_feat = query_feat + roi_feat
            # NOTE: mmdet 3.x decoder expects batch-first,
            # (bs, num_queries, dim).
            # query_feat is (bs, C, num_queries) → need (bs, num_queries, C)
            # pos_lidar_feat_flatten is (bs, C, num_value), so it needs
            # to become (bs, num_value, C).
            # query_pos_embed is already (bs, num_queries, C) from MLP output
            query_feat_in = query_feat.permute(0, 2, 1)  # (bs, num_queries, C)
            value_in = pos_lidar_feat_flatten.permute(0, 2,
                                                      1)  # (bs, num_value, C)
            query_feat_out, reference_points = self.decoder[i](
                query=query_feat_in,
                value=value_in,
                query_pos=query_pos_embed,
                reference_points=reference_points,
                **kwargs)
            # query_feat_out is (bs, num_queries, C), convert it back
            # to (bs, C, num_queries).
            query_feat = query_feat_out.permute(0, 2, 1)
            query_pos = reference_points * torch.flip(
                spatial_shapes[:1], dims=(1, ))[:, None]
            res_layer = self.prediction_heads[i](query_feat)

            if self.classaware_reg:
                for k in ['center', 'height', 'dim', 'rot']:
                    res_layer[k] = res_layer[k].view(batch_size,
                                                     self.num_classes, -1,
                                                     num_proposals_new)
                    res_layer[k] = res_layer[k].gather(
                        index=query_labels[:, None, None, :].expand(
                            -1, -1, res_layer[k].shape[2],
                            -1).clip(0, self.num_classes - 1),
                        dim=1)[:, 0]
            res_layer['center'] = res_layer['center'] + query_pos.permute(
                0, 2, 1)
            query_pos = res_layer['center'].detach().clone().permute(0, 2, 1)

            if self.roi_based_reg and query_box is not None:
                res_layer['dim'][:, :2] = res_layer[
                    'dim'][:, :2] + query_box[:, 3:5].detach()
                res_layer['rot'] = res_layer['rot'] + query_box[:,
                                                                6:8].detach()
            query_box = [
                res_layer['center'], res_layer['height'], res_layer['dim'],
                res_layer['rot']
            ]

            if 'vel' in res_layer:
                query_box.append(res_layer['vel'])
            query_box = torch.cat(query_box, dim=1).detach()
            ret_dicts.append(res_layer)

        if self.training and self.add_gt_groups > 0:
            ret_dicts[0]['batch_valid_gt_mask'] = batch_valid_gt_mask
            ret_dicts[0]['batch_gt_query_labels'] = batch_gt_query_labels

        if self.initialize_by_heatmap:
            ret_dicts[0]['query_heatmap_score'] = query_heatmap_score
            ret_dicts[0]['dense_heatmap'] = heatmap_train

            if self.multistage_heatmap:
                ret_dicts[0]['multistage_masks'] = multistage_masks

        new_res = {}

        for key in ret_dicts[0].keys():
            if key in [
                    'dense_heatmap', 'query_heatmap_score', 'multistage_masks',
                    'gt_query_bbox_targets', 'batch_valid_gt_mask',
                    'batch_gt_query_labels'
            ]:
                new_res[key] = ret_dicts[0][key]

            else:
                if self.training and self.add_gt_groups > 0:
                    gt_offset = self.max_num_gts * self.add_gt_groups
                    new_res[key] = torch.cat([
                        ret_dict[key][:, :, :-gt_offset]
                        for ret_dict in ret_dicts
                    ],
                                             dim=-1)
                    new_res[key + '_gtgroups'] = torch.cat([
                        ret_dict[key][:, :, -gt_offset:]
                        for ret_dict in ret_dicts
                    ],
                                                           dim=-1)
                else:
                    new_res[key] = torch.cat(
                        [ret_dict[key] for ret_dict in ret_dicts], dim=-1)

        if self.heatmap_box:
            new_res['multistage_bev_preds'] = multistage_bev_preds
            new_res['query_pos'] = query_pos
            new_res['query_box'] = query_box

        return [[new_res]]

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, preds_dict):
        """Generate training targets.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            preds_dicts (tuple of dict): first index by layer (default 1)
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.

                - torch.Tensor: classification target.  [BS, num_proposals]
                - torch.Tensor: classification weights (mask)
                  [BS, num_proposals]
                - torch.Tensor: regression target. [BS, num_proposals, 8]
                - torch.Tensor: regression weights. [BS, num_proposals, 8]
        """
        list_of_pred_dict = []

        for batch_idx in range(len(gt_bboxes_3d)):
            pred_dict = {}

            for key in preds_dict[0].keys():
                pred_dict[key] = preds_dict[0][key][batch_idx:batch_idx + 1]
            list_of_pred_dict.append(pred_dict)

        assert len(gt_bboxes_3d) == len(list_of_pred_dict)

        res_tuple = multi_apply(self.get_targets_single, gt_bboxes_3d,
                                gt_labels_3d, list_of_pred_dict,
                                np.arange(len(gt_labels_3d)))

        labels = torch.cat(res_tuple[0], dim=0)

        label_weights = torch.cat(res_tuple[1], dim=0)

        bbox_targets = torch.cat(res_tuple[2], dim=0)

        bbox_weights = torch.cat(res_tuple[3], dim=0)

        ious = torch.cat(res_tuple[4], dim=0)

        num_pos = np.sum(res_tuple[5])

        matched_ious = np.mean(res_tuple[6])

        if self.initialize_by_heatmap:
            heatmap = torch.cat(res_tuple[7], dim=0)
            return (labels, label_weights, bbox_targets, bbox_weights, ious,
                    num_pos, matched_ious, heatmap)

        else:
            return (labels, label_weights, bbox_targets, bbox_weights, ious,
                    num_pos, matched_ious)

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, preds_dict,
                           batch_idx):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            preds_dict (dict): dict of prediction result for a single sample
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.

                - torch.Tensor: classification target.  [1, num_proposals]
                - torch.Tensor: classification weights (mask)
                  [1, num_proposals]
                - torch.Tensor: regression target. [1, num_proposals, 8]
                - torch.Tensor: regression weights. [1, num_proposals, 8]
                - torch.Tensor: iou target. [1, num_proposals]
                - int: number of positive proposals
        """
        num_proposals = preds_dict['center'].shape[-1]

        score = copy.deepcopy(preds_dict['heatmap'].detach())

        center = copy.deepcopy(preds_dict['center'].detach())

        height = copy.deepcopy(preds_dict['height'].detach())

        dim = copy.deepcopy(preds_dict['dim'].detach())

        rot = copy.deepcopy(preds_dict['rot'].detach())

        if 'vel' in preds_dict.keys():
            vel = copy.deepcopy(preds_dict['vel'].detach())

        else:
            vel = None

        boxes_dict = self.bbox_coder.decode(score, rot, dim, center, height,
                                            vel)

        bboxes_tensor = boxes_dict[0]['bboxes']

        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(score.device)

        assign_result_list = []

        for idx_layer in range(self.num_decoder_layers):
            s = self.num_proposals * idx_layer
            e = self.num_proposals * (idx_layer + 1)
            bboxes_tensor_layer = bboxes_tensor[s:e, :]
            score_layer = score[..., s:e]

            if self.train_cfg.assigner.type == 'HungarianAssigner3D':
                assign_result = self.bbox_assigner.assign(
                    bboxes_tensor_layer, gt_bboxes_tensor, gt_labels_3d,
                    score_layer, self.train_cfg)

            elif self.train_cfg.assigner.type == 'HeuristicAssigner':
                assign_result = self.bbox_assigner.assign(
                    bboxes_tensor_layer, gt_bboxes_tensor, None, gt_labels_3d,
                    self.query_labels[batch_idx])

            else:
                raise NotImplementedError

            if self.gt_center_limit is not None:
                pos_gt_bboxes_tensor = gt_bboxes_tensor[
                    assign_result.gt_inds[assign_result.gt_inds > 0] - 1]
                pos_bboxes_tensor_layer = bboxes_tensor_layer[
                    assign_result.gt_inds > 0]
                invalid_gt_idxs = (pos_gt_bboxes_tensor[:, :2] -
                                   pos_bboxes_tensor_layer[:, :2]).norm(
                                       dim=1) > self.gt_center_limit
                assign_result.gt_inds[torch.nonzero(
                    assign_result.gt_inds > 0)[:, 0][invalid_gt_idxs]] = 0
            assign_result_list.append(assign_result)
        assign_result_ensemble = AssignResult(
            num_gts=sum([res.num_gts for res in assign_result_list]),
            gt_inds=torch.cat([res.gt_inds for res in assign_result_list]),
            max_overlaps=torch.cat(
                [res.max_overlaps for res in assign_result_list]),
            labels=torch.cat([res.labels for res in assign_result_list]))

        # mmdet 3.x PseudoSampler expects InstanceData, not raw tensors
        from mmengine.structures import InstanceData
        pred_instances = InstanceData(priors=bboxes_tensor)
        gt_instances = InstanceData(bboxes=gt_bboxes_tensor)

        sampling_result = self.bbox_sampler.sample(assign_result_ensemble,
                                                   pred_instances,
                                                   gt_instances)

        pos_inds = sampling_result.pos_inds

        neg_inds = sampling_result.neg_inds

        assert len(pos_inds) + len(neg_inds) == num_proposals

        bbox_targets = torch.zeros([num_proposals, self.bbox_coder.code_size
                                    ]).to(center.device)

        bbox_weights = torch.zeros([num_proposals, self.bbox_coder.code_size
                                    ]).to(center.device)

        ious = assign_result_ensemble.max_overlaps
        ious = torch.clamp(ious, min=0.0, max=1.0)

        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)

        label_weights = bboxes_tensor.new_zeros(
            num_proposals, dtype=torch.long)

        if gt_labels_3d is not None:
            labels += self.num_classes

        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(
                sampling_result.pos_gt_bboxes)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1

            else:
                labels[pos_inds] = gt_labels_3d[
                    sampling_result.pos_assigned_gt_inds]

            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0

            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        if self.initialize_by_heatmap:
            device = labels.device
            gt_bboxes_3d = torch.cat(
                [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]],
                dim=1).to(device)
            grid_size = torch.tensor(self.train_cfg['grid_size'])
            pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
            voxel_size = torch.tensor(self.train_cfg['voxel_size'])
            feature_map_size = grid_size[:2] // self.train_cfg[
                'out_size_factor']
            heatmap = gt_bboxes_3d.new_zeros(self.num_classes,
                                             feature_map_size[1],
                                             feature_map_size[0])

            for idx in range(len(gt_bboxes_3d)):
                width = gt_bboxes_3d[idx][3]
                length = gt_bboxes_3d[idx][4]
                width = width / voxel_size[0] / self.train_cfg[
                    'out_size_factor']
                length = length / voxel_size[1] / self.train_cfg[
                    'out_size_factor']
                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (length, width),
                        min_overlap=self.train_cfg['gaussian_overlap'])
                    radius = max(self.train_cfg['min_radius'], int(radius))
                    x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]
                    coor_x = (
                        x - pc_range[0]
                    ) / voxel_size[0] / self.train_cfg['out_size_factor']
                    coor_y = (
                        y - pc_range[1]
                    ) / voxel_size[1] / self.train_cfg['out_size_factor']
                    center = torch.tensor([coor_x, coor_y],
                                          dtype=torch.float32,
                                          device=device)
                    center_int = center.to(torch.int32)
                    draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]],
                                          center_int, radius)
            mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
            return labels[None], label_weights[None], bbox_targets[
                None], bbox_weights[None], ious[None], int(
                    pos_inds.shape[0]), float(mean_iou), heatmap[None]

        else:
            mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
            return labels[None], label_weights[None], bbox_targets[
                None], bbox_weights[None], ious[None], int(
                    pos_inds.shape[0]), float(mean_iou)

    # NOTE: @force_fp32 REMOVED — handled by AmpOptimWrapper in mmdet3d 1.1+

    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):

        if self.initialize_by_heatmap:
            (labels, label_weights, bbox_targets, bbox_weights, ious, num_pos,
             matched_ious, heatmap) = self.get_targets(gt_bboxes_3d,
                                                       gt_labels_3d,
                                                       preds_dicts[0])

        else:
            (labels, label_weights, bbox_targets, bbox_weights, ious, num_pos,
             matched_ious) = self.get_targets(gt_bboxes_3d, gt_labels_3d,
                                              preds_dicts[0])

        preds_dict = preds_dicts[0][0]

        loss_dict = dict()

        if self.initialize_by_heatmap:
            if isinstance(preds_dict['dense_heatmap'], (tuple, list)):
                if 'multistage_masks' in preds_dict:
                    multistage_masks = torch.cat(
                        preds_dict['multistage_masks'], dim=0)
                    heatmap = heatmap.repeat(
                        len(preds_dict['dense_heatmap']), 1, 1,
                        1) * multistage_masks
                else:
                    heatmap = heatmap.repeat(
                        len(preds_dict['dense_heatmap']), 1, 1, 1)
                preds_dict['dense_heatmap'] = torch.cat(
                    preds_dict['dense_heatmap'], dim=0)
                loss_heatmap = self.loss_heatmap(
                    clip_sigmoid(preds_dict['dense_heatmap']),
                    heatmap,
                    weight=(multistage_masks
                            if 'multistage_masks' in preds_dict else None),
                    avg_factor=max(heatmap.eq(1).float().sum().item(), 1))

            else:
                loss_heatmap = self.loss_heatmap(
                    clip_sigmoid(preds_dict['dense_heatmap']),
                    heatmap,
                    avg_factor=max(heatmap.eq(1).float().sum().item(), 1))
            loss_dict['loss_heatmap'] = loss_heatmap * self.loss_weight_heatmap

        for idx_layer in range(self.num_decoder_layers):
            prefix = f'layer_{idx_layer}'
            s = idx_layer * self.num_proposals
            e = (idx_layer + 1) * self.num_proposals
            layer_labels = labels[..., s:e].reshape(-1)
            layer_label_weights = label_weights[..., s:e].reshape(-1)
            layer_score = preds_dict['heatmap'][..., s:e]
            layer_cls_score = layer_score.permute(0, 2, 1).reshape(
                -1, self.num_classes)
            layer_loss_cls = self.loss_cls(
                layer_cls_score,
                layer_labels,
                layer_label_weights,
                avg_factor=max(num_pos, 1))
            layer_center = preds_dict['center'][..., s:e]
            layer_height = preds_dict['height'][..., s:e]
            layer_rot = preds_dict['rot'][..., s:e]
            layer_dim = preds_dict['dim'][..., s:e]
            preds = torch.cat(
                [layer_center, layer_height, layer_dim, layer_rot],
                dim=1).permute(0, 2, 1)

            if 'vel' in preds_dict.keys():
                layer_vel = preds_dict['vel'][..., s:e]
                preds = torch.cat([
                    layer_center, layer_height, layer_dim, layer_rot, layer_vel
                ],
                                  dim=1).permute(0, 2, 1)
            code_weights = self.train_cfg.get('code_weights', None)
            layer_bbox_weights = bbox_weights[:, s:e, :]
            layer_reg_weights = (
                layer_bbox_weights *
                layer_bbox_weights.new_tensor(code_weights))
            layer_bbox_targets = bbox_targets[:, s:e, :]
            layer_loss_bbox = self.loss_bbox(
                preds,
                layer_bbox_targets,
                layer_reg_weights,
                avg_factor=max(num_pos, 1))
            loss_dict[f'{prefix}_loss_cls'] = layer_loss_cls
            loss_dict[f'{prefix}_loss_bbox'] = layer_loss_bbox

        if self.add_gt_groups > 0:
            num_gt_groups_layers = self.num_decoder_layers
            batch_valid_gt_mask = preds_dict['batch_valid_gt_mask'].float()
            batch_gt_query_labels = preds_dict['batch_gt_query_labels'].repeat(
                1, num_gt_groups_layers)
            preds_gt_query = torch.cat([
                preds_dict['center_gtgroups'], preds_dict['height_gtgroups'],
                preds_dict['rot_gtgroups'], preds_dict['dim_gtgroups']
            ],
                                       dim=1)

            if 'vel' in preds_dict.keys():
                preds_gt_query = torch.cat(
                    [preds_gt_query, preds_dict['vel_gtgroups']], dim=1)
            preds_gt_query = preds_gt_query.permute(0, 2, 1)
            score_gt_query = preds_dict['heatmap_gtgroups'].permute(
                0, 2, 1).reshape(-1, self.num_classes)
            gt_query_bbox_targets = torch.zeros(
                (preds_gt_query.shape[0], self.max_num_gts,
                 self.bbox_coder.code_size),
                dtype=preds_gt_query.dtype,
                device=preds_gt_query.device)

            for batch_idx, g in enumerate(gt_bboxes_3d):
                gt_query_bbox_targets[
                    batch_idx, :len(g.tensor)] = self.bbox_coder.encode(
                        g.tensor)
            positive_mask = (batch_gt_query_labels != self.num_classes)
            gt_query_bbox_targets = gt_query_bbox_targets.repeat(
                1, self.add_gt_groups * num_gt_groups_layers, 1)
            gt_query_bbox_weights = batch_valid_gt_mask[:, :, None].repeat(
                1, num_gt_groups_layers, gt_query_bbox_targets.shape[-1])
            gt_query_reg_weights = (
                gt_query_bbox_weights *
                gt_query_bbox_weights.new_tensor(code_weights))
            gt_query_reg_weights = gt_query_reg_weights * positive_mask[
                ..., None].float()
            loss_dict['gt_query_loss_box'] = self.loss_bbox(
                preds_gt_query,
                gt_query_bbox_targets,
                gt_query_reg_weights,
                avg_factor=max(
                    sum(self.num_gts) * self.add_gt_groups *
                    num_gt_groups_layers, 1)) * self.gt_query_loss_weight
            gt_query_label_weights = batch_valid_gt_mask.repeat(
                1, num_gt_groups_layers).reshape(-1)
            loss_dict['gt_query_loss_cls'] = self.loss_cls(
                score_gt_query,
                batch_gt_query_labels.reshape(-1),
                gt_query_label_weights,
                avg_factor=max(
                    sum(self.num_gts) * self.add_gt_groups *
                    num_gt_groups_layers, 1)) * self.gt_query_loss_weight
        loss_dict['matched_ious'] = layer_loss_cls.new_tensor(matched_ious)

        if self.heatmap_box and self.multistage_heatmap:
            multistage_bev_preds = preds_dict['multistage_bev_preds']
            heatmaps, anno_boxes, inds, masks = self.get_heatmap_targets(
                gt_bboxes_3d, gt_labels_3d)
            loss_heatmap_multistage = torch.as_tensor(0., device='cuda')
            loss_bbox_multistage = torch.as_tensor(0., device='cuda')

            for stage_id, dense_preds in enumerate(multistage_bev_preds):
                ignore_mask = preds_dict['multistage_masks'][
                    -self.multistage_heatmap + stage_id]
                class_id = 0
                for task_id, dense_pred in enumerate(dense_preds):
                    classwise_ignore_mask = ignore_mask[:, class_id:class_id +
                                                        self.
                                                        heatmap_tasks[task_id]
                                                        ['num_class']]
                    class_id += self.heatmap_tasks[task_id]['num_class']
                    if not self.thin_heatmap_box:
                        dense_pred['heatmap'] = clip_sigmoid(
                            dense_pred['heatmap'])
                        num_pos_h = heatmaps[task_id].eq(
                            1).float().sum().item()
                        loss_heatmap = self.heatmap_loss_cls(
                            dense_pred['heatmap'],
                            heatmaps[task_id],
                            weight=classwise_ignore_mask,
                            avg_factor=max(num_pos_h, 1))
                        loss_heatmap_multistage += loss_heatmap
                    target_box = anno_boxes[task_id]
                    anno_box = [
                        dense_pred['reg'], dense_pred['height'],
                        dense_pred['dim'], dense_pred['rot']
                    ]
                    if 'vel' in dense_pred:
                        anno_box.append(dense_pred['vel'])
                    anno_box = torch.cat(anno_box, dim=1)
                    ind = inds[task_id]
                    num = masks[task_id].float().sum()
                    pred = anno_box.permute(0, 2, 3, 1).contiguous()
                    pred = pred.view(pred.size(0), -1, pred.size(3))
                    pred = _gather_feat(pred, ind)
                    mask = masks[task_id].unsqueeze(2).expand_as(
                        target_box).float()
                    isnotnan = (~torch.isnan(target_box)).float()
                    mask *= isnotnan
                    code_weights = self.train_cfg.get('code_weights', None)
                    bbox_weights_h = mask * mask.new_tensor(code_weights)
                    box_ignore_mask = (classwise_ignore_mask.sum(
                        dim=1, keepdims=True) > 0.1).float()
                    box_ignore_mask = box_ignore_mask.permute(0, 2, 3,
                                                              1).contiguous()
                    box_ignore_mask = box_ignore_mask.view(
                        box_ignore_mask.size(0), -1, box_ignore_mask.size(3))
                    box_ignore_mask = _gather_feat(box_ignore_mask, ind)
                    bbox_weights_h = bbox_weights_h * box_ignore_mask
                    loss_bbox = self.heatmap_loss_bbox(
                        pred,
                        target_box,
                        bbox_weights_h,
                        avg_factor=(num + 1e-4))
                    loss_bbox_multistage += loss_bbox
            loss_dict['separate_loss_heatmap'] = (
                loss_heatmap_multistage / self.multistage_heatmap *
                self.loss_weight_separate_heatmap)
            loss_dict['separate_loss_bbox'] = (
                loss_bbox_multistage / self.multistage_heatmap *
                self.loss_weight_separate_bbox)

        return loss_dict

    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False):
        rets = []

        for layer_id, preds_dict in enumerate(preds_dicts):
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_score = preds_dict[0]['heatmap'][
                ..., -self.num_proposals:].sigmoid()
            one_hot = F.one_hot(
                self.query_labels,
                num_classes=self.num_classes).permute(0, 2, 1)
            batch_score = batch_score * preds_dict[0][
                'query_heatmap_score'] * one_hot
            batch_center = preds_dict[0]['center'][..., -self.num_proposals:]
            batch_height = preds_dict[0]['height'][..., -self.num_proposals:]
            batch_dim = preds_dict[0]['dim'][..., -self.num_proposals:]
            batch_rot = preds_dict[0]['rot'][..., -self.num_proposals:]
            batch_vel = None

            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel'][..., -self.num_proposals:]

            temp = self.bbox_coder.decode(
                batch_score,
                batch_rot,
                batch_dim,
                batch_center,
                batch_height,
                batch_vel,
                filter=True)

            if self.test_cfg['dataset'] == 'nuScenes':
                self.tasks = [
                    dict(
                        num_class=8,
                        class_names=[],
                        indices=[0, 1, 2, 3, 4, 5, 6, 7],
                        radius=-1),
                    dict(
                        num_class=1,
                        class_names=['pedestrian'],
                        indices=[8],
                        radius=0.175),
                    dict(
                        num_class=1,
                        class_names=['traffic_cone'],
                        indices=[9],
                        radius=0.175)
                ]

            elif self.test_cfg['dataset'] == 'Waymo':
                self.tasks = [
                    dict(
                        num_class=1,
                        class_names=['Car'],
                        indices=[0],
                        radius=0.7),
                    dict(
                        num_class=1,
                        class_names=['Pedestrian'],
                        indices=[1],
                        radius=0.7),
                    dict(
                        num_class=1,
                        class_names=['Cyclist'],
                        indices=[2],
                        radius=0.7)
                ]
            ret_layer = []

            for i in range(batch_size):
                boxes3d = temp[i]['bboxes']
                scores = temp[i]['scores']
                labels = temp[i]['labels']
                if self.test_cfg['nms_type'] is not None:
                    keep_mask = torch.zeros_like(scores)
                    for task in self.tasks:
                        task_mask = torch.zeros_like(scores)
                        for cls_idx in task['indices']:
                            task_mask += labels == cls_idx
                        task_mask = task_mask.bool()
                        if task['radius'] > 0:
                            if self.test_cfg['nms_type'] == 'circle':
                                boxes_for_nms = torch.cat([
                                    boxes3d[task_mask][:, :2],
                                    scores[:, None][task_mask]
                                ],
                                                          dim=1)
                                task_keep_indices = torch.tensor(
                                    circle_nms(
                                        boxes_for_nms.detach().cpu().numpy(),
                                        task['radius']))
                            else:
                                boxes_for_nms = xywhr2xyxyr(
                                    img_metas[i]['box_type_3d'](
                                        boxes3d[task_mask][:, :7], 7).bev)
                                top_scores = scores[task_mask]
                                # nms_gpu → nms_bev
                                task_keep_indices = nms_bev(
                                    boxes_for_nms,
                                    top_scores,
                                    thresh=task['radius'],
                                    pre_maxsize=self.test_cfg['pre_maxsize'],
                                    post_max_size=self.test_cfg['post_maxsize']
                                )
                        else:
                            task_keep_indices = torch.arange(task_mask.sum())
                        if task_keep_indices.shape[0] != 0:
                            keep_indices = torch.where(
                                task_mask != 0)[0][task_keep_indices]
                            keep_mask[keep_indices] = 1
                    keep_mask = keep_mask.bool()
                    boxes3d = boxes3d[keep_mask]
                    scores = scores[keep_mask]
                    labels = labels[keep_mask]
                    if len(boxes3d) > 200:
                        inds = scores.argsort(descending=True)[:200]
                        boxes3d = boxes3d[inds]
                        scores = scores[inds]
                        labels = labels[inds]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                else:
                    if len(boxes3d) > 200:
                        inds = scores.argsort(descending=True)[:200]
                        boxes3d = boxes3d[inds]
                        scores = scores[inds]
                        labels = labels[inds]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                ret_layer.append(ret)
            rets.append(ret_layer)

        assert len(rets) == 1

        assert len(rets[0]) == 1

        res = [[
            img_metas[0]['box_type_3d'](
                rets[0][0]['bboxes'], box_dim=rets[0][0]['bboxes'].shape[-1]),
            rets[0][0]['scores'], rets[0][0]['labels'].int()
        ]]

        return res

    def get_heatmap_targets(self, gt_bboxes_3d, gt_labels_3d):
        heatmaps, anno_boxes, inds, masks = multi_apply(
            self.get_heatmap_targets_single, gt_bboxes_3d, gt_labels_3d)

        heatmaps = np.array(heatmaps).transpose(1, 0).tolist()
        heatmaps = [torch.stack(hms_) for hms_ in heatmaps]

        anno_boxes = np.array(anno_boxes).transpose(1, 0).tolist()
        anno_boxes = [torch.stack(anno_boxes_) for anno_boxes_ in anno_boxes]

        inds = np.array(inds).transpose(1, 0).tolist()
        inds = [torch.stack(inds_) for inds_ in inds]

        masks = np.array(masks).transpose(1, 0).tolist()
        masks = [torch.stack(masks_) for masks_ in masks]

        return heatmaps, anno_boxes, inds, masks

    def get_heatmap_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        device = gt_labels_3d.device

        gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]),
            dim=1).to(device)

        max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg']

        grid_size = torch.tensor(self.train_cfg['grid_size'])

        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])

        voxel_size = torch.tensor(self.train_cfg['voxel_size'])

        feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']

        task_masks = []

        flag = 0

        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []

        task_classes = []

        flag2 = 0

        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []

            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                task_class.append(gt_labels_3d[m] + 1 - flag2)
            task_boxes.append(torch.cat(task_box, axis=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)

        draw_gaussian = draw_heatmap_gaussian

        heatmaps, anno_boxes, inds, masks = [], [], [], []

        for idx in range(len(self.heatmap_tasks)):
            heatmap = gt_bboxes_3d.new_zeros(
                (len(self.class_names[idx]), feature_map_size[1],
                 feature_map_size[0]))
            anno_box = gt_bboxes_3d.new_zeros((max_objs, 10),
                                              dtype=torch.float32)
            ind = gt_labels_3d.new_zeros((max_objs), dtype=torch.int64)
            mask = gt_bboxes_3d.new_zeros((max_objs), dtype=torch.uint8)
            num_objs = min(task_boxes[idx].shape[0], max_objs)

            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1
                width = task_boxes[idx][k][3]
                length = task_boxes[idx][k][4]
                width = width / voxel_size[0] / self.train_cfg[
                    'out_size_factor']
                length = length / voxel_size[1] / self.train_cfg[
                    'out_size_factor']
                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (length, width),
                        min_overlap=self.train_cfg['gaussian_overlap'])
                    radius = max(self.train_cfg['min_radius'], int(radius))
                    x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][
                        1], task_boxes[idx][k][2]
                    coor_x = (
                        x - pc_range[0]
                    ) / voxel_size[0] / self.train_cfg['out_size_factor']
                    coor_y = (
                        y - pc_range[1]
                    ) / voxel_size[1] / self.train_cfg['out_size_factor']
                    center = torch.tensor([coor_x, coor_y],
                                          dtype=torch.float32,
                                          device=device)
                    center_int = center.to(torch.int32)
                    if not (0 <= center_int[0] < feature_map_size[0]
                            and 0 <= center_int[1] < feature_map_size[1]):
                        continue
                    draw_gaussian(heatmap[cls_id], center_int, radius)
                    new_idx = k
                    x, y = center_int[0], center_int[1]

                    assert (y * feature_map_size[0] + x <
                            feature_map_size[0] * feature_map_size[1])
                    ind[new_idx] = y * feature_map_size[0] + x
                    mask[new_idx] = 1
                    vx, vy = task_boxes[idx][k][7:]
                    rot = task_boxes[idx][k][6]
                    box_dim = task_boxes[idx][k][3:6]
                    if self.norm_bbox:
                        box_dim = box_dim.log()
                    anno_box[new_idx] = torch.cat([
                        center - torch.tensor([x, y], device=device),
                        z.unsqueeze(0), box_dim,
                        torch.sin(rot).unsqueeze(0),
                        torch.cos(rot).unsqueeze(0),
                        vx.unsqueeze(0),
                        vy.unsqueeze(0)
                    ])
            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            masks.append(mask)
            inds.append(ind)

        return heatmaps, anno_boxes, inds, masks

    def get_heatmap_bboxes(self,
                           preds_dicts,
                           img_metas,
                           img=None,
                           rescale=False):
        rets = []

        for layer_id, preds_dict in enumerate(preds_dicts):
            batch_size = preds_dict[0]['heatmap'].shape[0]
            one_hot = F.one_hot(
                self.query_labels,
                num_classes=self.num_classes).permute(0, 2, 1)
            batch_score = preds_dict[0]['query_heatmap_score'] * one_hot
            query_box = preds_dict[0]['query_box']
            rot = query_box[:, 6:8]
            dim = query_box[:, 3:6]
            center = query_box[:, 0:2]
            height = query_box[:, 2:3]
            vel = query_box[:, 8:]
            temp = self.bbox_coder.decode(
                batch_score,
                rot.clone(),
                dim.clone(),
                center.clone(),
                height.clone(),
                vel.clone(),
                filter=True)
            self.test_cfg['nms_type'] = 'circle'
            self.test_cfg['use_rotate_nms'] = True
            self.test_cfg['max_num'] = 500
            self.test_cfg['nms_thr'] = 0.2
            self.test_cfg['score_threshold'] = 0.01
            self.test_cfg['pre_maxsize'] = 600
            self.test_cfg['post_maxsize'] = 200

            if self.test_cfg['dataset'] == 'nuScenes':
                self.tasks = [
                    dict(
                        num_class=8,
                        class_names=[],
                        indices=[0, 1, 2, 3, 4, 5, 6, 7],
                        radius=-1),
                    dict(
                        num_class=1,
                        class_names=['pedestrian'],
                        indices=[8],
                        radius=0.175),
                    dict(
                        num_class=1,
                        class_names=['traffic_cone'],
                        indices=[9],
                        radius=0.175)
                ]

            elif self.test_cfg['dataset'] == 'Waymo':
                self.tasks = [
                    dict(
                        num_class=1,
                        class_names=['Car'],
                        indices=[0],
                        radius=0.7),
                    dict(
                        num_class=1,
                        class_names=['Pedestrian'],
                        indices=[1],
                        radius=0.7),
                    dict(
                        num_class=1,
                        class_names=['Cyclist'],
                        indices=[2],
                        radius=0.7)
                ]
            ret_layer = []

            for i in range(batch_size):
                boxes3d = temp[i]['bboxes']
                scores = temp[i]['scores']
                labels = temp[i]['labels']
                if self.test_cfg['nms_type'] is not None:
                    keep_mask = torch.zeros_like(scores)
                    for task in self.tasks:
                        task_mask = torch.zeros_like(scores)
                        for cls_idx in task['indices']:
                            task_mask += labels == cls_idx
                        task_mask = task_mask.bool()
                        if task['radius'] > 0:
                            if self.test_cfg['nms_type'] == 'circle':
                                boxes_for_nms = torch.cat([
                                    boxes3d[task_mask][:, :2],
                                    scores[:, None][task_mask]
                                ],
                                                          dim=1)
                                task_keep_indices = torch.tensor(
                                    circle_nms(
                                        boxes_for_nms.detach().cpu().numpy(),
                                        task['radius']))
                            else:
                                boxes_for_nms = xywhr2xyxyr(
                                    img_metas[i]['box_type_3d'](
                                        boxes3d[task_mask][:, :7], 7).bev)
                                top_scores = scores[task_mask]
                                # nms_gpu → nms_bev
                                task_keep_indices = nms_bev(
                                    boxes_for_nms,
                                    top_scores,
                                    thresh=task['radius'],
                                    pre_maxsize=self.test_cfg['pre_maxsize'],
                                    post_max_size=self.test_cfg['post_maxsize']
                                )
                        else:
                            task_keep_indices = torch.arange(task_mask.sum())
                        if task_keep_indices.shape[0] != 0:
                            keep_indices = torch.where(
                                task_mask != 0)[0][task_keep_indices]
                            keep_mask[keep_indices] = 1
                    keep_mask = keep_mask.bool()
                    boxes3d = boxes3d[keep_mask]
                    scores = scores[keep_mask]
                    labels = labels[keep_mask]
                    if len(boxes3d) > 200:
                        inds = scores.argsort(descending=True)[:200]
                        boxes3d = boxes3d[inds]
                        scores = scores[inds]
                        labels = labels[inds]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                else:
                    if len(boxes3d) > 200:
                        inds = scores.argsort(descending=True)[:200]
                        boxes3d = boxes3d[inds]
                        scores = scores[inds]
                        labels = labels[inds]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                ret_layer.append(ret)
            rets.append(ret_layer)

        assert len(rets) == 1

        assert len(rets[0]) == 1

        res = [[
            img_metas[0]['box_type_3d'](
                rets[0][0]['bboxes'], box_dim=rets[0][0]['bboxes'].shape[-1]),
            rets[0][0]['scores'], rets[0][0]['labels'].int()
        ]]

        return res

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size))

        dense_idx = faked_features.nonzero()
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:5]

        roi_grid_points = (
            (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) -
            local_roi_size.unsqueeze(dim=1) / 2)

        return roi_grid_points


def _gather_feat(feat, ind, mask=None):
    """Gather feature map.

    Given feature map and index, return indexed feature map.

    Args:
        feat (torch.tensor): Feature map with the shape of [B, H*W, 10].
        ind (torch.Tensor): Index of the ground truth boxes with the
            shape of [B, max_obj].
        mask (torch.Tensor): Mask of the feature map with the shape
            of [B, max_obj]. Default: None.

    Returns:
        torch.Tensor: Feature map after gathering with the shape
            of [B, max_obj, 10].
    """
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)

    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)

        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat
