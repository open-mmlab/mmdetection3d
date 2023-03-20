# modify from https://github.com/TuSimple/centerformer/blob/master/det3d/models/necks/rpn_transformer.py # noqa

from typing import List, Tuple

import numpy as np
import torch
from mmcv.cnn import build_norm_layer
from mmdet.models.utils import multi_apply
from mmengine.logging import print_log
from mmengine.structures import InstanceData
from torch import Tensor, nn

from mmdet3d.models.utils import draw_heatmap_gaussian, gaussian_radius
from mmdet3d.registry import MODELS
from mmdet3d.structures import center_to_corner_box2d
from .transformer import DeformableTransformerDecoder


class ChannelAttention(nn.Module):

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(
            2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        return self.sigmoid(y) * x


class MultiFrameSpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(MultiFrameSpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(
            2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, curr, prev):
        avg_out = torch.mean(curr, dim=1, keepdim=True)
        max_out, _ = torch.max(curr, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        return self.sigmoid(y) * prev


class BaseDecoderRPN(nn.Module):

    def __init__(
            self,
            layer_nums,  # [2,2,2]
            ds_num_filters,  # [128,256,64]
            num_input_features,  # 256
            transformer_config=None,
            hm_head_layer=2,
            corner_head_layer=2,
            corner=False,
            assign_label_window_size=1,
            classes=3,
            use_gt_training=False,
            norm_cfg=None,
            logger=None,
            init_bias=-2.19,
            score_threshold=0.1,
            obj_num=500,
            **kwargs):
        super(BaseDecoderRPN, self).__init__()
        self._layer_strides = [1, 2, -4]
        self._num_filters = ds_num_filters
        self._layer_nums = layer_nums
        self._num_input_features = num_input_features
        self.score_threshold = score_threshold
        self.transformer_config = transformer_config
        self.corner = corner
        self.obj_num = obj_num
        self.use_gt_training = use_gt_training
        self.window_size = assign_label_window_size**2
        self.cross_attention_kernel_size = [3, 3, 3]
        self.batch_id = None

        if norm_cfg is None:
            norm_cfg = dict(type='BN', eps=1e-3, momentum=0.01)
        self._norm_cfg = norm_cfg

        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)
        assert self.transformer_config is not None

        in_filters = [
            self._num_input_features,
            self._num_filters[0],
            self._num_filters[1],
        ]
        blocks = []

        for i, layer_num in enumerate(self._layer_nums):
            block, num_out_filters = self._make_layer(
                in_filters[i],
                self._num_filters[i],
                layer_num,
                stride=self._layer_strides[i],
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                self._num_filters[0],
                self._num_filters[2],
                2,
                stride=2,
                bias=False),
            build_norm_layer(self._norm_cfg, self._num_filters[2])[1],
            nn.ReLU())
        # heatmap prediction
        hm_head = []
        for i in range(hm_head_layer - 1):
            hm_head.append(
                nn.Conv2d(
                    self._num_filters[-1] * 2,
                    64,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ))
            hm_head.append(build_norm_layer(self._norm_cfg, 64)[1])
            hm_head.append(nn.ReLU())

        hm_head.append(
            nn.Conv2d(
                64, classes, kernel_size=3, stride=1, padding=1, bias=True))
        hm_head[-1].bias.data.fill_(init_bias)
        self.hm_head = nn.Sequential(*hm_head)

        if self.corner:
            self.corner_head = []
            for i in range(corner_head_layer - 1):
                self.corner_head.append(
                    nn.Conv2d(
                        self._num_filters[-1] * 2,
                        64,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                    ))
                self.corner_head.append(
                    build_norm_layer(self._norm_cfg, 64)[1])
                self.corner_head.append(nn.ReLU())

            self.corner_head.append(
                nn.Conv2d(
                    64, 1, kernel_size=3, stride=1, padding=1, bias=True))
            self.corner_head[-1].bias.data.fill_(init_bias)
            self.corner_head = nn.Sequential(*self.corner_head)

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):

        if stride > 0:
            block = [
                nn.ZeroPad2d(1),
                nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
                build_norm_layer(self._norm_cfg, planes)[1],
                nn.ReLU(),
            ]
        else:
            block = [
                nn.ConvTranspose2d(
                    inplanes, planes, -stride, stride=-stride, bias=False),
                build_norm_layer(self._norm_cfg, planes)[1],
                nn.ReLU(),
            ]

        for j in range(num_blocks):
            block.append(nn.Conv2d(planes, planes, 3, padding=1, bias=False))
            block.append(build_norm_layer(self._norm_cfg, planes)[1], )
            block.append(nn.ReLU())

        block.append(ChannelAttention(planes))
        block.append(SpatialAttention())
        block = nn.Sequential(*block)

        return block, planes

    def forward(self, x, example=None):
        pass

    def get_multi_scale_feature(self, center_pos, feats):
        """
        Args:
            center_pos: center coor at the lowest scale feature map [B 500 2]
            feats: multi scale BEV feature 3*[B C H W]
        Returns:
            neighbor_feat: [B 500 K C]
            neighbor_pos: [B 500 K 2]
        """
        kernel_size = self.cross_attention_kernel_size
        batch, num_cls, H, W = feats[0].size()

        center_num = center_pos.shape[1]

        relative_pos_list = []
        neighbor_feat_list = []
        for i, k in enumerate(kernel_size):
            neighbor_coords = torch.arange(-(k // 2), (k // 2) + 1)
            neighbor_coords = torch.flatten(
                torch.stack(
                    torch.meshgrid([neighbor_coords, neighbor_coords]), dim=0),
                1,
            )  # [2, k]
            neighbor_coords = (neighbor_coords.permute(
                1,
                0).contiguous().to(center_pos))  # relative coordinate [k, 2]
            neighbor_coords = (center_pos[:, :, None, :] // (2**i) +
                               neighbor_coords[None, None, :, :]
                               )  # coordinates [B, 500, k, 2]
            neighbor_coords = torch.clamp(
                neighbor_coords, min=0,
                max=H // (2**i) - 1)  # prevent out of bound
            feat_id = (neighbor_coords[:, :, :, 1] * (W // (2**i)) +
                       neighbor_coords[:, :, :, 0])  # pixel id [B, 500, k]
            feat_id = feat_id.reshape(batch, -1)  # pixel id [B, 500*k]
            selected_feat = (
                feats[i].reshape(batch, num_cls, (H * W) // (4**i)).permute(
                    0, 2, 1).contiguous()[self.batch_id.repeat(1, k**2),
                                          feat_id])  # B, 500*k, C
            neighbor_feat_list.append(
                selected_feat.reshape(batch, center_num, -1,
                                      num_cls))  # B, 500, k, C
            relative_pos_list.append(neighbor_coords * (2**i))  # B, 500, k, 2

        neighbor_pos = torch.cat(relative_pos_list, dim=2)  # B, 500, K, 2/3
        neighbor_feats = torch.cat(neighbor_feat_list, dim=2)  # B, 500, K, C
        return neighbor_feats, neighbor_pos

    def get_multi_scale_feature_multiframe(self, center_pos, feats, timeframe):
        """
        Args:
            center_pos: center coor at the lowest scale feature map [B 500 2]
            feats: multi scale BEV feature (3+k)*[B C H W]
            timeframe: timeframe [B,k]
        Returns:
            neighbor_feat: [B 500 K C]
            neighbor_pos: [B 500 K 2]
            neighbor_time: [B 500 K 1]
        """
        kernel_size = self.cross_attention_kernel_size
        batch, num_cls, H, W = feats[0].size()

        center_num = center_pos.shape[1]

        relative_pos_list = []
        neighbor_feat_list = []
        timeframe_list = []
        for i, k in enumerate(kernel_size):
            neighbor_coords = torch.arange(-(k // 2), (k // 2) + 1)
            neighbor_coords = torch.flatten(
                torch.stack(
                    torch.meshgrid([neighbor_coords, neighbor_coords]), dim=0),
                1,
            )  # [2, k]
            neighbor_coords = (neighbor_coords.permute(
                1,
                0).contiguous().to(center_pos))  # relative coordinate [k, 2]
            neighbor_coords = (center_pos[:, :, None, :] // (2**i) +
                               neighbor_coords[None, None, :, :]
                               )  # coordinates [B, 500, k, 2]
            neighbor_coords = torch.clamp(
                neighbor_coords, min=0,
                max=H // (2**i) - 1)  # prevent out of bound
            feat_id = (neighbor_coords[:, :, :, 1] * (W // (2**i)) +
                       neighbor_coords[:, :, :, 0])  # pixel id [B, 500, k]
            feat_id = feat_id.reshape(batch, -1)  # pixel id [B, 500*k]
            selected_feat = (
                feats[i].reshape(batch, num_cls, (H * W) // (4**i)).permute(
                    0, 2, 1).contiguous()[self.batch_id.repeat(1, k**2),
                                          feat_id])  # B, 500*k, C
            neighbor_feat_list.append(
                selected_feat.reshape(batch, center_num, -1,
                                      num_cls))  # B, 500, k, C
            relative_pos_list.append(neighbor_coords * (2**i))  # B, 500, k, 2
            timeframe_list.append(
                torch.full_like(neighbor_coords[:, :, :, 0:1], 0))  # B, 500, k
            if i == 0:
                # add previous frame feature
                for frame_num in range(feats[-1].shape[1]):
                    selected_feat = (feats[-1][:, frame_num, :, :, :].reshape(
                        batch, num_cls, (H * W) // (4**i)).permute(
                            0, 2,
                            1).contiguous()[self.batch_id.repeat(1, k**2),
                                            feat_id])  # B, 500*k, C
                    neighbor_feat_list.append(
                        selected_feat.reshape(batch, center_num, -1, num_cls))
                    relative_pos_list.append(neighbor_coords * (2**i))
                    time = timeframe[:, frame_num + 1].to(selected_feat)  # B
                    timeframe_list.append(
                        time[:, None, None, None] * torch.full_like(
                            neighbor_coords[:, :, :, 0:1], 1))  # B, 500, k

        neighbor_pos = torch.cat(relative_pos_list, dim=2)  # B, 500, K, 2/3
        neighbor_feats = torch.cat(neighbor_feat_list, dim=2)  # B, 500, K, C
        neighbor_time = torch.cat(timeframe_list, dim=2)  # B, 500, K, 1

        return neighbor_feats, neighbor_pos, neighbor_time


@MODELS.register_module()
class DeformableDecoderRPN(BaseDecoderRPN):
    """The original implement of CenterFormer modules.

    It fuse the backbone, neck and heatmap head into one module. The backbone
    is `SECOND` with attention and the neck is `SECONDFPN` with attention.

    TODO: split this module into backbone、neck and head.
    """

    def __init__(self,
                 layer_nums,
                 ds_num_filters,
                 num_input_features,
                 tasks=dict(),
                 transformer_config=None,
                 hm_head_layer=2,
                 corner_head_layer=2,
                 corner=False,
                 parametric_embedding=False,
                 assign_label_window_size=1,
                 classes=3,
                 use_gt_training=False,
                 norm_cfg=None,
                 logger=None,
                 init_bias=-2.19,
                 score_threshold=0.1,
                 obj_num=500,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        super(DeformableDecoderRPN, self).__init__(
            layer_nums,
            ds_num_filters,
            num_input_features,
            transformer_config,
            hm_head_layer,
            corner_head_layer,
            corner,
            assign_label_window_size,
            classes,
            use_gt_training,
            norm_cfg,
            logger,
            init_bias,
            score_threshold,
            obj_num,
        )
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.tasks = tasks
        self.class_names = [t['class_names'] for t in tasks]

        self.transformer_decoder = DeformableTransformerDecoder(
            self._num_filters[-1] * 2,
            depth=transformer_config.depth,
            n_heads=transformer_config.n_heads,
            dim_single_head=transformer_config.dim_single_head,
            dim_ffn=transformer_config.dim_ffn,
            dropout=transformer_config.dropout,
            out_attention=transformer_config.out_attn,
            n_points=transformer_config.get('n_points', 9),
        )
        self.pos_embedding_type = transformer_config.get(
            'pos_embedding_type', 'linear')
        if self.pos_embedding_type == 'linear':
            self.pos_embedding = nn.Linear(2, self._num_filters[-1] * 2)
        else:
            raise NotImplementedError()
        self.parametric_embedding = parametric_embedding
        if self.parametric_embedding:
            self.query_embed = nn.Embedding(self.obj_num,
                                            self._num_filters[-1] * 2)
            nn.init.uniform_(self.query_embed.weight, -1.0, 1.0)

        print_log('Finish RPN_transformer_deformable Initialization',
                  'current')

    def _sigmoid(self, x):
        y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
        return y

    def forward(self, x, batch_data_samples):

        batch_gt_instance_3d = []
        for data_sample in batch_data_samples:
            batch_gt_instance_3d.append(data_sample.gt_instances_3d)

        # FPN
        x = self.blocks[0](x)
        x_down = self.blocks[1](x)
        x_up = torch.cat([self.blocks[2](x_down), self.up(x)], dim=1)

        # heatmap head
        hm = self.hm_head(x_up)

        if self.corner and self.corner_head.training:
            corner_hm = self.corner_head(x_up)
            corner_hm = self._sigmoid(corner_hm)

        # find top K center location
        hm = self._sigmoid(hm)
        batch, num_cls, H, W = hm.size()

        scores, labels = torch.max(
            hm.reshape(batch, num_cls, H * W), dim=1)  # b,H*W
        self.batch_id = torch.from_numpy(np.indices(
            (batch, self.obj_num))[0]).to(labels)

        if self.training:
            heatmaps, anno_boxes, gt_inds, gt_masks, corner_heatmaps, cat_labels = self.get_targets(  # noqa: E501
                batch_gt_instance_3d)
            batch_targets = dict(
                ind=gt_inds,
                mask=gt_masks,
                hm=heatmaps,
                anno_box=anno_boxes,
                corners=corner_heatmaps,
                cat=cat_labels)
            inds = gt_inds[0][:, (self.window_size // 2)::self.window_size]
            masks = gt_masks[0][:, (self.window_size // 2)::self.window_size]
            batch_id_gt = torch.from_numpy(
                np.indices((batch, inds.shape[1]))[0]).to(labels)
            scores[batch_id_gt, inds] = scores[batch_id_gt, inds] + masks
            order = scores.sort(1, descending=True)[1]
            order = order[:, :self.obj_num]
            scores[batch_id_gt, inds] = scores[batch_id_gt, inds] - masks
        else:
            order = scores.sort(1, descending=True)[1]
            order = order[:, :self.obj_num]
            batch_targets = None

        scores = torch.gather(scores, 1, order)
        labels = torch.gather(labels, 1, order)
        mask = scores > self.score_threshold

        ct_feat = x_up.reshape(batch, -1, H * W).transpose(2, 1).contiguous()
        ct_feat = ct_feat[self.batch_id, order]  # B, 500, C

        # create position embedding for each center
        y_coor = order // W
        x_coor = order - y_coor * W
        y_coor, x_coor = y_coor.to(ct_feat), x_coor.to(ct_feat)
        y_coor, x_coor = y_coor / H, x_coor / W
        pos_features = torch.stack([x_coor, y_coor], dim=2)

        if self.parametric_embedding:
            ct_feat = self.query_embed.weight
            ct_feat = ct_feat.unsqueeze(0).expand(batch, -1, -1)

        # run transformer
        src = torch.cat(
            (
                x_up.reshape(batch, -1, H * W).transpose(2, 1).contiguous(),
                x.reshape(batch, -1,
                          (H * W) // 4).transpose(2, 1).contiguous(),
                x_down.reshape(batch, -1,
                               (H * W) // 16).transpose(2, 1).contiguous(),
            ),
            dim=1,
        )  # B ,sum(H*W), C
        spatial_shapes = torch.as_tensor(
            [(H, W), (H // 2, W // 2), (H // 4, W // 4)],
            dtype=torch.long,
            device=ct_feat.device,
        )
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1, )),
            spatial_shapes.prod(1).cumsum(0)[:-1],
        ))

        transformer_out = self.transformer_decoder(
            ct_feat,
            self.pos_embedding,
            src,
            spatial_shapes,
            level_start_index,
            center_pos=pos_features,
        )  # (B,N,C)

        ct_feat = (transformer_out['ct_feat'].transpose(2, 1).contiguous()
                   )  # B, C, 500

        out_dict = {
            'hm': hm,
            'scores': scores,
            'labels': labels,
            'order': order,
            'ct_feat': ct_feat,
            'mask': mask,
        }
        if 'out_attention' in transformer_out:
            out_dict.update(
                {'out_attention': transformer_out['out_attention']})
        if self.corner and self.corner_head.training:
            out_dict.update({'corner_hm': corner_hm})

        return out_dict, batch_targets

    def get_targets(
        self,
        batch_gt_instances_3d: List[InstanceData],
    ) -> Tuple[List[Tensor]]:
        """Generate targets. How each output is transformed: Each nested list
        is transposed so that all same-index elements in each sub-list (1, ...,
        N) become the new sub-lists.

                [ [a0, a1, a2, ... ], [b0, b1, b2, ... ], ... ]
                ==> [ [a0, b0, ... ], [a1, b1, ... ], [a2, b2, ... ] ]
            The new transposed nested list is converted into a list of N
            tensors generated by concatenating tensors in the new sub-lists.
                [ tensor0, tensor1, tensor2, ... ]
        Args:
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances. It usually includes ``bboxes_3d`` and
                ``labels_3d`` attributes.
        Returns:
            Returns:
                tuple[list[torch.Tensor]]: Tuple of target including
                    the following results in order.
                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the
                    position of the valid boxes.
                - list[torch.Tensor]: Masks indicating which
                    boxes are valid.
                - list[torch.Tensor]: catagrate labels.
        """
        heatmaps, anno_boxes, inds, masks, corner_heatmaps, cat_labels = multi_apply(  # noqa: E501
            self.get_targets_single, batch_gt_instances_3d)
        # Transpose heatmaps
        heatmaps = list(map(list, zip(*heatmaps)))
        heatmaps = [torch.stack(hms_) for hms_ in heatmaps]
        # Transpose heatmaps
        corner_heatmaps = list(map(list, zip(*corner_heatmaps)))
        corner_heatmaps = [torch.stack(hms_) for hms_ in corner_heatmaps]
        # Transpose anno_boxes
        anno_boxes = list(map(list, zip(*anno_boxes)))
        anno_boxes = [torch.stack(anno_boxes_) for anno_boxes_ in anno_boxes]
        # Transpose inds
        inds = list(map(list, zip(*inds)))
        inds = [torch.stack(inds_) for inds_ in inds]
        # Transpose inds
        masks = list(map(list, zip(*masks)))
        masks = [torch.stack(masks_) for masks_ in masks]
        # Transpose cat_labels
        cat_labels = list(map(list, zip(*cat_labels)))
        cat_labels = [torch.stack(labels_) for labels_ in cat_labels]
        return heatmaps, anno_boxes, inds, masks, corner_heatmaps, cat_labels

    def get_targets_single(self,
                           gt_instances_3d: InstanceData) -> Tuple[Tensor]:
        """Generate training targets for a single sample.
        Args:
            gt_instances_3d (:obj:`InstanceData`): Gt_instances of
                single data sample. It usually includes
                ``bboxes_3d`` and ``labels_3d`` attributes.
        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including
                the following results in order.
                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes
                    are valid.
                - list[torch.Tensor]: catagrate labels.
        """
        gt_labels_3d = gt_instances_3d.labels_3d
        gt_bboxes_3d = gt_instances_3d.bboxes_3d
        device = gt_labels_3d.device
        gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]),
            dim=1).to(device)
        max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg']
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])

        feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']

        # reorganize the gt_dict by tasks
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
                # 0 is background for each task, so we need to add 1 here.
                task_class.append(gt_labels_3d[m] + 1 - flag2)
            task_boxes.append(torch.cat(task_box, axis=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)
        draw_gaussian = draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks, corner_heatmaps, cat_labels = [], [], [], [], [], []  # noqa: E501

        for idx in range(len(self.tasks)):
            heatmap = gt_bboxes_3d.new_zeros(
                (len(self.class_names[idx]), feature_map_size[1],
                 feature_map_size[0]))
            corner_heatmap = torch.zeros(
                (1, feature_map_size[1], feature_map_size[0]),
                dtype=torch.float32,
                device=device)

            anno_box = gt_bboxes_3d.new_zeros((max_objs, 8),
                                              dtype=torch.float32)

            ind = gt_labels_3d.new_zeros((max_objs), dtype=torch.int64)
            mask = gt_bboxes_3d.new_zeros((max_objs), dtype=torch.uint8)
            cat_label = gt_bboxes_3d.new_zeros((max_objs), dtype=torch.int64)

            num_objs = min(task_boxes[idx].shape[0], max_objs)

            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1

                # gt boxes [xyzlwhr]
                length = task_boxes[idx][k][3]
                width = task_boxes[idx][k][4]
                length = length / voxel_size[0] / self.train_cfg[
                    'out_size_factor']
                width = width / voxel_size[1] / self.train_cfg[
                    'out_size_factor']

                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (width, length),
                        min_overlap=self.train_cfg['gaussian_overlap'])
                    radius = max(self.train_cfg['min_radius'], int(radius))

                    # be really careful for the coordinate system of
                    # your box annotation.
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

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (0 <= center_int[0] < feature_map_size[0]
                            and 0 <= center_int[1] < feature_map_size[1]):
                        continue

                    draw_gaussian(heatmap[cls_id], center_int, radius)

                    radius = radius // 2
                    # # draw four corner and center TODO: use torch
                    rot = task_boxes[idx][k][6]
                    corner_keypoints = center_to_corner_box2d(
                        center.unsqueeze(0).cpu().numpy(),
                        torch.tensor([[length, width]],
                                     dtype=torch.float32).numpy(),
                        angles=rot,
                        origin=0.5)
                    corner_keypoints = torch.from_numpy(corner_keypoints).to(
                        center)

                    draw_gaussian(corner_heatmap[0], center_int, radius)
                    draw_gaussian(
                        corner_heatmap[0],
                        (corner_keypoints[0, 0] + corner_keypoints[0, 1]) / 2,
                        radius)
                    draw_gaussian(
                        corner_heatmap[0],
                        (corner_keypoints[0, 2] + corner_keypoints[0, 3]) / 2,
                        radius)
                    draw_gaussian(
                        corner_heatmap[0],
                        (corner_keypoints[0, 0] + corner_keypoints[0, 3]) / 2,
                        radius)
                    draw_gaussian(
                        corner_heatmap[0],
                        (corner_keypoints[0, 1] + corner_keypoints[0, 2]) / 2,
                        radius)

                    new_idx = k
                    x, y = center_int[0], center_int[1]

                    assert (y * feature_map_size[0] + x <
                            feature_map_size[0] * feature_map_size[1])

                    ind[new_idx] = y * feature_map_size[0] + x
                    mask[new_idx] = 1
                    cat_label[new_idx] = cls_id
                    # TODO: support other outdoor dataset
                    # vx, vy = task_boxes[idx][k][7:]
                    rot = task_boxes[idx][k][6]
                    box_dim = task_boxes[idx][k][3:6]
                    box_dim = box_dim.log()
                    anno_box[new_idx] = torch.cat([
                        center - torch.tensor([x, y], device=device),
                        z.unsqueeze(0), box_dim,
                        torch.sin(rot).unsqueeze(0),
                        torch.cos(rot).unsqueeze(0)
                    ])

            heatmaps.append(heatmap)
            corner_heatmaps.append(corner_heatmap)
            anno_boxes.append(anno_box)
            masks.append(mask)
            inds.append(ind)
            cat_labels.append(cat_label)
        return heatmaps, anno_boxes, inds, masks, corner_heatmaps, cat_labels


@MODELS.register_module()
class MultiFrameDeformableDecoderRPN(BaseDecoderRPN):
    """The original implementation of CenterFormer modules.

    The difference between this module and
    `DeformableDecoderRPN` is that this module uses information from multi
    frames.

    TODO: split this module into backbone、neck and head.
    """

    def __init__(
            self,
            layer_nums,  # [2,2,2]
            ds_num_filters,  # [128,256,64]
            num_input_features,  # 256
            transformer_config=None,
            hm_head_layer=2,
            corner_head_layer=2,
            corner=False,
            parametric_embedding=False,
            assign_label_window_size=1,
            classes=3,
            use_gt_training=False,
            norm_cfg=None,
            logger=None,
            init_bias=-2.19,
            score_threshold=0.1,
            obj_num=500,
            frame=1,
            **kwargs):
        super(MultiFrameDeformableDecoderRPN, self).__init__(
            layer_nums,
            ds_num_filters,
            num_input_features,
            transformer_config,
            hm_head_layer,
            corner_head_layer,
            corner,
            assign_label_window_size,
            classes,
            use_gt_training,
            norm_cfg,
            logger,
            init_bias,
            score_threshold,
            obj_num,
        )
        self.frame = frame

        self.out = nn.Sequential(
            nn.Conv2d(
                self._num_filters[0] * frame,
                self._num_filters[0],
                3,
                padding=1,
                bias=False,
            ),
            build_norm_layer(self._norm_cfg, self._num_filters[0])[1],
            nn.ReLU(),
        )
        self.mtf_attention = MultiFrameSpatialAttention()
        self.time_embedding = nn.Linear(1, self._num_filters[0])

        self.transformer_decoder = DeformableTransformerDecoder(
            self._num_filters[-1] * 2,
            depth=transformer_config.depth,
            n_heads=transformer_config.n_heads,
            n_levels=2 + self.frame,
            dim_single_head=transformer_config.dim_single_head,
            dim_ffn=transformer_config.dim_ffn,
            dropout=transformer_config.dropout,
            out_attention=transformer_config.out_attn,
            n_points=transformer_config.get('n_points', 9),
        )
        self.pos_embedding_type = transformer_config.get(
            'pos_embedding_type', 'linear')
        if self.pos_embedding_type == 'linear':
            self.pos_embedding = nn.Linear(2, self._num_filters[-1] * 2)
        else:
            raise NotImplementedError()
        self.parametric_embedding = parametric_embedding
        if self.parametric_embedding:
            self.query_embed = nn.Embedding(self.obj_num,
                                            self._num_filters[-1] * 2)
            nn.init.uniform_(self.query_embed.weight, -1.0, 1.0)

        print_log('Finish RPN_transformer_deformable Initialization',
                  'current')

    def forward(self, x, example=None):

        # FPN
        x = self.blocks[0](x)
        x_down = self.blocks[1](x)
        x_up = torch.cat([self.blocks[2](x_down), self.up(x)], dim=1)

        # take out the BEV feature on current frame
        x = torch.split(x, self.frame)
        x_up = torch.split(x_up, self.frame)
        x_down = torch.split(x_down, self.frame)
        x_prev = torch.stack([t[1:] for t in x_up], dim=0)  # B,K,C,H,W
        x = torch.stack([t[0] for t in x], dim=0)
        x_down = torch.stack([t[0] for t in x_down], dim=0)

        x_up = torch.stack([t[0] for t in x_up], dim=0)  # B,C,H,W
        # use spatial attention in current frame on previous feature
        x_prev_cat = self.mtf_attention(
            x_up,
            x_prev.reshape(x_up.shape[0], -1, x_up.shape[2],
                           x_up.shape[3]))  # B,K*C,H,W
        # time embedding
        x_up_fuse = torch.cat((x_up, x_prev_cat), dim=1) + self.time_embedding(
            example['times'][:, :, None].to(x_up)).reshape(
                x_up.shape[0], -1, 1, 1)
        # fuse mtf feature
        x_up_fuse = self.out(x_up_fuse)

        # heatmap head
        hm = self.hm_head(x_up_fuse)

        if self.corner and self.corner_head.training:
            corner_hm = self.corner_head(x_up_fuse)
            corner_hm = torch.sigmoid(corner_hm)

        # find top K center location
        hm = torch.sigmoid(hm)
        batch, num_cls, H, W = hm.size()

        scores, labels = torch.max(
            hm.reshape(batch, num_cls, H * W), dim=1)  # b,H*W
        self.batch_id = torch.from_numpy(np.indices(
            (batch, self.obj_num))[0]).to(labels)

        if self.use_gt_training and self.hm_head.training:
            gt_inds = example['ind'][0][:, (self.window_size //
                                            2)::self.window_size]
            gt_masks = example['mask'][0][:, (self.window_size //
                                              2)::self.window_size]
            batch_id_gt = torch.from_numpy(
                np.indices((batch, gt_inds.shape[1]))[0]).to(labels)
            scores[batch_id_gt,
                   gt_inds] = scores[batch_id_gt, gt_inds] + gt_masks
            order = scores.sort(1, descending=True)[1]
            order = order[:, :self.obj_num]
            scores[batch_id_gt,
                   gt_inds] = scores[batch_id_gt, gt_inds] - gt_masks
        else:
            order = scores.sort(1, descending=True)[1]
            order = order[:, :self.obj_num]

        scores = torch.gather(scores, 1, order)
        labels = torch.gather(labels, 1, order)
        mask = scores > self.score_threshold

        ct_feat = (x_up.reshape(batch, -1,
                                H * W).transpose(2,
                                                 1).contiguous()[self.batch_id,
                                                                 order]
                   )  # B, 500, C

        # create position embedding for each center
        y_coor = order // W
        x_coor = order - y_coor * W
        y_coor, x_coor = y_coor.to(ct_feat), x_coor.to(ct_feat)
        y_coor, x_coor = y_coor / H, x_coor / W
        pos_features = torch.stack([x_coor, y_coor], dim=2)

        if self.parametric_embedding:
            ct_feat = self.query_embed.weight
            ct_feat = ct_feat.unsqueeze(0).expand(batch, -1, -1)

        # run transformer
        src_list = [
            x_up.reshape(batch, -1, H * W).transpose(2, 1).contiguous(),
            x.reshape(batch, -1, (H * W) // 4).transpose(2, 1).contiguous(),
            x_down.reshape(batch, -1, (H * W) // 16).transpose(2,
                                                               1).contiguous(),
        ]
        for frame in range(x_prev.shape[1]):
            src_list.append(x_prev[:, frame].reshape(batch,
                                                     -1, (H * W)).transpose(
                                                         2, 1).contiguous())
        src = torch.cat(src_list, dim=1)  # B ,sum(H*W), C
        spatial_list = [(H, W), (H // 2, W // 2), (H // 4, W // 4)]
        spatial_list += [(H, W) for frame in range(x_prev.shape[1])]
        spatial_shapes = torch.as_tensor(
            spatial_list, dtype=torch.long, device=ct_feat.device)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1, )),
            spatial_shapes.prod(1).cumsum(0)[:-1],
        ))

        transformer_out = self.transformer_decoder(
            ct_feat,
            self.pos_embedding,
            src,
            spatial_shapes,
            level_start_index,
            center_pos=pos_features,
        )  # (B,N,C)

        ct_feat = (transformer_out['ct_feat'].transpose(2, 1).contiguous()
                   )  # B, C, 500

        out_dict = {
            'hm': hm,
            'scores': scores,
            'labels': labels,
            'order': order,
            'ct_feat': ct_feat,
            'mask': mask,
        }
        if 'out_attention' in transformer_out:
            out_dict.update(
                {'out_attention': transformer_out['out_attention']})
        if self.corner and self.corner_head.training:
            out_dict.update({'corner_hm': corner_hm})

        return out_dict
