# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
from typing import Dict, List, Tuple

import numpy as np
import torch
from mmcv.cnn import build_norm_layer
from mmdet.models.utils import multi_apply
from mmengine.logging import print_log
from mmengine.model import BaseModule, kaiming_init
from mmengine.structures import InstanceData
from torch import Tensor, nn
from torch.nn.modules.conv import _ConvNd

from mmdet3d.models.utils import draw_heatmap_gaussian, gaussian_radius
from mmdet3d.models.utils.transformer import Deform_Transformer
from mmdet3d.registry import MODELS
from mmdet3d.structures import (Det3DDataSample, bbox_overlaps_3d,
                                center_to_corner_box2d)
from mmdet3d.structures.bbox_3d.box_torch_ops import rotate_nms_pcdet
from mmdet3d.structures.ops.iou3d_calculator import boxes_iou3d_gpu_pcdet
from ..layers import circle_nms, nms_bev


@MODELS.register_module()
class CenterFormerHead(BaseModule):

    def __init__(self,
                 in_channels: int = 256,
                 tasks: List = [],
                 common_heads: dict = dict(),
                 share_conv_channel: int = 64,
                 bbox_code_size: int = 7,
                 num_heatmap_convs: int = 2,
                 num_cornermap_convs: int = 2,
                 iou_factor: List = [1, 1, 4],
                 transformer_config=dict(
                     depth=2,
                     heads=6,
                     dim_head=64,
                     MLP_dim=256,
                     DP_rate=0.3,
                     out_att=False,
                     n_points=15,
                 ),
                 separate_head: dict = dict(
                     type='mmdet.SeparateHead',
                     init_bias=-2.19,
                     final_kernel=3),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 loss_cls=dict(
                     type='FastFocalLoss', reduction='mean', loss_weight=1),
                 loss_bbox=dict(
                     type='mmdet.L1Loss', reduction='mean', loss_weight=2),
                 loss_corner=dict(
                     type='mmdet.MSELoss', reduction='mean', loss_weight=1),
                 loss_iou=dict(
                     type='mmdet.SmoothL1Loss',
                     beta=1.0,
                     reduction='mean',
                     loss_weight=1),
                 test_cfg=None,
                 train_cfg=None,
                 init_cfg=None):
        super(CenterFormerHead, self).__init__(init_cfg=init_cfg)

        num_classes = [len(t['class_names']) for t in tasks]
        self.class_names = [t['class_names'] for t in tasks]
        self.tasks = tasks
        self.bbox_code_size = bbox_code_size
        self.iou_factor = iou_factor

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_iou = MODELS.build(loss_iou)
        self.loss_corner = MODELS.build(loss_corner)

        self.box_n_dim = 9 if 'vel' in common_heads else 7

        self.transformer_layer = Deform_Transformer(
            in_channels,
            depth=transformer_config.depth,
            heads=transformer_config.heads,
            dim_head=transformer_config.dim_head,
            mlp_dim=transformer_config.MLP_dim,
            dropout=transformer_config.DP_rate,
            out_attention=transformer_config.out_att,
            n_points=transformer_config.get('n_points', 9),
        )
        self.pos_embedding = nn.Linear(2, in_channels)

        heatmap_head = copy.deepcopy(separate_head)
        heatmap_head['conv_cfg'] = dict(type='Conv2d')
        heatmap_head['norm_cfg'] = dict(
            type='naiveSyncBN2d', eps=0.001, momentum=0.01)
        heatmap_head['final_kernel'] = 3
        heatmap_head.update(
            in_channels=in_channels,
            heads=dict(
                center_heatmap=(sum(num_classes), num_heatmap_convs),
                corner_heatmap=(1, num_cornermap_convs)),
            head_conv=share_conv_channel)
        self.heatmap_head = MODELS.build(heatmap_head)

        # a shared convolution
        self.shared_conv = nn.Sequential(
            nn.Conv1d(
                in_channels, share_conv_channel, kernel_size=1, bias=True),
            build_norm_layer(norm_cfg, share_conv_channel)[1],
            nn.ReLU(inplace=True),
        )
        self.task_heads = nn.ModuleList()
        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            separate_head.update(
                in_channels=share_conv_channel,
                heads=heads,
                head_conv=share_conv_channel)
            self.task_heads.append(MODELS.build(separate_head))

    def init_weights(self):
        self.heatmap_head.init_weights()
        for name, m in self.named_modules():
            if 'task_heads' in name and isinstance(m, _ConvNd):
                kaiming_init(m)
            elif 'heatmap_head' in name and isinstance(m, _ConvNd):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))

    def forward(self, x, *kwargs):
        ret_dicts = []

        y = self.shared_conv(x)

        for task in self.task_heads:
            ret_dicts.append(task(y))

        return ret_dicts

    def _sigmoid(self, x):
        y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
        return y

    def forward_heatmap(self, feat):
        outs = dict()
        outs.update(self.heatmap_head(feat))
        outs['hm'] = self._sigmoid(outs['center_heatmap'])
        outs['corner_hm'] = self._sigmoid(outs['corner_heatmap'])

        return outs

    def forward_decoder(self, query, feats, proposal_indices):
        # create position embedding for each center
        batch, num_cls, H, W = feats[-1].size()
        y_coor = proposal_indices // W
        x_coor = proposal_indices - y_coor * W
        y_coor, x_coor = y_coor.to(query), x_coor.to(query)
        y_coor, x_coor = y_coor / H, x_coor / W
        pos_features = torch.stack([x_coor, y_coor], dim=2)

        src = torch.cat(
            (feats[-1].reshape(batch, -1, H * W).transpose(2, 1).contiguous(),
             feats[0].reshape(batch, -1,
                              (H * W) // 4).transpose(2, 1).contiguous(),
             feats[1].reshape(batch, -1,
                              (H * W) // 16).transpose(2, 1).contiguous()),
            dim=1,
        )  # B ,sum(H*W), C
        spatial_shapes = torch.as_tensor(
            [(H, W), (H // 2, W // 2), (H // 4, W // 4)],
            dtype=torch.long,
            device=query.device,
        )
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1, )),
            spatial_shapes.prod(1).cumsum(0)[:-1],
        ))

        transformer_out = self.transformer_layer(
            query,
            self.pos_embedding,
            src,
            spatial_shapes,
            level_start_index,
            center_pos=pos_features,
        )  # (B,N,C)

        center_feat = transformer_out['ct_feat'].transpose(
            2, 1).contiguous()  # B, C, 500

        return center_feat

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
                gt_instances. It usually includes ``bboxes_3d`` and\
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
        """
        heatmaps, anno_boxes, inds, masks, corner_heatmaps, cat_labels = multi_apply(
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
        """
        gt_labels_3d = gt_instances_3d.labels_3d
        gt_bboxes_3d = gt_instances_3d.bboxes_3d
        device = gt_labels_3d.device
        gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]),
            dim=1).to(device)
        max_objs = self.train_cfg['num_center_proposals']
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
        heatmaps, anno_boxes, inds, masks, corner_heatmaps, cat_labels = [], [], [], [], [], []

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
                        torch.tensor([[width, length]],
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

    def loss(self, pts_feats: List[Tensor],
             batch_data_samples: List[Det3DDataSample], *args,
             **kwargs) -> Dict[str, Tensor]:

        batch_gt_instance_3d = []
        for data_sample in batch_data_samples:
            batch_gt_instance_3d.append(data_sample.gt_instances_3d)

        fpn_feat = pts_feats[-1]
        outs = dict()
        outs.update(self.forward_heatmap(fpn_feat))

        batch, num_cls, H, W = outs['hm'].size()
        scores, labels = torch.max(
            outs['hm'].reshape(batch, num_cls, H * W), dim=1)  # b, H*W

        heatmaps, anno_boxes, gt_inds, gt_masks, corner_heatmaps, cat_labels = self.get_targets(  # noqa: E501
            batch_gt_instance_3d)
        batch_targets = dict(
            ind=gt_inds,
            mask=gt_masks,
            hm=heatmaps,
            anno_box=anno_boxes,
            corners=corner_heatmaps,
            cat=cat_labels)

        inds = gt_inds[0]
        masks = gt_masks[0]
        batch_id_gt = torch.from_numpy(np.indices(
            (batch, inds.shape[1]))[0]).to(labels)
        scores[batch_id_gt, inds] = scores[batch_id_gt, inds] + masks
        order = scores.sort(1, descending=True)[1]
        order = order[:, :self.train_cfg['num_center_proposals']]
        scores[batch_id_gt, inds] = scores[batch_id_gt, inds] - masks

        # scores = torch.gather(scores, 1, order)
        # labels = torch.gather(labels, 1, order)

        query = fpn_feat.reshape(batch, -1, H * W).transpose(2, 1).contiguous()
        query = query[batch_id_gt, order]  # B, 500, C

        ct_feat = self.forward_decoder(query, pts_feats, order)

        # Only support one head now
        outs.update(self(ct_feat)[0])
        outs.update(dict(order=order))

        losses = self.loss_by_feat([outs], batch_targets)

        return losses

    def loss_by_feat(self, preds_dicts, example, **kwargs):
        losses = {}
        for task_id, preds_dict in enumerate(preds_dicts):
            # Heatmap focal loss
            hm_loss = self.loss_cls(
                preds_dict['hm'],
                example['hm'][task_id],
                example['ind'][task_id],
                example['mask'][task_id],
                example['cat'][task_id],
            )
            losses.update({f'task{task_id}.loss_heatmap': hm_loss})

            target_box = example['anno_box'][task_id]

            # Corner loss
            mask_corner_loss = example['corners'][task_id] > 0
            num_corners = mask_corner_loss.float().sum().item()

            corner_loss = self.loss_corner(
                preds_dict['corner_hm'][task_id],
                example['corners'][task_id],
                mask_corner_loss,
                avg_factor=(num_corners + 1e-4))

            losses.update({f'task{task_id}.loss_corner': corner_loss})

            # reconstruct the anno_box from multiple reg heads
            # (B, 7, num_proposals)
            preds_dict['anno_box'] = torch.cat(
                (
                    preds_dict['reg'],
                    preds_dict['height'],
                    preds_dict['dim'],
                    preds_dict['rot'],
                ),
                dim=1,
            )
            target_box = target_box[..., [0, 1, 2, 3, 4, 5, -2,
                                          -1]]  # remove vel target

            # get corresponding gt box # (B, num_proposals, 7)
            target_box, selected_mask, selected_cls = get_corresponding_box(
                preds_dict['order'],
                example['ind'][task_id],
                example['mask'][task_id],
                example['cat'][task_id],
                target_box,
            )
            mask = selected_mask.float().unsqueeze(2)  # (B, num_proposals, 1)

            # Bbox loss
            gt_num = mask.float().sum()
            isnotnan = (~torch.isnan(target_box)).float()
            mask_bbox_loss = mask * isnotnan
            bbox_weights = mask_bbox_loss * mask_bbox_loss.new_tensor(
                self.train_cfg['code_weights'])
            bbox_loss = self.loss_bbox(
                preds_dict['anno_box'].transpose(1, 2),
                target_box,
                bbox_weights,
                avg_factor=(gt_num + 1e-4))
            losses.update({f'task{task_id}.loss_bbox': bbox_loss})

            # IoU loss
            with torch.no_grad():
                preds_box = get_box(
                    preds_dict['anno_box'],
                    preds_dict['order'],
                    self.test_cfg,
                    preds_dict['hm'].shape[2],
                    preds_dict['hm'].shape[3],
                )
                cur_gt = get_box_gt(
                    target_box,
                    preds_dict['order'],
                    self.test_cfg,
                    preds_dict['hm'].shape[2],
                    preds_dict['hm'].shape[3],
                )
                # iou_targets = bbox_overlaps_3d(
                #     preds_box.reshape(-1, 7),
                #     cur_gt.reshape(-1, 7),
                #     coordinate='lidar')[
                #         range(preds_box.reshape(-1, 7).shape[0]),
                #         range(cur_gt.reshape(-1, 7).shape[0])]

                # (cx, cy, cz, l, w, h, theta)
                preds_box[:, :, 2] += preds_box[:, :, 5] / 2
                cur_gt[:, :, 2] += cur_gt[:, :, 5] / 2
                iou_targets = boxes_iou3d_gpu_pcdet(
                    preds_box.reshape(-1, 7), cur_gt.reshape(
                        -1, 7))[range(preds_box.reshape(-1, 7).shape[0]),
                                range(cur_gt.reshape(-1, 7).shape[0])]
                iou_targets[torch.isnan(iou_targets)] = 0
                iou_targets = 2 * iou_targets - 1

            isnotnan = (~torch.isnan(iou_targets)).float()
            mask_iou_loss = mask.reshape(-1) * isnotnan
            iou_loss = self.loss_iou(
                preds_dict['iou'].reshape(-1),
                iou_targets,
                mask_iou_loss,
                avg_factor=(gt_num + 1e-4))
            losses.update({f'task{task_id}.loss_iou': iou_loss})

        return losses

    def predict(self,
                pts_feats: Dict[str, torch.Tensor],
                batch_data_samples: List[Det3DDataSample],
                rescale=True,
                **kwargs) -> List[InstanceData]:
        """
        Args:
            pts_feats (dict): Point features..
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes meta information of data.
            rescale (bool): Whether rescale the resutls to
                the original scale.

        Returns:
            list[:obj:`InstanceData`]: List of processed predictions. Each
            InstanceData contains 3d Bounding boxes and corresponding
            scores and labels.
        """
        batch_size = len(batch_data_samples)
        batch_input_metas = []
        for batch_index in range(batch_size):
            metainfo = batch_data_samples[batch_index].metainfo
            batch_input_metas.append(metainfo)

        fpn_feat = pts_feats[-1]
        outs = dict()
        outs.update(self.forward_heatmap(fpn_feat))

        batch, num_cls, H, W = outs['hm'].size()
        scores, labels = torch.max(
            outs['hm'].reshape(batch, num_cls, H * W), dim=1)  # b, H*W

        order = scores.sort(1, descending=True)[1]
        order = order[:, :self.test_cfg['num_center_proposals']]

        scores = torch.gather(scores, 1, order)
        labels = torch.gather(labels, 1, order)

        batch_id_gt = torch.from_numpy(
            np.indices(
                (batch, self.test_cfg['num_center_proposals']))[0]).to(labels)
        query = fpn_feat.reshape(batch, -1, H * W).transpose(2, 1).contiguous()
        query = query[batch_id_gt, order]  # B, 500, C

        ct_feat = self.forward_decoder(query, pts_feats, order)

        # Only support one head now
        outs.update(self(ct_feat)[0])

        masks = scores > self.test_cfg['score_threshold']
        outs.update(
            dict(order=order, masks=masks, scores=scores, labels=labels))

        ret_list = self.predict_by_feat(outs, batch_input_metas)

        return ret_list

    def predict_by_feat(self, outs, batch_input_metas):
        rets = []
        post_center_range = self.test_cfg.post_center_limit_range
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=outs['scores'].dtype,
                device=outs['scores'].device,
            )

        # convert B C N to B N C
        for key, val in outs.items():
            if torch.is_tensor(outs[key]):
                if len(outs[key].shape) == 3:
                    outs[key] = val.permute(0, 2, 1).contiguous()

        batch_score = outs['scores']
        batch_label = outs['labels']
        batch_mask = outs['masks']
        batch_iou = outs['iou'].squeeze(2)

        batch_dim = torch.exp(outs['dim'])

        batch_rots = outs['rot'][..., 0:1]
        batch_rotc = outs['rot'][..., 1:2]

        batch_reg = outs['reg']
        batch_hei = outs['height']
        batch_rot = torch.atan2(batch_rots, batch_rotc)
        batch_iou = (batch_iou + 1) * 0.5
        batch_iou = torch.clamp(batch_iou, min=0.0, max=1.0)

        batch, _, H, W = outs['hm'].size()

        ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
        ys = ys.view(1, H, W).repeat(batch, 1, 1).to(batch_score)
        xs = xs.view(1, H, W).repeat(batch, 1, 1).to(batch_score)

        obj_num = outs['order'].shape[1]
        batch_id = np.indices((batch, obj_num))[0]
        batch_id = torch.from_numpy(batch_id).to(outs['order'])

        xs = (
            xs.view(batch, -1, 1)[batch_id, outs['order']] +
            batch_reg[:, :, 0:1])
        ys = (
            ys.view(batch, -1, 1)[batch_id, outs['order']] +
            batch_reg[:, :, 1:2])

        xs = (
            xs * self.test_cfg.out_size_factor * self.test_cfg.voxel_size[0] +
            self.test_cfg.point_cloud_range[0])
        ys = (
            ys * self.test_cfg.out_size_factor * self.test_cfg.voxel_size[1] +
            self.test_cfg.point_cloud_range[1])

        if 'vel' in outs:
            batch_vel = outs['vel']
            batch_box_preds = torch.cat(
                [xs, ys, batch_hei, batch_dim, batch_vel, batch_rot], dim=2)
        else:
            batch_box_preds = torch.cat(
                [xs, ys, batch_hei, batch_dim, batch_rot], dim=2)

        rets.append(
            self.post_processing(
                batch_input_metas,
                batch_box_preds,
                batch_score,
                batch_label,
                self.test_cfg,
                post_center_range,
                batch_mask,
                batch_iou,
            ))

        # Merge branches results
        ret_list = []
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            temp_instances = InstanceData()
            for k in rets[0][i].keys():
                if k == 'bboxes':
                    bboxes = torch.cat([ret[i][k] for ret in rets])
                    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
                    # The original CenterFormer model predict (..., w,l,h)
                    # Note that this is used to align the precision of
                    # converted model
                    # bboxes[:, 4], bboxes[:, 3] = bboxes[:, 3].clone(
                    # ), bboxes[:, 4].clone()
                    # bboxes[:, 6] = -bboxes[:, 6] - np.pi / 2
                    bboxes = batch_input_metas[i]['box_type_3d'](
                        bboxes, self.bbox_code_size)
                elif k == 'labels':
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    labels = torch.cat([ret[i][k] for ret in rets])
                elif k == 'scores':
                    scores = torch.cat([ret[i][k] for ret in rets])
            # ret['metadata'] = metas[0][i]

            temp_instances.bboxes_3d = bboxes
            temp_instances.scores_3d = scores
            temp_instances.labels_3d = labels
            ret_list.append(temp_instances)

        return ret_list

    @torch.no_grad()
    def post_processing(
        self,
        img_metas,
        batch_box_preds,
        batch_score,
        batch_label,
        test_cfg,
        post_center_range,
        batch_mask,
        batch_iou,
    ):
        batch_size = len(batch_score)

        prediction_dicts = []
        for i in range(batch_size):
            box_preds = batch_box_preds[i]
            scores = batch_score[i]
            labels = batch_label[i]
            mask = batch_mask[i]

            distance_mask = (box_preds[..., :3] >= post_center_range[:3]).all(
                1) & (box_preds[..., :3] <= post_center_range[3:]).all(1)
            mask = mask & distance_mask

            box_preds = box_preds[mask]
            scores = scores[mask]
            labels = labels[mask]

            iou_factor = torch.LongTensor(self.iou_factor).to(labels)
            ious = batch_iou[i][mask]
            ious = torch.pow(ious, iou_factor[labels])
            scores = scores * ious

            boxes_for_nms = box_preds[:, [0, 1, 2, 3, 4, 5, -1]]

            # multi class nms
            selected = []
            for c in range(3):
                class_mask = labels == c
                if class_mask.sum() > 0:
                    class_idx = class_mask.nonzero()
                    # boxes_for_nms = xywhr2xyxyr(
                    #     img_metas[i]['box_type_3d'](
                    #         box_preds[:, :], self.bbox_code_size).bev)
                    # select = nms_bev(
                    #     boxes_for_nms[class_mask].float(),
                    #     scores[class_mask].float(),
                    #     thresh=test_cfg.nms_iou_threshold[c],
                    #     pre_max_size=test_cfg.nms_pre_max_size[c],
                    #     post_max_size=test_cfg.nms_post_max_size[c],
                    # )
                    select = rotate_nms_pcdet(
                        boxes_for_nms[class_mask].float(),
                        scores[class_mask].float(),
                        thresh=test_cfg.nms_iou_thres[c],
                        pre_maxsize=test_cfg.nms_pre_max_size[c],
                        post_max_size=test_cfg.nms_post_max_size[c],
                    )
                    selected.append(class_idx[select, 0])
            if len(selected) > 0:
                selected = torch.cat(selected, dim=0)

            selected_boxes = box_preds[selected]
            selected_scores = scores[selected]
            selected_labels = labels[selected]

            prediction_dict = {
                'bboxes': selected_boxes,
                'scores': selected_scores,
                'labels': selected_labels,
            }

            prediction_dicts.append(prediction_dict)

        return prediction_dicts


def get_box(pred_boxs, order, test_cfg, H, W):
    batch = pred_boxs.shape[0]
    obj_num = order.shape[1]
    ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
    ys = ys.view(1, H, W).repeat(batch, 1, 1).to(pred_boxs)
    xs = xs.view(1, H, W).repeat(batch, 1, 1).to(pred_boxs)

    batch_id = np.indices((batch, obj_num))[0]
    batch_id = torch.from_numpy(batch_id).to(order)
    xs = xs.view(batch, H * W)[batch_id, order].unsqueeze(1) + pred_boxs[:,
                                                                         0:1]
    ys = ys.view(batch, H * W)[batch_id, order].unsqueeze(1) + pred_boxs[:,
                                                                         1:2]

    xs = xs * test_cfg.out_size_factor * test_cfg.voxel_size[
        0] + test_cfg.point_cloud_range[0]
    ys = ys * test_cfg.out_size_factor * test_cfg.voxel_size[
        1] + test_cfg.point_cloud_range[1]

    rot = torch.atan2(pred_boxs[:, 6:7], pred_boxs[:, 7:8])
    pred = torch.cat(
        [xs, ys, pred_boxs[:, 2:3],
         torch.exp(pred_boxs[:, 3:6]), rot], dim=1)
    pred[:, 2] = pred[:, 2] - pred[:, 5] / 2

    return torch.transpose(pred, 1, 2).contiguous()  # B M 7


def get_box_gt(gt_boxs, order, test_cfg, H, W):
    batch = gt_boxs.shape[0]
    obj_num = order.shape[1]
    ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
    ys = ys.view(1, H, W).repeat(batch, 1, 1).to(gt_boxs)
    xs = xs.view(1, H, W).repeat(batch, 1, 1).to(gt_boxs)

    batch_id = np.indices((batch, obj_num))[0]
    batch_id = torch.from_numpy(batch_id).to(order)

    batch_gt_dim = torch.exp(gt_boxs[..., 3:6])
    batch_gt_hei = gt_boxs[..., 2:3]
    batch_gt_rot = torch.atan2(gt_boxs[..., -2:-1], gt_boxs[..., -1:])
    xs = xs.view(batch, H * W)[batch_id, order].unsqueeze(2) + gt_boxs[...,
                                                                       0:1]
    ys = ys.view(batch, H * W)[batch_id, order].unsqueeze(2) + gt_boxs[...,
                                                                       1:2]

    xs = xs * test_cfg.out_size_factor * test_cfg.voxel_size[
        0] + test_cfg.point_cloud_range[0]
    ys = ys * test_cfg.out_size_factor * test_cfg.voxel_size[
        1] + test_cfg.point_cloud_range[1]

    batch_box_targets = torch.cat(
        [xs, ys, batch_gt_hei, batch_gt_dim, batch_gt_rot], dim=-1)

    batch_box_targets[...,
                      2] = batch_box_targets[...,
                                             2] - batch_box_targets[..., 5] / 2

    return batch_box_targets  # B M 7


def get_corresponding_box(x_ind, y_ind, y_mask, y_cls, target_box):
    # find the id in y which has the same ind in x
    select_target = torch.zeros(x_ind.shape[0], x_ind.shape[1],
                                target_box.shape[2]).to(target_box)
    select_mask = torch.zeros_like(x_ind).to(y_mask)
    select_cls = torch.zeros_like(x_ind).to(y_cls)

    for i in range(x_ind.shape[0]):
        idx = torch.arange(y_ind[i].shape[-1]).to(x_ind)
        idx = idx[y_mask[i]]
        box_cls = y_cls[i][y_mask[i]]
        valid_y_ind = y_ind[i][y_mask[i]]
        match = (x_ind[i].unsqueeze(1) == valid_y_ind.unsqueeze(0)).nonzero()
        select_target[i, match[:, 0]] = target_box[i, idx[match[:, 1]]]
        select_mask[i, match[:, 0]] = 1
        select_cls[i, match[:, 0]] = box_cls[match[:, 1]]

    return select_target, select_mask, select_cls
