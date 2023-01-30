# ------------------------------------------------------------------------------
# Portions of this code are from
# det3d (https://github.com/poodarchu/Det3D/tree/56402d4761a5b73acd23080f537599b0888cce07) # noqa
# Copyright (c) 2019 朱本金
# Licensed under the MIT License
# ------------------------------------------------------------------------------

import copy
import logging

import numpy as np
import torch
from mmcv.cnn import build_norm_layer
from mmcv.ops import boxes_iou3d
from mmengine.logging import print_log
from mmengine.model import kaiming_init
from mmengine.structures import InstanceData
from torch import nn

from mmdet3d.models.layers import circle_nms, nms_bev
from mmdet3d.registry import MODELS
from .bbox_ops import nms_iou3d
from .losses import FastFocalLoss


class SepHead(nn.Module):
    """TODO: This module is the original implementation in CenterFormer and it
    has few differences with ``SeperateHead`` in `mmdet3d` but refactor this
    module will lower the performance a little.
    """

    def __init__(
            self,
            in_channels,
            heads,
            head_conv=64,
            final_kernel=1,
            bn=False,
            init_bias=-2.19,
            norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
            **kwargs,
    ):
        super(SepHead, self).__init__(**kwargs)

        self.heads = heads
        for head in self.heads:
            classes, num_conv = self.heads[head]

            fc = []
            for i in range(num_conv - 1):
                fc.append(
                    nn.Conv1d(
                        in_channels,
                        head_conv,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        bias=True,
                    ))
                if bn:
                    fc.append(build_norm_layer(norm_cfg, head_conv)[1])
                fc.append(nn.ReLU())

            fc.append(
                nn.Conv1d(
                    head_conv,
                    classes,
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    bias=True,
                ))

            if 'hm' in head:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc:
                    if isinstance(m, nn.Conv1d):
                        kaiming_init(m)

            fc = nn.Sequential(*fc)
            self.__setattr__(head, fc)

    def forward(self, x, y):
        for head in self.heads:
            x[head] = self.__getattr__(head)(y)

        return x


@MODELS.register_module()
class CenterFormerBboxHead(nn.Module):

    def __init__(self,
                 in_channels,
                 tasks,
                 weight=0.25,
                 iou_weight=1,
                 corner_weight=1,
                 code_weights=[],
                 common_heads=dict(),
                 logger=None,
                 init_bias=-2.19,
                 share_conv_channel=64,
                 assign_label_window_size=1,
                 iou_loss=False,
                 corner_loss=False,
                 iou_factor=[1, 1, 4],
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 bbox_code_size=7,
                 test_cfg=None,
                 **kawrgs):
        super(CenterFormerBboxHead, self).__init__()

        num_classes = [len(t['class_names']) for t in tasks]
        self.class_names = [t['class_names'] for t in tasks]
        self.code_weights = code_weights
        self.bbox_code_size = 7
        self.weight = weight  # weight between hm loss and loc loss
        self.iou_weight = iou_weight
        self.corner_weight = corner_weight
        self.iou_factor = iou_factor

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.test_cfg = test_cfg

        self.crit = FastFocalLoss(assign_label_window_size)
        self.crit_reg = torch.nn.L1Loss(reduction='none')
        self.use_iou_loss = iou_loss
        if self.use_iou_loss:
            self.crit_iou = torch.nn.SmoothL1Loss(reduction='none')
        self.corner_loss = corner_loss
        if self.corner_loss:
            self.corner_crit = torch.nn.MSELoss(reduction='none')

        self.box_n_dim = 9 if 'vel' in common_heads else 7
        self.use_direction_classifier = False

        if not logger:
            logger = logging.getLogger('CenterFormerBboxHead')
        self.logger = logger

        logger.info(f'num_classes: {num_classes}')

        # a shared convolution
        self.shared_conv = nn.Sequential(
            nn.Conv1d(
                in_channels, share_conv_channel, kernel_size=1, bias=True),
            build_norm_layer(norm_cfg, share_conv_channel)[1],
            nn.ReLU(inplace=True),
        )

        self.tasks = nn.ModuleList()
        print_log(f'Use HM Bias: {init_bias}', 'current')

        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            self.tasks.append(
                SepHead(
                    share_conv_channel,
                    heads,
                    bn=True,
                    init_bias=init_bias,
                    final_kernel=1,
                    norm_cfg=norm_cfg))

        logger.info('Finish CenterHeadIoU Initialization')

    def forward(self, x, *kwargs):
        ret_dicts = []

        y = self.shared_conv(x['ct_feat'].float())

        for task in self.tasks:
            ret_dicts.append(task(x, y))

        return ret_dicts

    def _sigmoid(self, x):
        y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
        return y

    def loss(self, preds_dicts, example, **kwargs):
        losses = {}
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            hm_loss = self.crit(
                preds_dict['hm'],
                example['hm'][task_id],
                example['ind'][task_id],
                example['mask'][task_id],
                example['cat'][task_id],
            )

            target_box = example['anno_box'][task_id]

            if self.corner_loss:
                corner_loss = self.corner_crit(preds_dict['corner_hm'],
                                               example['corners'][task_id])
                corner_mask = (example['corners'][task_id] > 0).to(corner_loss)
                corner_loss = (corner_loss * corner_mask).sum() / (
                    corner_mask.sum() + 1e-4)
                losses.update({
                    f'{task_id}_corner_loss':
                    corner_loss * self.corner_weight
                })

            # reconstruct the anno_box from multiple reg heads
            if 'vel' in preds_dict:
                preds_dict['anno_box'] = torch.cat(
                    (
                        preds_dict['reg'],
                        preds_dict['height'],
                        preds_dict['dim'],
                        preds_dict['vel'],
                        preds_dict['rot'],
                    ),
                    dim=1,
                )
            else:
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

            # Regression loss for dimension, offset, height, rotation
            # get corresponding gt box # B, 500
            target_box, selected_mask, selected_cls = get_corresponding_box(
                preds_dict['order'],
                example['ind'][task_id],
                example['mask'][task_id],
                example['cat'][task_id],
                target_box,
            )
            mask = selected_mask.float().unsqueeze(2)

            weights = self.code_weights

            box_loss = self.crit_reg(
                preds_dict['anno_box'].transpose(1, 2) * mask,
                target_box * mask)
            box_loss = box_loss / (mask.sum() + 1e-4)
            box_loss = box_loss.transpose(2, 0).sum(dim=2).sum(dim=1)

            loc_loss = (box_loss * box_loss.new_tensor(weights)).sum()

            if self.use_iou_loss:
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

                    iou_targets = boxes_iou3d(
                        preds_box.reshape(-1, 7), cur_gt.reshape(
                            -1, 7))[range(preds_box.reshape(-1, 7).shape[0]),
                                    range(cur_gt.reshape(-1, 7).shape[0])]
                    iou_targets[torch.isnan(iou_targets)] = 0
                    iou_targets = 2 * iou_targets - 1
                iou_loss = self.crit_iou(preds_dict['iou'].reshape(-1),
                                         iou_targets) * mask.reshape(-1)
                iou_loss = iou_loss.sum() / (mask.sum() + 1e-4)

                losses.update(
                    {f'{task_id}_iou_loss': iou_loss * self.iou_weight})

            losses.update({
                f'{task_id}_hm_loss': hm_loss,
                f'{task_id}_loc_loss': loc_loss * self.weight
            })

        return losses

    def predict(self, preds_dicts, batch_input_metas, **kwargs):
        """decode, nms, then return the detection result.

        Additionally support double flip testing
        """
        rets = []

        post_center_range = self.test_cfg.post_center_limit_range
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=preds_dicts[0]['scores'].dtype,
                device=preds_dicts[0]['scores'].device,
            )

        for task_id, preds_dict in enumerate(preds_dicts):
            # convert B C N to B N C
            for key, val in preds_dict.items():
                if torch.is_tensor(preds_dict[key]):
                    if len(preds_dict[key].shape) == 3:
                        preds_dict[key] = val.permute(0, 2, 1).contiguous()

            batch_score = preds_dict['scores']
            batch_label = preds_dict['labels']
            batch_mask = preds_dict['mask']
            if self.use_iou_loss:
                batch_iou = preds_dict['iou'].squeeze(2)
            else:
                batch_iou = None

            batch_dim = torch.exp(preds_dict['dim'])

            batch_rots = preds_dict['rot'][..., 0:1]
            batch_rotc = preds_dict['rot'][..., 1:2]

            batch_reg = preds_dict['reg']
            batch_hei = preds_dict['height']
            batch_rot = torch.atan2(batch_rots, batch_rotc)
            if self.use_iou_loss:
                batch_iou = (batch_iou + 1) * 0.5
                batch_iou = torch.clamp(batch_iou, min=0.0, max=1.0)

            batch, _, H, W = preds_dict['hm'].size()

            ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
            ys = ys.view(1, H, W).repeat(batch, 1, 1).to(batch_score)
            xs = xs.view(1, H, W).repeat(batch, 1, 1).to(batch_score)

            obj_num = preds_dict['order'].shape[1]
            batch_id = np.indices((batch, obj_num))[0]
            batch_id = torch.from_numpy(batch_id).to(preds_dict['order'])

            xs = (
                xs.view(batch, -1, 1)[batch_id, preds_dict['order']] +
                batch_reg[:, :, 0:1])
            ys = (
                ys.view(batch, -1, 1)[batch_id, preds_dict['order']] +
                batch_reg[:, :, 1:2])

            xs = (
                xs * self.test_cfg.out_size_factor *
                self.test_cfg.voxel_size[0] + self.test_cfg.pc_range[0])
            ys = (
                ys * self.test_cfg.out_size_factor *
                self.test_cfg.voxel_size[1] + self.test_cfg.pc_range[1])

            if 'vel' in preds_dict:
                batch_vel = preds_dict['vel']
                batch_box_preds = torch.cat(
                    [xs, ys, batch_hei, batch_dim, batch_vel, batch_rot],
                    dim=2)
            else:
                batch_box_preds = torch.cat(
                    [xs, ys, batch_hei, batch_dim, batch_rot], dim=2)

            if self.test_cfg.get('per_class_nms', False):
                pass
            else:
                rets.append(
                    self.post_processing(
                        batch_input_metas,
                        batch_box_preds,
                        batch_score,
                        batch_label,
                        self.test_cfg,
                        post_center_range,
                        task_id,
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

            temp_instances.bboxes_3d = bboxes
            temp_instances.scores_3d = scores
            temp_instances.labels_3d = labels
            ret_list.append(temp_instances)

        return ret_list

    def post_processing(
        self,
        img_metas,
        batch_box_preds,
        batch_score,
        batch_label,
        test_cfg,
        post_center_range,
        task_id,
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

            if self.use_iou_loss:
                iou_factor = torch.LongTensor(self.iou_factor).to(labels)
                ious = batch_iou[i][mask]
                ious = torch.pow(ious, iou_factor[labels])
                scores = scores * ious

            boxes_for_nms = box_preds[:, [0, 1, 2, 3, 4, 5, -1]]

            if test_cfg.get('circular_nms', False):
                centers = boxes_for_nms[:, [0, 1]]
                boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                selected = _circle_nms(
                    boxes,
                    min_radius=test_cfg.min_radius[task_id],
                    post_max_size=test_cfg.nms.nms_post_max_size,
                )
            elif test_cfg.nms.get('use_multi_class_nms', False):
                # multi class nms
                selected = []
                for c in range(3):
                    class_mask = labels == c
                    if class_mask.sum() > 0:
                        class_idx = class_mask.nonzero()
                        select = nms_iou3d(
                            boxes_for_nms[class_mask].float(),
                            scores[class_mask].float(),
                            thresh=test_cfg.nms.nms_iou_threshold[c],
                            pre_maxsize=test_cfg.nms.nms_pre_max_size[c],
                            post_max_size=test_cfg.nms.nms_post_max_size[c],
                        )
                        selected.append(class_idx[select, 0])
                if len(selected) > 0:
                    selected = torch.cat(selected, dim=0)
            else:
                selected = nms_bev(
                    boxes_for_nms.float(),
                    scores.float(),
                    thresh=test_cfg.nms.nms_iou_threshold,
                    pre_max_size=test_cfg.nms.nms_pre_max_size,
                    post_max_size=test_cfg.nms.nms_post_max_size,
                )

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


def _circle_nms(boxes, min_radius, post_max_size=83):
    """NMS according to center distance."""
    keep = np.array(circle_nms(boxes.cpu().numpy(),
                               thresh=min_radius))[:post_max_size]

    keep = torch.from_numpy(keep).long().to(boxes.device)

    return keep


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
        0] + test_cfg.pc_range[0]
    ys = ys * test_cfg.out_size_factor * test_cfg.voxel_size[
        1] + test_cfg.pc_range[1]

    rot = torch.atan2(pred_boxs[:, 6:7], pred_boxs[:, 7:8])
    pred = torch.cat(
        [xs, ys, pred_boxs[:, 2:3],
         torch.exp(pred_boxs[:, 3:6]), rot], dim=1)

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
        0] + test_cfg.pc_range[0]
    ys = ys * test_cfg.out_size_factor * test_cfg.voxel_size[
        1] + test_cfg.pc_range[1]

    batch_box_targets = torch.cat(
        [xs, ys, batch_gt_hei, batch_gt_dim, batch_gt_rot], dim=-1)

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
