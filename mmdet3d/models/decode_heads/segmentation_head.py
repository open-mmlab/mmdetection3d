# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn as nn

from .decode_head import Base3DDecodeHead
from mmengine.model import normal_init

from mmdet3d.models.layers.sst import build_mlp, scatter_v2
from torch.utils.checkpoint import checkpoint

from mmdet3d.registry import MODELS
from .. import LOSSES


@MODELS.register_module()
class VoteSegHead(Base3DDecodeHead):

    def __init__(self,
                 in_channel,
                 num_classes,
                 hidden_dims=[],
                 dropout_ratio=0.5,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='naiveSyncBN1d'),
                 act_cfg=dict(type='ReLU'),
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     class_weight=None,
                     loss_weight=1.0),
                 loss_vote=dict(
                     type='L1Loss',
                 ),
                 loss_aux=None,
                 ignore_index=255,
                 logit_scale=1,
                 checkpointing=False,
                 init_bias=None,
                 init_cfg=None):
        end_channel = hidden_dims[-1] if len(hidden_dims) > 0 else in_channel
        super(VoteSegHead, self).__init__(
                 end_channel,
                 num_classes,
                 dropout_ratio,
                 conv_cfg,
                 norm_cfg,
                 act_cfg,
                 loss_decode,
                 ignore_index,
                 init_cfg
        )

        self.pre_seg_conv = None
        if len(hidden_dims) > 0:
            self.pre_seg_conv = build_mlp(in_channel, hidden_dims, norm_cfg, act=act_cfg['type'])

        self.use_sigmoid = loss_decode.get('use_sigmoid', False)
        self.bg_label = self.num_classes
        if not self.use_sigmoid:
            self.num_classes += 1


        self.logit_scale = logit_scale
        self.conv_seg = nn.Linear(end_channel, self.num_classes)
        self.voting = nn.Linear(end_channel, self.num_classes * 3)
        self.fp16_enabled = False
        self.checkpointing = checkpointing
        self.init_bias = init_bias

        if loss_aux is not None:
            self.loss_aux = LOSSES.build(loss_aux)
        else:
            self.loss_aux = None
        if loss_decode['type'] == 'FocalLoss':
            self.loss_decode = LOSSES.build(loss_decode)  # mmdet has a better focal loss supporting single class
        
        self.loss_vote = LOSSES.build(loss_vote)

    def init_weights(self):
        """Initialize weights."""
        super().init_weights()
        if self.init_bias is not None:
            self.conv_seg.bias.data.fill_(self.init_bias)
            print(f'Segmentation Head bias is initialized to {self.init_bias}')
        else:
            normal_init(self.conv_seg, mean=0, std=0.01)

    # @auto_fp16(apply_to=('voxel_feat',))
    def forward(self, voxel_feat):
        """Forward pass.

        """

        output = voxel_feat
        if self.pre_seg_conv is not None:
            if self.checkpointing:
                output = checkpoint(self.pre_seg_conv, voxel_feat)
            else:
                output = self.pre_seg_conv(voxel_feat)
        logits = self.cls_seg(output)
        vote_preds = self.voting(output)

        return logits, vote_preds

    # @force_fp32(apply_to=('seg_logit', 'vote_preds'))
    def losses(self, seg_logit, vote_preds, seg_label, vote_targets, vote_mask):
        """Compute semantic segmentation loss.

        Args:
            seg_logit (torch.Tensor): Predicted per-point segmentation logits \
                of shape [B, num_classes, N].
            seg_label (torch.Tensor): Ground-truth segmentation label of \
                shape [B, N].
        """
        seg_logit = seg_logit * self.logit_scale
        loss = dict()
        loss['loss_sem_seg'] = self.loss_decode(seg_logit, seg_label)
        if self.loss_aux is not None:
            loss['loss_aux'] = self.loss_aux(seg_logit, seg_label)

        vote_preds = vote_preds.reshape(-1, self.num_classes, 3)
        if not self.use_sigmoid:
            assert seg_label.max().item() == self.num_classes - 1
        else:
            assert seg_label.max().item() == self.num_classes
        valid_vote_preds = vote_preds[vote_mask] # [n_valid, num_cls, 3]
        valid_vote_preds = valid_vote_preds.reshape(-1, 3)
        num_valid = vote_mask.sum()

        valid_label = seg_label[vote_mask]

        if num_valid > 0:
            assert valid_label.max().item() < self.num_classes
            assert valid_label.min().item() >= 0

            indices = torch.arange(num_valid, device=valid_label.device) * self.num_classes + valid_label
            valid_vote_preds = valid_vote_preds[indices, :] #[n_valid, 3]

            valid_vote_targets = vote_targets[vote_mask]

            loss['loss_vote'] = self.loss_vote(valid_vote_preds, valid_vote_targets)
        else:
            loss['loss_vote'] = vote_preds.sum() * 0

        train_cfg = self.train_cfg
        if train_cfg.get('score_thresh', None) is not None:
            score_thresh = train_cfg['score_thresh']
            if self.use_sigmoid:
                scores = seg_logit.sigmoid()
                for i in range(len(score_thresh)):
                    thr = score_thresh[i]
                    name = train_cfg['class_names'][i]
                    this_scores = scores[:, i]
                    pred_true = this_scores > thr
                    real_true = seg_label == i
                    tp = (pred_true & real_true).sum().float()
                    loss[f'recall_{name}'] = tp / (real_true.sum().float() + 1e-5)
            else:
                score = seg_logit.softmax(1)
                group_lens = train_cfg['group_lens']
                group_score = self.gather_group(score[:, :-1], group_lens)
                num_fg = score.new_zeros(1)
                for gi in range(len(group_lens)):
                    pred_true = group_score[:, gi] > score_thresh[gi]
                    num_fg += pred_true.sum().float()
                    for i in range(group_lens[gi]):
                        name = train_cfg['group_names'][gi][i]
                        real_true = seg_label == train_cfg['class_names'].index(name)
                        tp = (pred_true & real_true).sum().float()
                        loss[f'recall_{name}'] = tp / (real_true.sum().float() + 1e-5)
                loss[f'num_fg'] = num_fg

        return loss

    def forward_train(self, inputs, img_metas, pts_semantic_mask, vote_targets, vote_mask, return_preds=False):

        seg_logits, vote_preds = self.forward(inputs)
        losses = self.losses(seg_logits, vote_preds, pts_semantic_mask, vote_targets, vote_mask)
        if return_preds:
            return losses, dict(seg_logits=seg_logits, vote_preds=vote_preds)
        else:
            return losses

    def gather_group(self, scores, group_lens):
        assert (scores >= 0).all()
        score_per_group = []
        beg = 0
        for group_len in group_lens:
            end = beg + group_len
            score_this_g = scores[:, beg:end].sum(1)
            score_per_group.append(score_this_g)
            beg = end
        assert end == scores.size(1) == sum(group_lens)
        gathered_score = torch.stack(score_per_group, dim=1)
        assert gathered_score.size(1) == len(group_lens)
        return  gathered_score

    def get_targets(self, points_list, gt_bboxes_list, gt_labels_list):
        bsz = len(points_list)
        label_list = []
        vote_target_list = []
        vote_mask_list = []

        for i in range(bsz):

            points = points_list[i][:, :3]
            bboxes = gt_bboxes_list[i]
            bbox_labels = gt_labels_list[i]

            # if self.num_classes < 3: # I don't know why there are some -1 labels when train car-only model.
            valid_gt_mask = bbox_labels >= 0
            bboxes = bboxes[valid_gt_mask]
            bbox_labels = bbox_labels[valid_gt_mask]
            
            if len(bbox_labels) == 0:
                this_label = torch.ones(len(points), device=points.device, dtype=torch.long) * self.bg_label
                this_vote_target = torch.zeros_like(points)
                vote_mask = torch.zeros_like(this_label).bool()
            else:
                extra_width = self.train_cfg.get('extra_width', None) 
                if extra_width is not None:
                    bboxes = bboxes.enlarged_box_hw(extra_width)
                inbox_inds = bboxes.points_in_boxes(points).long()
                this_label = self.get_point_labels(inbox_inds, bbox_labels)
                this_vote_target, vote_mask = self.get_vote_target(inbox_inds, points, bboxes)

            label_list.append(this_label)
            vote_target_list.append(this_vote_target)
            vote_mask_list.append(vote_mask)

        labels = torch.cat(label_list, dim=0)
        vote_targets = torch.cat(vote_target_list, dim=0)
        vote_mask = torch.cat(vote_mask_list, dim=0)

        return labels, vote_targets, vote_mask
    

    def get_point_labels(self, inbox_inds, bbox_labels):

        bg_mask = inbox_inds < 0
        label = -1 * torch.ones(len(inbox_inds), dtype=torch.long, device=inbox_inds.device)
        class_labels = bbox_labels[inbox_inds]
        class_labels[bg_mask] = self.bg_label
        return class_labels

    def get_vote_target(self, inbox_inds, points, bboxes):

        bg_mask = inbox_inds < 0
        if self.train_cfg.get('centroid_offset', False):
            centroid, _, inv = scatter_v2(points, inbox_inds, mode='avg', return_inv=True)
            center_per_point = centroid[inv]
        else:
            center_per_point = bboxes.gravity_center[inbox_inds]
        delta = center_per_point.to(points.device) - points
        delta[bg_mask] = 0
        target = self.encode_vote_targets(delta)
        vote_mask = ~bg_mask
        return target, vote_mask
    
    def encode_vote_targets(self, delta):
        return torch.sign(delta) * (delta.abs() ** 0.5) 
    
    def decode_vote_targets(self, preds):
        return preds * preds.abs()
