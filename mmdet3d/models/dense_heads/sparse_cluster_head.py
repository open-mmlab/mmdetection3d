# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from mmdet.models.task_modules import AssignResult
from mmdet.models.utils import multi_apply
from mmdet.utils import reduce_mean
from mmengine.model import BaseModule

from mmdet3d.models.builder import build_loss
from mmdet3d.models.layers import box3d_multiclass_nms
from mmdet3d.models.layers.sst import build_mlp
from mmdet3d.models.task_modules import PseudoSampler
from mmdet3d.models.task_modules.builder import build_bbox_coder
from mmdet3d.registry import MODELS
from mmdet3d.structures import xywhr2xyxyr, LiDARInstance3DBoxes, bbox_overlaps_3d


@MODELS.register_module()
class SparseClusterHead(BaseModule):

    def __init__(self,
                 num_classes,
                 bbox_coder,
                 loss_cls,
                 loss_center,
                 loss_size,
                 loss_rot,
                 in_channel,
                 shared_mlp_dims,
                 shared_dropout=0,
                 cls_mlp=None,
                 reg_mlp=None,
                 iou_mlp=None,
                 train_cfg=None,
                 test_cfg=None,
                 norm_cfg=dict(type='LN'),
                 loss_iou=None,
                 act='relu',
                 corner_loss_cfg=None,
                 enlarge_width=None,
                 as_rpn=False,
                 init_cfg=None):
        super(SparseClusterHead, self).__init__(init_cfg=init_cfg)

        self.print_info = {}
        self.loss_center = build_loss(loss_center)
        self.loss_size = build_loss(loss_size)
        self.loss_rot = build_loss(loss_rot)
        self.loss_cls = build_loss(loss_cls)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.box_code_size = self.bbox_coder.code_size
        self.corner_loss_cfg = corner_loss_cfg
        self.num_classes = num_classes
        self.enlarge_width = enlarge_width
        self.sampler = PseudoSampler()
        self.sync_reg_avg_factor = False if train_cfg is None else train_cfg.get('sync_reg_avg_factor', True)
        self.sync_cls_avg_factor = False if train_cfg is None else train_cfg.get('sync_cls_avg_factor', False)
        self.as_rpn = as_rpn
        if train_cfg is not None:
            self.cfg = self.train_cfg = train_cfg
        if test_cfg is not None:
            self.cfg = self.test_cfg = test_cfg
        
        self.num_anchors = num_anchors = 1 # deprecated due to removing assign twice


        if loss_iou is not None:
            self.loss_iou = build_loss(loss_iou)
            # self.loss_iou = nn.binary_cross_entropy_with_logits
        else:
            self.loss_iou = None

        self.fp16_enabled = False

        # Bbox classification and regression
        self.shared_mlp = None
        if len(shared_mlp_dims) > 0:
            self.shared_mlp = build_mlp(in_channel, shared_mlp_dims, norm_cfg, act=act, dropout=shared_dropout)
        

        end_channel = shared_mlp_dims[-1] if len(shared_mlp_dims) > 0 else in_channel

        if cls_mlp is not None:
            self.conv_cls = build_mlp(end_channel, cls_mlp + [num_classes * num_anchors,], norm_cfg, True, act=act)
        else:
            self.conv_cls = nn.Linear(end_channel, num_classes * num_anchors)

        if reg_mlp is not None:
            self.conv_reg = build_mlp(end_channel, reg_mlp + [self.box_code_size * num_anchors,], norm_cfg, True, act=act)
        else:
            self.conv_reg = nn.Linear(end_channel, self.box_code_size * num_anchors)

        if loss_iou is not None:
            if iou_mlp is not None:
                self.conv_iou = build_mlp(end_channel, iou_mlp + [1,], norm_cfg, True, act=act)
            else:
                self.conv_iou = nn.Linear(end_channel, 1)
        
        self.save_list = []


    def forward(self, feats, pts_xyz=None, pts_inds=None):

        if self.shared_mlp is not None:
            feats = self.shared_mlp(feats)

        cls_logits = self.conv_cls(feats)
        reg_preds = self.conv_reg(feats)
        outs = dict(
            cls_logits=cls_logits,
            reg_preds=reg_preds,
        )
        if self.loss_iou is not None:
            outs['iou_logits'] = self.conv_iou(feats)

        return outs

    # @force_fp32(apply_to=('cls_logits', 'reg_preds', 'cluster_xyz'))
    def loss(self,
             cls_logits,
             reg_preds,
             cluster_xyz,
             cluster_inds,
             gt_bboxes_3d,
             gt_labels_3d,
             img_metas=None,
             iou_logits=None,
             gt_bboxes_ignore=None,
             ):
        
        if iou_logits is not None and iou_logits.dtype == torch.float16:
            iou_logits = iou_logits.to(torch.float)

        cluster_batch_idx = cluster_inds[:, 1]
        num_total_samples = len(reg_preds)

        targets = self.get_targets(cluster_xyz, cluster_batch_idx, gt_bboxes_3d, gt_labels_3d, reg_preds)
        labels, label_weights, bbox_targets, bbox_weights, iou_labels = targets
        assert (label_weights == 1).all(), 'for now'

        cls_avg_factor = num_total_samples * 1.0
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                bbox_weights.new_tensor([cls_avg_factor]))

        loss_cls = self.loss_cls(
            cls_logits, labels, label_weights, avg_factor=cls_avg_factor)

        # regression loss
        pos_inds = ((labels >= 0)& (labels < self.num_classes)).nonzero(as_tuple=False).reshape(-1)
        num_pos = len(pos_inds)
        assert num_pos == bbox_weights.sum() / self.box_code_size

        pos_reg_preds = reg_preds[pos_inds]
        pos_bbox_targets = bbox_targets[pos_inds]
        pos_bbox_weights = bbox_weights[pos_inds]
        assert (pos_bbox_weights > 0).all()

        reg_avg_factor = num_pos * 1.0
        if self.sync_reg_avg_factor:
            reg_avg_factor = reduce_mean(
                bbox_weights.new_tensor([reg_avg_factor]))

        if num_pos > 0:
            code_weight = self.train_cfg.get('code_weight', None)
            if code_weight:
                pos_bbox_weights = pos_bbox_weights * bbox_weights.new_tensor(
                    code_weight)[None, :]


            loss_center = self.loss_center(
                pos_reg_preds[:, :3],
                pos_bbox_targets[:, :3],
                pos_bbox_weights[:, :3],
                avg_factor=reg_avg_factor)
            loss_size = self.loss_size(
                pos_reg_preds[:, 3:6],
                pos_bbox_targets[:, 3:6],
                pos_bbox_weights[:, 3:6],
                avg_factor=reg_avg_factor)
            loss_rot = self.loss_rot(
                pos_reg_preds[:, 6:8],
                pos_bbox_targets[:, 6:8],
                pos_bbox_weights[:, 6:8],
                avg_factor=reg_avg_factor)
        else:
            loss_center = pos_reg_preds.sum() * 0
            loss_size = pos_reg_preds.sum() * 0
            loss_rot = pos_reg_preds.sum() * 0
        
        losses = dict(
            loss_cls=loss_cls,
            loss_center=loss_center,
            loss_size=loss_size,
            loss_rot=loss_rot,
        )

        if self.corner_loss_cfg is not None:
            losses['loss_corner'] = self.get_corner_loss(pos_reg_preds, pos_bbox_targets, cluster_xyz[pos_inds], reg_avg_factor)

        if self.loss_iou is not None:
            losses['loss_iou'] = self.loss_iou(iou_logits.reshape(-1), iou_labels, label_weights, avg_factor=cls_avg_factor)
            losses['max_iou'] = iou_labels.max()
            losses['mean_iou'] = iou_labels[iou_labels > 0].mean()

        return losses
    
    def get_corner_loss(self, reg_preds, bbox_targets, base_points, reg_avg_factor):
        if len(base_points) == 0:
            return base_points.new_zeros(1).sum()
        dets = self.bbox_coder.decode(reg_preds, base_points, self.corner_loss_cfg.get('detach_yaw', True))
        gts = self.bbox_coder.decode(bbox_targets, base_points)
        corner_loss = self.corner_loss(dets, gts, self.corner_loss_cfg['delta']).sum() 
        corner_loss = corner_loss.sum() / reg_avg_factor * self.corner_loss_cfg['loss_weight'] 
        return corner_loss

    def corner_loss(self, pred_bbox3d, gt_bbox3d, delta=1):
        """Calculate corner loss of given boxes.

        Args:
            pred_bbox3d (torch.FloatTensor): Predicted boxes in shape (N, 7).
            gt_bbox3d (torch.FloatTensor): Ground truth boxes in shape (N, 7).

        Returns:
            torch.FloatTensor: Calculated corner loss in shape (N).
        """
        assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

        gt_boxes_structure = LiDARInstance3DBoxes(gt_bbox3d)
        pred_box_corners = LiDARInstance3DBoxes(pred_bbox3d).corners
        gt_box_corners = gt_boxes_structure.corners

        # This flip only changes the heading direction of GT boxes
        gt_bbox3d_flip = gt_boxes_structure.clone()
        gt_bbox3d_flip.tensor[:, 6] += np.pi
        gt_box_corners_flip = gt_bbox3d_flip.corners

        corner_dist = torch.min(
            torch.norm(pred_box_corners - gt_box_corners, dim=2),
            torch.norm(pred_box_corners - gt_box_corners_flip,
                       dim=2))  # (N, 8)
        # huber loss
        abs_error = torch.abs(corner_dist)
        quadratic = torch.clamp(abs_error, max=delta)
        linear = (abs_error - quadratic)
        corner_loss = 0.5 * quadratic**2 + delta * linear

        return corner_loss.mean(1)


    def get_targets(self,
                    cluster_xyz,
                    batch_idx,
                    gt_bboxes_3d,
                    gt_labels_3d,
                    reg_preds=None):
        batch_size = len(gt_bboxes_3d)
        cluster_xyz_list = self.split_by_batch(cluster_xyz, batch_idx, batch_size)

        if reg_preds is not None:
            reg_preds_list = self.split_by_batch(reg_preds, batch_idx, batch_size)
        else:
            reg_preds_list = [None,] * len(cluster_xyz_list)

        target_list_per_sample = multi_apply(self.get_targets_single, cluster_xyz_list, gt_bboxes_3d, gt_labels_3d, reg_preds_list)
        targets = [self.combine_by_batch(t, batch_idx, batch_size) for t in target_list_per_sample]
        # targets == [labels, label_weights, bbox_targets, bbox_weights]
        return targets

    def split_by_batch(self, data, batch_idx, batch_size):
        if self.training:
            assert batch_idx.max().item() + 1 <= batch_size
        if batch_size == 1:
            return [data, ]
        data_list = []
        for i in range(batch_size):
            sample_mask = batch_idx == i
            data_list.append(data[sample_mask])
        return data_list

    def combine_by_batch(self, data_list, batch_idx, batch_size):
        assert len(data_list) == batch_size
        if data_list[0] is None:
            return None
        data_shape = (len(batch_idx),) + data_list[0].shape[1:]
        full_data = data_list[0].new_zeros(data_shape)
        for i, data in enumerate(data_list):
            sample_mask = batch_idx == i
            full_data[sample_mask] = data
        return full_data
        

    def get_targets_single(self,
                           cluster_xyz,
                           gt_bboxes_3d,
                           gt_labels_3d,
                           reg_preds=None):
        """Generate targets of vote head for single batch.

        """
        valid_gt_mask = gt_labels_3d >= 0
        gt_bboxes_3d = gt_bboxes_3d[valid_gt_mask]
        gt_labels_3d = gt_labels_3d[valid_gt_mask]

        gt_bboxes_3d = gt_bboxes_3d.to(cluster_xyz.device)
        if self.train_cfg.get('assign_by_dist', False):
            assign_result = self.assign_by_dist_single(cluster_xyz, gt_bboxes_3d, gt_labels_3d)
        else:
            assign_result = self.assign_single(cluster_xyz, gt_bboxes_3d, gt_labels_3d)
        
        # Do not put this before assign

        sample_result = self.sampler.sample(assign_result, cluster_xyz, gt_bboxes_3d.tensor) # Pseudo Sampler, use cluster_xyz as pseudo bbox here.

        pos_inds = sample_result.pos_inds
        neg_inds = sample_result.neg_inds

        # label targets
        num_cluster = len(cluster_xyz)
        labels = gt_labels_3d.new_full((num_cluster, ), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels_3d[sample_result.pos_assigned_gt_inds]
        assert (labels >= 0).all()
        label_weights = cluster_xyz.new_ones(num_cluster)

        # bbox targets
        bbox_targets = cluster_xyz.new_zeros((num_cluster, self.box_code_size))

        bbox_weights = cluster_xyz.new_zeros((num_cluster, self.box_code_size))
        bbox_weights[pos_inds] = 1.0

        bbox_targets[pos_inds] = self.bbox_coder.encode(sample_result.pos_gt_bboxes, cluster_xyz[pos_inds])

        if self.loss_iou is not None:
            iou_labels = self.get_iou_labels(reg_preds, cluster_xyz, gt_bboxes_3d.tensor, pos_inds)
        else:
            iou_labels = None

        return labels, label_weights, bbox_targets, bbox_weights, iou_labels
    
    def get_iou_labels(self, reg_preds, cluster_xyz, gt_bboxes_3d, pos_inds):
        assert reg_preds is not None
        num_pos = len(pos_inds)
        num_preds = len(reg_preds)
        if num_pos == 0:
            return cluster_xyz.new_zeros(num_preds)
        bbox_preds = self.bbox_coder.decode(reg_preds, cluster_xyz)
        ious = bbox_overlaps_3d(bbox_preds, gt_bboxes_3d, mode='iou', coordinate='lidar') #[num_preds, num_gts]
        ious = ious.max(1)[0]
        if not ((ious >= 0) & (ious <= 1)).all():
            print(f'*************** Got illegal iou:{ious.min()} or {ious.max()}')
            ious = torch.clamp(ious, min=0, max=1)

        iou_bg_thresh = self.train_cfg.iou_bg_thresh
        iou_fg_thresh = self.train_cfg.iou_fg_thresh
        fg_mask = ious > iou_fg_thresh
        bg_mask = ious < iou_bg_thresh
        interval_mask = (fg_mask == 0) & (bg_mask == 0)

        iou_labels = (fg_mask > 0).float()
        iou_labels[interval_mask] = \
            (ious[interval_mask] - iou_bg_thresh) / (iou_fg_thresh - iou_bg_thresh)
        return iou_labels


    def assign_single(self,
                      cluster_xyz,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      ):
        """Generate targets of vote head for single batch.

        """

        num_cluster = cluster_xyz.size(0)
        num_gts = gt_bboxes_3d.tensor.size(0)

        # initialize as all background
        assigned_gt_inds = cluster_xyz.new_zeros((num_cluster, ), dtype=torch.long) # 0 indicates assign to backgroud
        assigned_labels = cluster_xyz.new_full((num_cluster, ), -1, dtype=torch.long)

        if num_gts == 0 or num_cluster == 0:
            # No ground truth or cluster, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        enlarged_box = self.enlarge_gt_bboxes(gt_bboxes_3d)
        inbox_inds = enlarged_box.points_in_boxes(cluster_xyz).long()
        inbox_inds = self.dist_constrain(inbox_inds, cluster_xyz, gt_bboxes_3d, gt_labels_3d)
        pos_cluster_mask = inbox_inds > -1

        if pos_cluster_mask.any():
            assigned_gt_inds[pos_cluster_mask] = inbox_inds[pos_cluster_mask] + 1
            assigned_labels[pos_cluster_mask] = gt_labels_3d[inbox_inds[pos_cluster_mask]]

        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)

    def assign_by_dist_single(self,
                      cluster_xyz,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      ):
        """Generate targets of vote head for single batch.

        """

        num_cluster = cluster_xyz.size(0)
        num_gts = gt_bboxes_3d.tensor.size(0)

        # initialize as all background
        assigned_gt_inds = cluster_xyz.new_zeros((num_cluster, ), dtype=torch.long) # 0 indicates assign to backgroud
        assigned_labels = cluster_xyz.new_full((num_cluster, ), -1, dtype=torch.long)

        if num_gts == 0 or num_cluster == 0:
            # No ground truth or cluster, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        gt_centers = gt_bboxes_3d.gravity_center[None, :, :2]
        pd_xy = cluster_xyz[None, :, :2]
        dist_mat = torch.cdist(pd_xy, gt_centers).squeeze(0)
        max_dist = self.train_cfg['max_dist']
        min_dist_v, matched_gt_inds = torch.min(dist_mat, dim=1)

        dist_mat[list(range(num_cluster//2)), matched_gt_inds] = 1e6

        matched_gt_inds[min_dist_v >= max_dist] = -1
        pos_cluster_mask = matched_gt_inds > -1

        # log 
        num_matched_gt = len(torch.unique(matched_gt_inds)) - 1
        num_matched_gt = torch.tensor(num_matched_gt, dtype=torch.float, device=cluster_xyz.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_matched_gt)

        num_gts_t = torch.tensor(num_gts, dtype=torch.float, device=cluster_xyz.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_gts_t)
        self.print_info['assign_recall'] = num_matched_gt / (num_gts_t + 1 + 1e-5)
        # end log


        if pos_cluster_mask.any():
            assigned_gt_inds[pos_cluster_mask] = matched_gt_inds[pos_cluster_mask] + 1
            assigned_labels[pos_cluster_mask] = gt_labels_3d[matched_gt_inds[pos_cluster_mask]]

        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # generate votes target
    def enlarge_gt_bboxes(self, gt_bboxes_3d, gt_labels_3d=None):
        if self.enlarge_width is not None:
            return gt_bboxes_3d.enlarged_box(self.enlarge_width)
        else:
            return gt_bboxes_3d
    
    def dist_constrain(self, inbox_inds, cluster_xyz, gt_bboxes_3d, gt_labels_3d):

        inbox_inds = inbox_inds.clone()
        max_dist = self.train_cfg.get('max_assign_dist', None)
        if max_dist is None:
            return inbox_inds

        if not (inbox_inds > -1).any():
            return inbox_inds

        pos_mask = inbox_inds > -1
        pos_inds = inbox_inds[pos_mask].clone()
        pos_xyz = cluster_xyz[pos_mask]
        pos_labels = gt_labels_3d[pos_inds]
        pos_box_center = gt_bboxes_3d.gravity_center[pos_inds]
        rel_dist = torch.linalg.norm(pos_xyz[:, :2] - pos_box_center[:, :2], ord=2, dim=1) # only xy-dist
        thresh = torch.zeros_like(rel_dist)
        assert len(max_dist) == self.num_classes
        for i in range(self.num_classes):
            thresh[pos_labels == i] = max_dist[i]
        
        pos_inds[rel_dist > thresh] = -1
        inbox_inds[pos_mask] = pos_inds
        return inbox_inds
        

    @torch.no_grad()
    def get_bboxes(self,
                   cls_logits,
                   reg_preds,
                   cluster_xyz,
                   cluster_inds,
                   input_metas,
                   iou_logits=None,
                   rescale=False,
                   ):


        batch_inds = cluster_inds[:, 1]
        batch_size = len(input_metas)
        cls_logits_list = self.split_by_batch(cls_logits, batch_inds, batch_size)
        reg_preds_list = self.split_by_batch(reg_preds, batch_inds, batch_size)
        cluster_xyz_list = self.split_by_batch(cluster_xyz, batch_inds, batch_size)

        if iou_logits is not None:
            iou_logits_list = self.split_by_batch(iou_logits, batch_inds, batch_size)
        else:
            iou_logits_list = [None,] * len(cls_logits_list)

        multi_results = multi_apply(
            self._get_bboxes_single,
            cls_logits_list,
            iou_logits_list,
            reg_preds_list,
            cluster_xyz_list,
            input_metas
        )
        # out_bboxes_list, out_scores_list, out_labels_list = multi_results
        results_list = [(b, s, l) for b, s, l in zip(*multi_results)]
        return results_list

    
    def _get_bboxes_single(
            self,
            cls_logits,
            iou_logits,
            reg_preds,
            cluster_xyz,
            input_meta,
        ):
        '''
        Get bboxes of a sample
        '''

        if self.as_rpn:
            cfg = self.train_cfg.rpn if self.training else self.test_cfg.rpn
        else:
            cfg = self.test_cfg

        assert cls_logits.size(0) == reg_preds.size(0) == cluster_xyz.size(0)
        assert cls_logits.size(1) == self.num_classes
        assert reg_preds.size(1) == self.box_code_size

        scores = cls_logits.sigmoid()

        if iou_logits is not None:
            iou_scores = iou_logits.sigmoid()
            a = cfg.get('iou_score_weight', 0.5)
            scores = (scores ** (1 - a)) * (iou_scores ** a)

        nms_pre = cfg.get('nms_pre', -1)
        if nms_pre > 0 and scores.shape[0] > nms_pre:
            max_scores, _ = scores.max(dim=1)
            _, topk_inds = max_scores.topk(nms_pre)
            reg_preds = reg_preds[topk_inds, :]
            scores = scores[topk_inds, :]
            cluster_xyz = cluster_xyz[topk_inds, :]

        bboxes = self.bbox_coder.decode(reg_preds, cluster_xyz)
        bboxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d'](bboxes).bev)

        # Add a dummy background class to the front when using sigmoid
        padding = scores.new_zeros(scores.shape[0], 1)
        scores = torch.cat([scores, padding], dim=1)

        score_thr = cfg.get('score_thr', 0)
        results = box3d_multiclass_nms(bboxes, bboxes_for_nms,
                                    scores, score_thr, cfg.max_num,
                                    cfg)

        out_bboxes, out_scores, out_labels = results

        out_bboxes = input_meta['box_type_3d'](out_bboxes)

        return (out_bboxes, out_scores, out_labels)
