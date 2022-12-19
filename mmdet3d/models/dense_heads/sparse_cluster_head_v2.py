# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmdet.models.utils import multi_apply
from mmdet.utils import reduce_mean
from mmdet3d.structures import xywhr2xyxyr, LiDARInstance3DBoxes
from mmdet3d.models.layers.sst import build_mlp
from mmdet3d.models.layers import box3d_multiclass_nms

from mmdet3d.models import builder
from mmdet3d.models.builder import build_loss
from mmdet3d.registry import MODELS
from .sparse_cluster_head import SparseClusterHead


@MODELS.register_module()
class FSDSeparateHead(BaseModule):

    def __init__(
        self,
        in_channels,
        attrs,
        norm_cfg=dict(type='LN'),
        act='relu',
        init_cfg=None,
        ):
        super().__init__(init_cfg=init_cfg)
        self.attrs = attrs
        for attr_name in self.attrs:
            out_dim, num_layer, hidden_dim = self.attrs[attr_name]
            mlp = build_mlp(in_channels, [hidden_dim,] * num_layer + [out_dim,], norm_cfg, is_head=True, act=act)
            self.__setattr__(attr_name, mlp)


    def forward(self, x):
        ret_dict = dict()
        for attr_name in self.attrs:
            ret_dict[attr_name] = self.__getattr__(attr_name)(x)

        return ret_dict


@MODELS.register_module()
class SparseClusterHeadV2(SparseClusterHead):

    def __init__(self,
                 num_classes,
                 bbox_coder,
                 loss_cls,
                 loss_center,
                 loss_size,
                 loss_rot,
                 in_channel,
                 shared_mlp_dims,
                 tasks,
                 class_names,
                 common_attrs,
                 num_cls_layer,
                 cls_hidden_dim,
                 separate_head,
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
                 init_cfg=None,
                 shared_dropout=0,
                 loss_vel=None,
                 ):
        super().__init__(
            num_classes,
            bbox_coder,
            loss_cls,
            loss_center,
            loss_size,
            loss_rot,
            in_channel,
            shared_mlp_dims,
            shared_dropout,
            cls_mlp,
            reg_mlp,
            iou_mlp,
            train_cfg,
            test_cfg,
            norm_cfg,
            loss_iou,
            act,
            corner_loss_cfg,
            enlarge_width,
            as_rpn,
            init_cfg
        )

        # override
        self.conv_cls = None
        self.conv_reg = None

        if self.shared_mlp is not None:
            sep_head_in_channels = shared_mlp_dims[-1]
        else:
            sep_head_in_channels = in_channel
        self.tasks = tasks
        self.task_heads = nn.ModuleList()

        for t in tasks:
            num_cls = len(t['class_names'])
            attrs = copy.deepcopy(common_attrs)
            attrs.update(dict(score=(num_cls, num_cls_layer, cls_hidden_dim), ))
            separate_head.update(
                in_channels=sep_head_in_channels, attrs=attrs)
            self.task_heads.append(builder.build_head(separate_head))

        self.class_names = class_names
        all_names = []
        for t in tasks:
            all_names += t['class_names']

        assert all_names == class_names

        if loss_vel is not None:
            self.loss_vel = build_loss(loss_vel)
        else:
            self.loss_vel = None
        


    def forward(self, feats, pts_xyz=None, pts_inds=None):

        if self.shared_mlp is not None:
            feats = self.shared_mlp(feats)

        cls_logit_list = []
        reg_pred_list = []
        for h in self.task_heads:
            ret_dict = h(feats)

            # keep consistent with v1, combine the regression prediction
            cls_logit = ret_dict['score']
            if 'vel' in ret_dict:
                reg_pred = torch.cat([ret_dict['center'], ret_dict['dim'], ret_dict['rot'], ret_dict['vel']], dim=-1)
            else:
                reg_pred = torch.cat([ret_dict['center'], ret_dict['dim'], ret_dict['rot']], dim=-1)
            cls_logit_list.append(cls_logit)
            reg_pred_list.append(reg_pred)

        outs = dict(
            cls_logits=cls_logit_list,
            reg_preds=reg_pred_list,
        )

        return outs

    # @force_fp32(apply_to=('cls_logits', 'reg_preds', 'cluster_xyz'))
    def loss(
        self,
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
        assert isinstance(cls_logits, list)
        assert isinstance(reg_preds, list)
        assert len(cls_logits) == len(reg_preds) == len(self.tasks)
        all_task_losses = {}
        for i in range(len(self.tasks)):
            losses_this_task = self.loss_single_task(
                i,
                cls_logits[i],
                reg_preds[i],
                cluster_xyz,
                cluster_inds,
                gt_bboxes_3d,
                gt_labels_3d,
                iou_logits,
            )
            all_task_losses.update(losses_this_task)
        return all_task_losses


    def loss_single_task(
            self,
            task_id,
            cls_logits,
            reg_preds,
            cluster_xyz,
            cluster_inds,
            gt_bboxes_3d,
            gt_labels_3d,
            iou_logits=None,
        ):


        gt_bboxes_3d, gt_labels_3d = self.modify_gt_for_single_task(gt_bboxes_3d, gt_labels_3d, task_id)
        
        if iou_logits is not None and iou_logits.dtype == torch.float16:
            iou_logits = iou_logits.to(torch.float)

        if cluster_inds.ndim == 1:
            cluster_batch_idx = cluster_inds
        else:
            cluster_batch_idx = cluster_inds[:, 1]

        num_total_samples = len(reg_preds)

        num_task_classes = len(self.tasks[task_id]['class_names'])
        targets = self.get_targets(num_task_classes, cluster_xyz, cluster_batch_idx, gt_bboxes_3d, gt_labels_3d, reg_preds)
        labels, label_weights, bbox_targets, bbox_weights, iou_labels = targets
        assert (label_weights == 1).all(), 'for now'

        cls_avg_factor = num_total_samples * 1.0
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                bbox_weights.new_tensor([cls_avg_factor]))

        loss_cls = self.loss_cls(
            cls_logits, labels, label_weights, avg_factor=cls_avg_factor)

        # regression loss
        pos_inds = ((labels >= 0)& (labels < num_task_classes)).nonzero(as_tuple=False).reshape(-1)
        num_pos = len(pos_inds)

        pos_reg_preds = reg_preds[pos_inds]
        pos_bbox_targets = bbox_targets[pos_inds]
        pos_bbox_weights = bbox_weights[pos_inds]

        reg_avg_factor = num_pos * 1.0
        if self.sync_reg_avg_factor:
            reg_avg_factor = reduce_mean(
                bbox_weights.new_tensor([reg_avg_factor]))

        if num_pos > 0:
            code_weight = self.train_cfg.get('code_weight', None)
            if code_weight:
                pos_bbox_weights = pos_bbox_weights * bbox_weights.new_tensor(
                    code_weight)


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
            if self.loss_vel is not None:
                loss_vel = self.loss_vel(
                    pos_reg_preds[:, 8:10],
                    pos_bbox_targets[:, 8:10],
                    pos_bbox_weights[:, 8:10],
                )
        else:
            loss_center = pos_reg_preds.sum() * 0
            loss_size = pos_reg_preds.sum() * 0
            loss_rot = pos_reg_preds.sum() * 0
            if self.loss_vel is not None:
                loss_vel = pos_reg_preds.sum() * 0
        
        losses = dict(
            loss_cls=loss_cls,
            loss_center=loss_center,
            loss_size=loss_size,
            loss_rot=loss_rot,
        )
        if self.loss_vel is not None:
            losses['loss_vel'] = loss_vel

        if self.corner_loss_cfg is not None:
            losses['loss_corner'] = self.get_corner_loss(pos_reg_preds, pos_bbox_targets, cluster_xyz[pos_inds], reg_avg_factor)

        if self.loss_iou is not None:
            losses['loss_iou'] = self.loss_iou(iou_logits.reshape(-1), iou_labels, label_weights, avg_factor=cls_avg_factor)
            losses['max_iou'] = iou_labels.max()
            losses['mean_iou'] = iou_labels[iou_labels > 0].mean()
        
        losses_with_task_id = {k + '.task' + str(task_id): v for k, v in losses.items()}

        return losses_with_task_id

    def modify_gt_for_single_task(self, gt_bboxes_3d, gt_labels_3d, task_id):
        out_bboxes_list, out_labels_list = [], []
        for gts_b, gts_l in zip(gt_bboxes_3d, gt_labels_3d):
            out_b, out_l = self.modify_gt_for_single_task_single_sample(gts_b, gts_l, task_id)
            out_bboxes_list.append(out_b)
            out_labels_list.append(out_l)
        return out_bboxes_list, out_labels_list
    
    def modify_gt_for_single_task_single_sample(self, gt_bboxes_3d, gt_labels_3d, task_id):
        assert gt_bboxes_3d.tensor.size(0) == gt_labels_3d.size(0)
        if gt_labels_3d.size(0) == 0:
            return gt_bboxes_3d, gt_labels_3d
        assert (gt_labels_3d >= 0).all() # I don't want -1 in gt_labels_3d

        class_names_this_task = self.tasks[task_id]['class_names']
        num_classes_this_task = len(class_names_this_task)
        out_gt_bboxes_list = []
        out_labels_list = []
        for i, name in enumerate(class_names_this_task):
            cls_id = self.class_names.index(name)
            this_cls_mask = gt_labels_3d == cls_id
            out_gt_bboxes_list.append(gt_bboxes_3d[this_cls_mask])
            out_labels_list.append(gt_labels_3d.new_ones(this_cls_mask.sum()) * i)
        out_gt_bboxes_3d = gt_bboxes_3d.cat(out_gt_bboxes_list)
        out_labels = torch.cat(out_labels_list, dim=0)
        if len(out_labels) > 0:
            assert out_labels.max().item() < num_classes_this_task
        return out_gt_bboxes_3d, out_labels

    def get_targets(self,
                    num_task_classes,
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

        num_task_class_list = [num_task_classes,] * len(cluster_xyz_list)
        target_list_per_sample = multi_apply(self.get_targets_single, num_task_class_list, cluster_xyz_list, gt_bboxes_3d, gt_labels_3d, reg_preds_list)
        targets = [self.combine_by_batch(t, batch_idx, batch_size) for t in target_list_per_sample]
        # targets == [labels, label_weights, bbox_targets, bbox_weights]
        return targets

    def get_targets_single(self,
                           num_task_classes,
                           cluster_xyz,
                           gt_bboxes_3d,
                           gt_labels_3d,
                           reg_preds=None):
        """Generate targets of vote head for single batch.

        """
        num_cluster = len(cluster_xyz)
        labels = gt_labels_3d.new_full((num_cluster, ), num_task_classes, dtype=torch.long)
        label_weights = cluster_xyz.new_ones(num_cluster)
        bbox_targets = cluster_xyz.new_zeros((num_cluster, self.box_code_size))
        bbox_weights = cluster_xyz.new_zeros((num_cluster, self.box_code_size))
        if num_cluster == 0:
            iou_labels = None
            if self.loss_iou is not None:
                iou_labels = cluster_xyz.new_zeros(0)
            return labels, label_weights, bbox_targets, bbox_weights, iou_labels

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
        labels[pos_inds] = gt_labels_3d[sample_result.pos_assigned_gt_inds]
        assert (labels >= 0).all()
        bbox_weights[pos_inds] = 1.0

        if len(pos_inds) > 0:
            bbox_targets[pos_inds] = self.bbox_coder.encode(sample_result.pos_gt_bboxes, cluster_xyz[pos_inds])
            if sample_result.pos_gt_bboxes.size(1) == 10: 
                # zeros velocity loss weight for pasted objects
                assert sample_result.pos_gt_bboxes[:, 9].max().item() in (0, 1)
                assert sample_result.pos_gt_bboxes[:, 9].min().item() in (0, 1)
                assert bbox_weights.size(1) == 10, 'It is not safe to use -2: as follows if size(1) != 10'
                bbox_weights[pos_inds, -2:] = sample_result.pos_gt_bboxes[:, [9]]

        if self.loss_iou is not None:
            iou_labels = self.get_iou_labels(reg_preds, cluster_xyz, gt_bboxes_3d.tensor, pos_inds)
        else:
            iou_labels = None

        return labels, label_weights, bbox_targets, bbox_weights, iou_labels
    

        # generate votes target
    def enlarge_gt_bboxes(self, gt_bboxes_3d, gt_labels_3d=None):
        if self.enlarge_width is not None:
            return gt_bboxes_3d.enlarged_box(self.enlarge_width)
        else:
            return gt_bboxes_3d

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


        assert isinstance(cls_logits, list)
        assert isinstance(reg_preds, list)

        assert len(cls_logits) == len(reg_preds) == len(self.tasks)
        alltask_result_list = []
        for i in range(len(self.tasks)):
            res_this_task = self.get_bboxes_single_task(
                i,
                cls_logits[i],
                reg_preds[i],
                cluster_xyz,
                cluster_inds,
                input_metas,
                iou_logits,
                rescale,
            )
            alltask_result_list.append(res_this_task)
        

        # concat results, I guess len of return list should equal to batch_size
        batch_size = len(input_metas)
        real_batch_size = len(alltask_result_list[0])
        assert  real_batch_size <= batch_size # may less than batch_size if no 
        concat_list = [] 


        for b_idx in range(batch_size):
            boxes = LiDARInstance3DBoxes.cat([task_res[b_idx][0] for task_res in alltask_result_list])
            score = torch.cat([task_res[b_idx][1] for task_res in alltask_result_list], dim=0)
            label = torch.cat([task_res[b_idx][2] for task_res in alltask_result_list], dim=0)
            concat_list.append((boxes, score, label))

        return concat_list


    @torch.no_grad()
    def get_bboxes_single_task(
        self,
        task_id,
        cls_logits,
        reg_preds,
        cluster_xyz,
        cluster_inds,
        input_metas,
        iou_logits=None,
        rescale=False,
        ):

        if cluster_inds.ndim == 1:
            batch_inds = cluster_inds
        else:
            batch_inds = cluster_inds[:, 1]

        batch_size = len(input_metas)
        cls_logits_list = self.split_by_batch(cls_logits, batch_inds, batch_size)
        reg_preds_list = self.split_by_batch(reg_preds, batch_inds, batch_size)
        cluster_xyz_list = self.split_by_batch(cluster_xyz, batch_inds, batch_size)

        if iou_logits is not None:
            iou_logits_list = self.split_by_batch(iou_logits, batch_inds, batch_size)
        else:
            iou_logits_list = [None,] * len(cls_logits_list)

        task_id_repeat = [task_id, ] * len(cls_logits_list)
        multi_results = multi_apply(
            self._get_bboxes_single,
            task_id_repeat,
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
            task_id,
            cls_logits,
            iou_logits,
            reg_preds,
            cluster_xyz,
            input_meta,
        ):
        '''
        Get bboxes of a single sample
        '''

        if self.as_rpn:
            cfg = self.train_cfg.rpn if self.training else self.test_cfg.rpn
        else:
            cfg = self.test_cfg

        assert cls_logits.size(0) == reg_preds.size(0) == cluster_xyz.size(0)
        assert cls_logits.size(1) == len(self.tasks[task_id]['class_names'])
        assert reg_preds.size(1) == self.box_code_size

        if len(cls_logits) == 0:
            out_bboxes = reg_preds.new_zeros((0, 7))
            out_bboxes = input_meta['box_type_3d'](out_bboxes)
            out_scores = reg_preds.new_zeros(0)
            out_labels = reg_preds.new_zeros(0)
            return (out_bboxes, out_scores, out_labels)

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
        bboxes_for_nms = xywhr2xyxyr(input_meta.box_type_3d(bboxes, box_dim=bboxes.size(1)).bev)

        # Add a dummy background class to the front when using sigmoid
        padding = scores.new_zeros(scores.shape[0], 1)
        scores = torch.cat([scores, padding], dim=1)

        score_thr = cfg.get('score_thr', 0)
        results = box3d_multiclass_nms(bboxes, bboxes_for_nms,
                                    scores, score_thr, cfg.max_num,
                                    cfg)

        out_bboxes, out_scores, out_labels = results

        out_bboxes = input_meta.box_type_3d(out_bboxes, out_bboxes.size(1))

        # modify task labels to global label indices
        new_labels = torch.zeros_like(out_labels) - 1 # all -1 
        if len(out_labels) > 0:
            for i, name in enumerate(self.tasks[task_id]['class_names']):
                global_cls_ind = self.class_names.index(name)
                new_labels[out_labels == i] = global_cls_ind

            assert (new_labels >= 0).all()

        out_labels = new_labels

        return (out_bboxes, out_scores, out_labels)