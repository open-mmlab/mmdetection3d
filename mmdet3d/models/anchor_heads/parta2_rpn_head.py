from __future__ import division

import numpy as np
import torch

from mmdet3d.core import box_torch_ops, boxes3d_to_bev_torch_lidar
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu, nms_normal_gpu
from mmdet.models import HEADS
from .second_head import SECONDHead


@HEADS.register_module()
class PartA2RPNHead(SECONDHead):
    """rpn head for PartA2

    Args:
        class_name (list[str]): name of classes (TODO: to be removed)
        in_channels (int): Number of channels in the input feature map.
        train_cfg (dict): train configs
        test_cfg (dict): test configs
        feat_channels (int): Number of channels of the feature map.
        use_direction_classifier (bool): Whether to add a direction classifier.
        encode_bg_as_zeros (bool): Whether to use sigmoid of softmax
            (TODO: to be removed)
        box_code_size (int): The size of box code.
        anchor_generator(dict): Config dict of anchor generator.
        assigner_per_size (bool): Whether to do assignment for each separate
            anchor size.
        assign_per_class (bool): Whether to do assignment for each class.
        diff_rad_by_sin (bool): Whether to change the difference into sin
            difference for box regression loss.
        dir_offset (float | int): The offset of BEV rotation angles
            (TODO: may be moved into box coder)
        dirlimit_offset (float | int): The limited range of BEV rotation angles
            (TODO: may be moved into box coder)
        box_coder (dict): Config dict of box coders.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_dir (dict): Config of direction classifier loss.
    """  # npqa:W293

    def __init__(self,
                 class_name,
                 in_channels,
                 train_cfg,
                 test_cfg,
                 feat_channels=256,
                 use_direction_classifier=True,
                 encode_bg_as_zeros=False,
                 box_code_size=7,
                 anchor_generator=dict(
                     type='Anchor3DRangeGenerator',
                     range=[0, -39.68, -1.78, 69.12, 39.68, -1.78],
                     strides=[2],
                     sizes=[[1.6, 3.9, 1.56]],
                     rotations=[0, 1.57],
                     custom_values=[],
                     reshape_out=False),
                 assigner_per_size=False,
                 assign_per_class=False,
                 diff_rad_by_sin=True,
                 dir_offset=0,
                 dir_limit_offset=1,
                 bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
                 loss_dir=dict(type='CrossEntropyLoss', loss_weight=0.2)):
        super().__init__(class_name, in_channels, train_cfg, test_cfg,
                         feat_channels, use_direction_classifier,
                         encode_bg_as_zeros, box_code_size, anchor_generator,
                         assigner_per_size, assign_per_class, diff_rad_by_sin,
                         dir_offset, dir_limit_offset, bbox_coder, loss_cls,
                         loss_bbox, loss_dir)

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   dir_cls_preds,
                   input_metas,
                   cfg,
                   rescale=False):
        assert len(cls_scores) == len(bbox_preds)
        assert len(cls_scores) == len(dir_cls_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        device = cls_scores[0].device
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)
        mlvl_anchors = [
            anchor.reshape(-1, self.box_code_size) for anchor in mlvl_anchors
        ]

        result_list = []
        for img_id in range(len(input_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            dir_cls_pred_list = [
                dir_cls_preds[i][img_id].detach() for i in range(num_levels)
            ]

            input_meta = input_metas[img_id]
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               dir_cls_pred_list, mlvl_anchors,
                                               input_meta, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          dir_cls_preds,
                          mlvl_anchors,
                          input_meta,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_max_scores = []
        mlvl_label_pred = []
        mlvl_dir_scores = []
        mlvl_cls_score = []
        for cls_score, bbox_pred, dir_cls_pred, anchors in zip(
                cls_scores, bbox_preds, dir_cls_preds, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            assert cls_score.size()[-2:] == dir_cls_pred.size()[-2:]
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]

            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.num_classes)

            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2,
                                          0).reshape(-1, self.box_code_size)

            nms_pre = cfg.get('nms_pre', -1)
            if self.use_sigmoid_cls:
                max_scores, pred_labels = scores.max(dim=1)
            else:
                max_scores, pred_labels = scores[:, :-1].max(dim=1)
            # get topk
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                topk_scores, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                max_scores = topk_scores
                cls_score = cls_score[topk_inds, :]
                dir_cls_score = dir_cls_score[topk_inds]
                pred_labels = pred_labels[topk_inds]

            bboxes = self.bbox_coder.decode(anchors, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_max_scores.append(max_scores)
            mlvl_cls_score.append(cls_score)
            mlvl_label_pred.append(pred_labels)
            mlvl_dir_scores.append(dir_cls_score)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_bboxes_for_nms = boxes3d_to_bev_torch_lidar(mlvl_bboxes)
        mlvl_max_scores = torch.cat(mlvl_max_scores)
        mlvl_label_pred = torch.cat(mlvl_label_pred)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)
        mlvl_cls_score = torch.cat(
            mlvl_cls_score)  # shape [k, num_class] before sigmoid

        score_thr = cfg.get('score_thr', 0)
        result = self.class_agnostic_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
                                         mlvl_max_scores, mlvl_label_pred,
                                         mlvl_cls_score, mlvl_dir_scores,
                                         score_thr, cfg.nms_post, cfg)

        result.update(dict(sample_idx=input_meta['sample_idx']))
        return result

    def class_agnostic_nms(self, mlvl_bboxes, mlvl_bboxes_for_nms,
                           mlvl_max_scores, mlvl_label_pred, mlvl_cls_score,
                           mlvl_dir_scores, score_thr, max_num, cfg):
        bboxes = []
        scores = []
        labels = []
        dir_scores = []
        cls_scores = []
        score_thr_inds = mlvl_max_scores > score_thr
        _scores = mlvl_max_scores[score_thr_inds]
        _bboxes_for_nms = mlvl_bboxes_for_nms[score_thr_inds, :]
        if cfg.use_rotate_nms:
            nms_func = nms_gpu
        else:
            nms_func = nms_normal_gpu
        selected = nms_func(_bboxes_for_nms, _scores, cfg.nms_thr)

        _mlvl_bboxes = mlvl_bboxes[score_thr_inds, :]
        _mlvl_dir_scores = mlvl_dir_scores[score_thr_inds]
        _mlvl_label_pred = mlvl_label_pred[score_thr_inds]
        _mlvl_cls_score = mlvl_cls_score[score_thr_inds]

        if len(selected) > 0:
            bboxes.append(_mlvl_bboxes[selected])
            scores.append(_scores[selected])
            labels.append(_mlvl_label_pred[selected])
            cls_scores.append(_mlvl_cls_score[selected])
            dir_scores.append(_mlvl_dir_scores[selected])
            dir_rot = box_torch_ops.limit_period(
                bboxes[-1][..., 6] - self.dir_offset, self.dir_limit_offset,
                np.pi)
            bboxes[-1][..., 6] = (
                dir_rot + self.dir_offset +
                np.pi * dir_scores[-1].to(bboxes[-1].dtype))

        if bboxes:
            bboxes = torch.cat(bboxes, dim=0)
            scores = torch.cat(scores, dim=0)
            cls_scores = torch.cat(cls_scores, dim=0)
            labels = torch.cat(labels, dim=0)
            dir_scores = torch.cat(dir_scores, dim=0)
            if bboxes.shape[0] > max_num:
                _, inds = scores.sort(descending=True)
                inds = inds[:max_num]
                bboxes = bboxes[inds, :]
                labels = labels[inds]
                scores = scores[inds]
                cls_scores = cls_scores[inds]
            return dict(
                box3d_lidar=bboxes,
                scores=scores,
                label_preds=labels,
                cls_preds=cls_scores  # raw scores [max_num, cls_num]
            )
        else:
            return dict(
                box3d_lidar=mlvl_bboxes.new_zeros([0, self.box_code_size]),
                scores=mlvl_bboxes.new_zeros([0]),
                label_preds=mlvl_bboxes.new_zeros([0]),
                cls_preds=mlvl_bboxes.new_zeros([0, mlvl_cls_score.shape[-1]]))
