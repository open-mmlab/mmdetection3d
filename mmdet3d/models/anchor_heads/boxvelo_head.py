import numpy as np
import torch
from mmcv.cnn import bias_init_with_prob, normal_init

from mmdet3d.core import box_torch_ops, boxes3d_to_bev_torch_lidar
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu, nms_normal_gpu
from mmdet.models import HEADS
from .second_head import SECONDHead


@HEADS.register_module
class Anchor3DVeloHead(SECONDHead):
    """Anchor-based head for 3D anchor with velocity
    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels of the feature map.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
    """  # noqa: W605

    def __init__(self,
                 class_names,
                 num_classes,
                 in_channels,
                 train_cfg,
                 test_cfg,
                 feat_channels=256,
                 use_direction_classifier=True,
                 encode_bg_as_zeros=False,
                 box_code_size=9,
                 anchor_generator=dict(
                     type='Anchor3DRangeGenerator',
                     range=[0, -39.68, -1.78, 69.12, 39.68, -1.78],
                     strides=[2],
                     sizes=[[1.6, 3.9, 1.56]],
                     rotations=[0, 1.57],
                     custom_values=[0, 0],
                     reshape_out=True,
                 ),
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
        super().__init__(class_names, in_channels, train_cfg, test_cfg,
                         feat_channels, use_direction_classifier,
                         encode_bg_as_zeros, box_code_size, anchor_generator,
                         assigner_per_size, assign_per_class, diff_rad_by_sin,
                         dir_offset, dir_limit_offset, bbox_coder, loss_cls,
                         loss_bbox, loss_dir)
        self.num_classes = num_classes
        # build head layers & losses
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self._init_layers()

    def init_weights(self):
        # pass
        # use the initialization when ready
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        normal_init(self.conv_reg, std=0.01)

    @staticmethod
    def add_sin_difference(boxes1, boxes2):
        # Caution: the 7th dim is the rotation, (last dim without velo)
        rad_pred_encoding = torch.sin(boxes1[..., 6:7]) * torch.cos(
            boxes2[..., 6:7])
        rad_tg_encoding = torch.cos(boxes1[..., 6:7]) * torch.sin(boxes2[...,
                                                                         6:7])
        boxes1 = torch.cat(
            [boxes1[..., :6], rad_pred_encoding, boxes1[..., 7:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :6], rad_tg_encoding, boxes2[..., 7:]],
                           dim=-1)
        return boxes1, boxes2

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          dir_cls_preds,
                          mlvl_anchors,
                          input_meta,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []
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

            nms_pre = self.test_cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                dir_cls_score = dir_cls_score[topk_inds]

            bboxes = self.bbox_coder.decode(anchors, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_bboxes_for_nms = boxes3d_to_bev_torch_lidar(mlvl_bboxes)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)

        if self.use_sigmoid_cls:
            # Add a dummy background class to the front when using sigmoid
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        score_thr = self.test_cfg.get('score_thr', 0)
        result = self.multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
                                     mlvl_scores, mlvl_dir_scores, score_thr,
                                     self.test_cfg.max_per_img)

        result.update(dict(sample_idx=input_meta['sample_idx']))
        return result

    def multiclass_nms(self, mlvl_bboxes, mlvl_bboxes_for_nms, mlvl_scores,
                       mlvl_dir_scores, score_thr, max_num):
        # do multi class nms
        # the fg class id range: [0, num_classes-1]
        num_classes = mlvl_scores.shape[1] - 1
        bboxes = []
        scores = []
        labels = []
        dir_scores = []
        for i in range(0, num_classes):
            # get bboxes and scores of this class
            cls_inds = mlvl_scores[:, i] > score_thr
            if not cls_inds.any():
                continue
            _scores = mlvl_scores[cls_inds, i]
            _bboxes_for_nms = mlvl_bboxes_for_nms[cls_inds, :]
            if self.test_cfg.use_rotate_nms:
                nms_func = nms_gpu
            else:
                nms_func = nms_normal_gpu
            selected = nms_func(_bboxes_for_nms, _scores,
                                self.test_cfg.nms_thr)

            _mlvl_bboxes = mlvl_bboxes[cls_inds, :]
            _mlvl_dir_scores = mlvl_dir_scores[cls_inds]

            if len(selected) > 0:
                bboxes.append(_mlvl_bboxes[selected])
                scores.append(_scores[selected])
                dir_scores.append(_mlvl_dir_scores[selected])
                dir_rot = box_torch_ops.limit_period(
                    bboxes[-1][..., 6] - self.dir_offset,
                    self.dir_limit_offset, np.pi)
                bboxes[-1][..., 6] = (
                    dir_rot + self.dir_offset +
                    np.pi * dir_scores[-1].to(bboxes[-1].dtype))

                cls_label = mlvl_bboxes.new_full((len(selected), ),
                                                 i,
                                                 dtype=torch.long)
                labels.append(cls_label)

        if bboxes:
            bboxes = torch.cat(bboxes, dim=0)
            scores = torch.cat(scores, dim=0)
            labels = torch.cat(labels, dim=0)
            dir_scores = torch.cat(dir_scores, dim=0)
            if bboxes.shape[0] > max_num:
                _, inds = scores.sort(descending=True)
                inds = inds[:max_num]
                bboxes = bboxes[inds, :]
                labels = labels[inds]
                scores = scores[inds]
                dir_scores = dir_scores[inds]
            return dict(
                box3d_lidar=bboxes.cpu(),
                scores=scores.cpu(),
                label_preds=labels.cpu(),
            )
        else:
            return dict(
                box3d_lidar=mlvl_bboxes.new_zeros([0,
                                                   self.box_code_size]).cpu(),
                scores=mlvl_bboxes.new_zeros([0]).cpu(),
                label_preds=mlvl_bboxes.new_zeros([0, 4]).cpu(),
            )
