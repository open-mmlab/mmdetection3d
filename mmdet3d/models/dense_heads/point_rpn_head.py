import torch
from mmcv.ops.nms import batched_nms
from mmcv.runner import BaseModule, force_fp32
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.core import xywhr2xyxyr
from mmdet3d.core.bbox.structures import (DepthInstance3DBoxes,
                                          LiDARInstance3DBoxes)
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
from mmdet.core import build_bbox_coder, multi_apply
from mmdet.models import HEADS, build_loss


@HEADS.register_module()
class PointRPNHead(BaseModule):

    def __init__(self,
                 num_classes,
                 num_dir_bins,
                 train_cfg,
                 test_cfg,
                 pred_layer_cfg=None,
                 cls_loss=None,
                 bbox_loss=None,
                 corner_loss=None,
                 bbox_coder=None,
                 init_cfg=None,
                 pretrained=None):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.num_dir_bins = num_dir_bins
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.num_classes = num_classes
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # build loss function
        self.bbox_loss = build_loss(bbox_loss)
        self.cls_loss = build_loss(cls_loss)
        self.corner_loss = build_loss(corner_loss)

        # build box coder
        self.bbox_coder = build_bbox_coder(bbox_coder)

        # build pred conv
        self.cls_layers = self._make_fc_layers(
            fc_cfg=pred_layer_cfg.cls_linear_channels,
            input_channels=pred_layer_cfg.in_channels,
            output_channels=self._get_cls_out_channels())

        self.reg_layers = self._make_fc_layers(
            fc_cfg=pred_layer_cfg.reg_linear_channels,
            input_channels=pred_layer_cfg.in_channels,
            output_channels=self._get_reg_out_channels())

    def _make_fc_layers(self, fc_cfg, input_channels, output_channels):
        fc_layers = []
        c_in = input_channels
        for k in range(0, fc_cfg.__len__()):
            fc_layers.extend([
                nn.Linear(c_in, fc_cfg[k], bias=False),
                nn.BatchNorm1d(fc_cfg[k]),
                nn.ReLU(),
            ])
            c_in = fc_cfg[k]
        fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
        return nn.Sequential(*fc_layers)

    def _get_cls_out_channels(self):
        """Return the channel number of classification outputs."""
        # Class numbers (k) + objectness (1)
        return self.num_classes

    def _get_reg_out_channels(self):
        """Return the channel number of regression outputs."""
        # Bbox classification and regression
        # (center residual (3), size regression (3)
        # torch.cos(yaw) (1), torch.sin(yaw)(1)
        return 3 + 3 + 2

    def forward(self, feat_dict):
        point_features = feat_dict['fp_features']
        point_features = point_features.permute(0, 2, 1).contiguous()
        bs = point_features.shape[0]
        x_cls = point_features.view(-1, point_features.shape[-1])
        x_reg = point_features.view(-1, point_features.shape[-1])

        point_cls_preds = self.cls_layers(x_cls).reshape(
            bs, -1, self._get_cls_out_channels())
        point_box_preds = self.reg_layers(x_reg).reshape(
            bs, -1, self._get_reg_out_channels())
        return (point_box_preds, point_cls_preds)

    @force_fp32(apply_to=('bbox_preds'))
    def loss(self,
             bbox_preds,
             cls_preds,
             points,
             gt_bboxes_3d,
             gt_labels_3d,
             img_metas=None):
        """Compute loss.

        Args:
            bbox_preds (dict): Predictions from forward of PointRCNN \
                RPN_Head.
            cls_preds (dict): Classification from forward of PointRCNN \
                RPN_Head.
            points (list[torch.Tensor]): Input points.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth \
                bboxes of each sample.
            gt_labels_3d (list[torch.Tensor]): Labels of each sample.
            img_metas (list[dict]): Contain pcd and img's meta info.

        Returns:
            dict: Losses of PointRCNN.
        """
        targets = self.get_targets(points, gt_bboxes_3d, gt_labels_3d)
        (bbox_targets, mask_targets, positive_mask, negative_mask,
         box_loss_weights, point_targets) = targets

        # bbox loss
        bbox_loss = self.bbox_loss(bbox_preds, bbox_targets,
                                   box_loss_weights.unsqueeze(-1))
        # calculate semantic loss
        semantic_points = cls_preds.reshape(-1, self.num_classes)
        semantic_targets = mask_targets
        semantic_targets[negative_mask] = self.num_classes
        semantic_points_label = semantic_targets
        # for ignore, but now we do not have ignore label
        semantic_loss_weight = negative_mask.float() + positive_mask.float()
        semantic_loss = self.cls_loss(semantic_points,
                                      semantic_points_label.reshape(-1),
                                      semantic_loss_weight.reshape(-1))
        semantic_loss /= positive_mask.float().sum()
        losses = dict(bbox_loss=bbox_loss, semantic_loss=semantic_loss)

        return losses

    def get_targets(self, points, gt_bboxes_3d, gt_labels_3d):
        """Generate targets of ssd3d head.

        Args:
            points (list[torch.Tensor]): Points of each batch.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth \
                bboxes of each batch.
            gt_labels_3d (list[torch.Tensor]): Labels of each batch.

        Returns:
            tuple[torch.Tensor]: Targets of PointRCNN RPN head.
        """
        # find empty example
        for index in range(len(gt_labels_3d)):
            if len(gt_labels_3d[index]) == 0:
                fake_box = gt_bboxes_3d[index].tensor.new_zeros(
                    1, gt_bboxes_3d[index].tensor.shape[-1])
                gt_bboxes_3d[index] = gt_bboxes_3d[index].new_box(fake_box)
                gt_labels_3d[index] = gt_labels_3d[index].new_zeros(1)

        (bbox_targets, mask_targets, positive_mask, negative_mask,
         point_targets) = multi_apply(self.get_targets_single, points,
                                      gt_bboxes_3d, gt_labels_3d)

        bbox_targets = torch.stack(bbox_targets)
        mask_targets = torch.stack(mask_targets)
        positive_mask = torch.stack(positive_mask)
        negative_mask = torch.stack(negative_mask)
        box_loss_weights = positive_mask / (positive_mask.sum() + 1e-6)

        return (bbox_targets, mask_targets, positive_mask, negative_mask,
                box_loss_weights, point_targets)

    def get_targets_single(self, points, gt_bboxes_3d, gt_labels_3d):
        """Generate targets of ssd3d head for single batch.

        Args:
            points (torch.Tensor): Points of each batch.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): Ground truth \
                boxes of each batch.
            gt_labels_3d (torch.Tensor): Labels of each batch.
            pts_semantic_mask (None | torch.Tensor): Point-wise semantic
                label of each batch.
            pts_instance_mask (None | torch.Tensor): Point-wise instance
                label of each batch.
            seed_points (torch.Tensor): Seed points of candidate points.

        Returns:
            tuple[torch.Tensor]: Targets of ssd3d head.
        """
        gt_bboxes_3d = gt_bboxes_3d.to(points.device)

        valid_gt = gt_labels_3d != -1
        gt_bboxes_3d = gt_bboxes_3d[valid_gt]
        gt_labels_3d = gt_labels_3d[valid_gt]

        # transform the bbox coordinate to the pointcloud coordinate
        gt_bboxes_3d_tensor = gt_bboxes_3d.tensor.clone()
        gt_bboxes_3d_tensor[..., 2] += gt_bboxes_3d_tensor[..., 5] / 2

        points_mask, assignment = self._assign_targets_by_points_inside(
            gt_bboxes_3d, points)
        gt_bboxes_3d_tensor = gt_bboxes_3d_tensor[assignment]
        mask_targets = gt_labels_3d[assignment]

        bbox_targets = self.bbox_coder.encode(gt_bboxes_3d_tensor,
                                              points[..., 0:3], mask_targets)

        positive_mask = (points_mask.max(1)[0] > 0)
        negative_mask = (points_mask.max(1)[0] == 0)
        # add ignore_mask
        extend_gt_bboxes_3d = gt_bboxes_3d.enlarged_box(0.1)
        points_mask, _ = self._assign_targets_by_points_inside(
            extend_gt_bboxes_3d, points)
        negative_mask = (points_mask.max(1)[0] == 0)

        point_targets = points[..., 0:3]
        return (bbox_targets, mask_targets, positive_mask, negative_mask,
                point_targets)

    def get_bboxes(self,
                   points,
                   bbox_preds,
                   cls_preds,
                   input_metas,
                   rescale=False):
        """Generate bboxes from sdd3d head predictions.

        Args:
            points (torch.Tensor): Input points.
            bbox_preds (dict): Predictions from PointRCNN head.
            input_metas (list[dict]): Point cloud and image's meta info.
            rescale (bool): Whether to rescale bboxes.

        Returns:
            list[tuple[torch.Tensor]]: Bounding boxes, scores and labels.
        """
        sem_scores = F.sigmoid(cls_preds)
        obj_scores = sem_scores.max(-1)[0]
        object_class = torch.argmax(sem_scores, -1)

        batch_size = sem_scores.shape[0]
        results = list()
        for b in range(batch_size):
            bbox3d = self.bbox_coder.decode(bbox_preds[b], points[b, ..., :3],
                                            object_class[b])
            bbox_selected, score_selected, labels, cls_preds_selected = \
                self.multiclass_nms_single(
                    obj_scores[b], sem_scores[b],
                    bbox3d, points[b, ..., :3],
                    input_metas[b])
            bbox = input_metas[b]['box_type_3d'](
                bbox_selected.clone(),
                box_dim=bbox_selected.shape[-1],
                with_yaw=True)
            results.append((bbox, score_selected, labels, cls_preds_selected))
        return results

    def class_agnostic_nms(self, obj_scores, sem_scores, bbox, points,
                           input_meta, training_flag):
        """Multi-class nms in single batch.

        Args:
            obj_scores (torch.Tensor): Objectness score of bounding boxes.
            sem_scores (torch.Tensor): semantic class score of bounding boxes.
            bbox (torch.Tensor): Predicted bounding boxes.
            points (torch.Tensor): Input points.
            input_meta (dict): Point cloud and image's meta info.

        Returns:
            tuple[torch.Tensor]: Bounding boxes, scores and labels.
        """
        bbox = input_meta['box_type_3d'](
            bbox.clone(),
            box_dim=bbox.shape[-1],
            with_yaw=True,
            origin=(0.5, 0.5, 0.5))
        bbox_for_nms = xywhr2xyxyr(bbox.bev)

        bbox_selected = []
        score_selected = []
        labels = []
        cls_preds = []

        num_rpn_proposal = self.test_cfg.max_output_num
        nms_cfg = self.test_cfg.nms_cfg
        score_thr = self.test_cfg.score_thr
        nms_pre = self.test_cfg.nms_pre
        if training_flag:
            num_rpn_proposal = self.train_cfg.rpn_proposal.max_num
            nms_cfg = self.train_cfg.rpn_proposal.nms_cfg
            score_thr = self.train_cfg.rpn_proposal.score_thr
            nms_pre = self.train_cfg.rpn_proposal.nms_pre

        score_thr_inds = obj_scores >= score_thr
        _scores = obj_scores[score_thr_inds]
        _bboxes_for_nms = bbox_for_nms[score_thr_inds, :]
        _sem_scores = sem_scores[score_thr_inds]
        classes = torch.argmax(_sem_scores, -1)
        _classes = classes[score_thr_inds]
        _bbox = bbox[score_thr_inds]
        # select top k bbox
        _, indices = _scores.topk(min(nms_pre, _scores.shape[0]))
        _scores = _scores[indices]
        _bboxes_for_nms = _bboxes_for_nms[indices]
        _classes = _classes[indices]
        _sem_scores = _sem_scores[indices]
        _bbox = _bbox[indices]
        selected = nms_gpu(_bboxes_for_nms, _scores, nms_cfg.iou_thr)
        if selected.shape[0] > num_rpn_proposal:
            selected = selected[:num_rpn_proposal]

        if len(selected) > 0:
            bbox_selected = _bbox[selected].tensor
            score_selected = _scores[selected]
            labels = _classes[selected]
            cls_preds = _sem_scores[selected]
        return bbox_selected, score_selected, labels, cls_preds

    def multiclass_nms_single(self, obj_scores, sem_scores, bbox, points,
                              input_meta):
        """Multi-class nms in single batch.

        Args:
            obj_scores (torch.Tensor): Objectness score of bounding boxes.
            sem_scores (torch.Tensor): semantic class score of bounding boxes.
            bbox (torch.Tensor): Predicted bounding boxes.
            points (torch.Tensor): Input points.
            input_meta (dict): Point cloud and image's meta info.

        Returns:
            tuple[torch.Tensor]: Bounding boxes, scores and labels.
        """

        num_rpn_proposal = self.test_cfg.max_output_num
        nms_cfg = self.test_cfg.nms_cfg
        score_thr = self.test_cfg.score_thr
        if self.training:
            num_rpn_proposal = self.train_cfg.rpn_proposal.max_num
            nms_cfg = self.train_cfg.rpn_proposal.nms_cfg
            score_thr = self.train_cfg.rpn_proposal.score_thr

        num_bbox = bbox.shape[0]
        bbox = input_meta['box_type_3d'](
            bbox.clone(),
            box_dim=bbox.shape[-1],
            with_yaw=True,
            origin=(0.5, 0.5, 0.5))

        if isinstance(bbox, LiDARInstance3DBoxes):
            box_idx = bbox.points_in_boxes(points)
            box_indices = box_idx.new_zeros([num_bbox + 1])
            box_idx[box_idx == -1] = num_bbox
            box_indices.scatter_add_(0, box_idx.long(),
                                     box_idx.new_ones(box_idx.shape))
            box_indices = box_indices[:-1]
            nonempty_box_mask = box_indices >= 0
        elif isinstance(bbox, DepthInstance3DBoxes):
            box_indices = bbox.points_in_boxes(points)
            nonempty_box_mask = box_indices.T.sum(1) >= 0
        else:
            raise NotImplementedError('Unsupported bbox type!')

        corner3d = bbox.corners
        minmax_box3d = corner3d.new(torch.Size((corner3d.shape[0], 6)))
        minmax_box3d[:, :3] = torch.min(corner3d, dim=1)[0]
        minmax_box3d[:, 3:] = torch.max(corner3d, dim=1)[0]

        bbox_classes = torch.argmax(sem_scores, -1)
        nms_selected = batched_nms(
            minmax_box3d[nonempty_box_mask][:, [0, 1, 3, 4]].detach(),
            obj_scores[nonempty_box_mask].detach(),
            bbox_classes[nonempty_box_mask].detach(), nms_cfg, True)[1]

        if nms_selected.shape[0] > num_rpn_proposal:
            nms_selected = nms_selected[:num_rpn_proposal]

        # filter empty boxes and boxes with low score
        scores_mask = (obj_scores >= score_thr)
        nonempty_box_inds = torch.nonzero(
            nonempty_box_mask, as_tuple=False).flatten()
        nonempty_mask = torch.zeros_like(bbox_classes).scatter(
            0, nonempty_box_inds[nms_selected], 1)
        selected = (nonempty_mask.bool() & scores_mask.bool())

        if self.test_cfg.per_class_proposal:
            bbox_selected, score_selected, labels, cls_preds = [], [], [], []
            for k in range(sem_scores.shape[-1]):
                bbox_selected.append(bbox[selected].tensor)
                score_selected.append(obj_scores[selected])
                cls_preds.append(sem_scores[selected])
                labels.append(
                    torch.zeros_like(bbox_classes[selected]).fill_(k))
            bbox_selected = torch.cat(bbox_selected, 0)
            score_selected = torch.cat(score_selected, 0)
            labels = torch.cat(labels, 0)
            cls_preds = torch.cat(cls_preds, 0)
        else:
            bbox_selected = bbox[selected].tensor
            score_selected = obj_scores[selected]
            labels = bbox_classes[selected]
            cls_preds = sem_scores[selected]

        return bbox_selected, score_selected, labels, cls_preds

    def _assign_targets_by_points_inside(self, bboxes_3d, points):
        """Compute assignment by checking whether point is inside bbox.

        Args:
            bboxes_3d (BaseInstance3DBoxes): Instance of bounding boxes.
            points (torch.Tensor): Points of a batch.

        Returns:
            tuple[torch.Tensor]: Flags indicating whether each point is
                inside bbox and the index of box where each point are in.
        """
        # TODO: align points_in_boxes function in each box_structures
        num_bbox = bboxes_3d.tensor.shape[0]
        if isinstance(bboxes_3d, LiDARInstance3DBoxes):
            assignment = bboxes_3d.points_in_boxes(points[:, 0:3]).long()
            points_mask = assignment.new_zeros(
                [assignment.shape[0], num_bbox + 1])
            assignment[assignment == -1] = num_bbox
            points_mask.scatter_(1, assignment.unsqueeze(1), 1)
            points_mask = points_mask[:, :-1]
            assignment[assignment == num_bbox] = num_bbox - 1
        elif isinstance(bboxes_3d, DepthInstance3DBoxes):
            points_mask = bboxes_3d.points_in_boxes(points)
            assignment = points_mask.argmax(dim=-1)
        else:
            raise NotImplementedError('Unsupported bbox type!')

        return points_mask, assignment
