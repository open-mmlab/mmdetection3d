# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import torch
from mmengine.device import get_device
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.registry import MODELS
from mmdet3d.structures import bbox_overlaps_nearest_3d
from mmdet3d.utils import InstanceList, OptInstanceList
from .anchor3d_head import Anchor3DHead
from .train_mixins import get_direction_target


@MODELS.register_module()
class FreeAnchor3DHead(Anchor3DHead):
    r"""`FreeAnchor <https://arxiv.org/abs/1909.02466>`_ head for 3D detection.

    Note:
        This implementation is directly modified from the `mmdet implementation
        <https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/dense_heads/free_anchor_retina_head.py>`_.
        We find it also works on 3D detection with minor modification, i.e.,
        different hyper-parameters and a additional direction classifier.

    Args:
        pre_anchor_topk (int): Number of boxes that be token in each bag.
        bbox_thr (float): The threshold of the saturated linear function. It is
            usually the same with the IoU threshold used in NMS.
        gamma (float): Gamma parameter in focal loss.
        alpha (float): Alpha parameter in focal loss.
        kwargs (dict): Other arguments are the same as those in :class:`Anchor3DHead`.
    """  # noqa: E501

    def __init__(self,
                 pre_anchor_topk: int = 50,
                 bbox_thr: float = 0.6,
                 gamma: float = 2.0,
                 alpha: float = 0.5,
                 init_cfg: dict = None,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.pre_anchor_topk = pre_anchor_topk
        self.bbox_thr = bbox_thr
        self.gamma = gamma
        self.alpha = alpha

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            dir_cls_preds: List[Tensor],
            batch_gt_instances_3d: InstanceList,
            batch_input_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> Dict:
        """Calculate loss of FreeAnchor head.

        Args:
            cls_scores (list[torch.Tensor]): Classification scores of
                different samples.
            bbox_preds (list[torch.Tensor]): Box predictions of
                different samples
            dir_cls_preds (list[torch.Tensor]): Direction predictions of
                different samples
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances. It usually includes ``bboxes_3d`` and
                ``labels_3d`` attributes.
            batch_input_metas (list[dict]): Contain pcd and img's meta info.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Loss items.

                - positive_bag_loss (torch.Tensor): Loss of positive samples.
                - negative_bag_loss (torch.Tensor): Loss of negative samples.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = get_device()
        anchor_list = self.get_anchors(featmap_sizes, batch_input_metas,
                                       device)
        mlvl_anchors = [torch.cat(anchor) for anchor in anchor_list]

        # concatenate each level
        cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(
                cls_score.size(0), -1, self.num_classes)
            for cls_score in cls_scores
        ]
        bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(
                bbox_pred.size(0), -1, self.box_code_size)
            for bbox_pred in bbox_preds
        ]
        dir_cls_preds = [
            dir_cls_pred.permute(0, 2, 3,
                                 1).reshape(dir_cls_pred.size(0), -1, 2)
            for dir_cls_pred in dir_cls_preds
        ]

        cls_scores = torch.cat(cls_scores, dim=1)
        bbox_preds = torch.cat(bbox_preds, dim=1)
        dir_cls_preds = torch.cat(dir_cls_preds, dim=1)

        cls_probs = torch.sigmoid(cls_scores)
        box_prob = []
        num_pos = 0
        positive_losses = []
        for _, (anchors, gt_instance_3d, cls_prob, bbox_pred,
                dir_cls_pred) in enumerate(
                    zip(mlvl_anchors, batch_gt_instances_3d, cls_probs,
                        bbox_preds, dir_cls_preds)):

            gt_bboxes = gt_instance_3d.bboxes_3d.tensor.to(anchors.device)
            gt_labels = gt_instance_3d.labels_3d.to(anchors.device)
            with torch.no_grad():
                # box_localization: a_{j}^{loc}, shape: [j, 4]
                pred_boxes = self.bbox_coder.decode(anchors, bbox_pred)

                # object_box_iou: IoU_{ij}^{loc}, shape: [i, j]
                object_box_iou = bbox_overlaps_nearest_3d(
                    gt_bboxes, pred_boxes)

                # object_box_prob: P{a_{j} -> b_{i}}, shape: [i, j]
                t1 = self.bbox_thr
                t2 = object_box_iou.max(
                    dim=1, keepdim=True).values.clamp(min=t1 + 1e-6)
                object_box_prob = ((object_box_iou - t1) / (t2 - t1)).clamp(
                    min=0, max=1)

                # object_cls_box_prob: P{a_{j} -> b_{i}}, shape: [i, c, j]
                num_obj = gt_labels.size(0)
                indices = torch.stack(
                    [torch.arange(num_obj).type_as(gt_labels), gt_labels],
                    dim=0)

                object_cls_box_prob = torch.sparse_coo_tensor(
                    indices, object_box_prob)

                # image_box_iou: P{a_{j} \in A_{+}}, shape: [c, j]
                """
                from "start" to "end" implement:
                image_box_iou = torch.sparse.max(object_cls_box_prob,
                                                 dim=0).t()

                """
                # start
                box_cls_prob = torch.sparse.sum(
                    object_cls_box_prob, dim=0).to_dense()

                indices = torch.nonzero(box_cls_prob, as_tuple=False).t_()
                if indices.numel() == 0:
                    image_box_prob = torch.zeros(
                        anchors.size(0),
                        self.num_classes).type_as(object_box_prob)
                else:
                    nonzero_box_prob = torch.where(
                        (gt_labels.unsqueeze(dim=-1) == indices[0]),
                        object_box_prob[:, indices[1]],
                        torch.tensor(
                            [0]).type_as(object_box_prob)).max(dim=0).values

                    # upmap to shape [j, c]
                    image_box_prob = torch.sparse_coo_tensor(
                        indices.flip([0]),
                        nonzero_box_prob,
                        size=(anchors.size(0), self.num_classes)).to_dense()
                # end

                box_prob.append(image_box_prob)

            # construct bags for objects
            match_quality_matrix = bbox_overlaps_nearest_3d(gt_bboxes, anchors)
            _, matched = torch.topk(
                match_quality_matrix,
                self.pre_anchor_topk,
                dim=1,
                sorted=False)
            del match_quality_matrix

            # matched_cls_prob: P_{ij}^{cls}
            matched_cls_prob = torch.gather(
                cls_prob[matched], 2,
                gt_labels.view(-1, 1, 1).repeat(1, self.pre_anchor_topk,
                                                1)).squeeze(2)

            # matched_box_prob: P_{ij}^{loc}
            matched_anchors = anchors[matched]
            matched_object_targets = self.bbox_coder.encode(
                matched_anchors,
                gt_bboxes.unsqueeze(dim=1).expand_as(matched_anchors))

            # direction classification loss
            loss_dir = None
            if self.use_direction_classifier:
                # also calculate direction prob: P_{ij}^{dir}
                matched_dir_targets = get_direction_target(
                    matched_anchors,
                    matched_object_targets,
                    self.dir_offset,
                    self.dir_limit_offset,
                    one_hot=False)
                loss_dir = self.loss_dir(
                    dir_cls_pred[matched].transpose(-2, -1),
                    matched_dir_targets,
                    reduction_override='none')

            # generate bbox weights
            if self.diff_rad_by_sin:
                bbox_preds_clone = bbox_pred.clone()
                bbox_preds_clone[matched], matched_object_targets = \
                    self.add_sin_difference(
                        bbox_preds_clone[matched], matched_object_targets)
            bbox_weights = matched_anchors.new_ones(matched_anchors.size())
            # Use pop is not right, check performance
            code_weight = self.train_cfg.get('code_weight', None)
            if code_weight:
                bbox_weights = bbox_weights * bbox_weights.new_tensor(
                    code_weight)
            loss_bbox = self.loss_bbox(
                bbox_preds_clone[matched],
                matched_object_targets,
                bbox_weights,
                reduction_override='none').sum(-1)

            if loss_dir is not None:
                loss_bbox += loss_dir
            matched_box_prob = torch.exp(-loss_bbox)

            # positive_losses: {-log( Mean-max(P_{ij}^{cls} * P_{ij}^{loc}) )}
            num_pos += len(gt_bboxes)
            positive_losses.append(
                self.positive_bag_loss(matched_cls_prob, matched_box_prob))

        positive_loss = torch.cat(positive_losses).sum() / max(1, num_pos)

        # box_prob: P{a_{j} \in A_{+}}
        box_prob = torch.stack(box_prob, dim=0)

        # negative_loss:
        # \sum_{j}{ FL((1 - P{a_{j} \in A_{+}}) * (1 - P_{j}^{bg})) } / n||B||
        negative_loss = self.negative_bag_loss(cls_prob, box_prob).sum() / max(
            1, num_pos * self.pre_anchor_topk)

        losses = {
            'positive_bag_loss': positive_loss,
            'negative_bag_loss': negative_loss
        }
        return losses

    def positive_bag_loss(self, matched_cls_prob: Tensor,
                          matched_box_prob: Tensor) -> Tensor:
        """Generate positive bag loss.

        Args:
            matched_cls_prob (torch.Tensor): Classification probability
                of matched positive samples.
            matched_box_prob (torch.Tensor): Bounding box probability
                of matched positive samples.

        Returns:
            torch.Tensor: Loss of positive samples.
        """
        # bag_prob = Mean-max(matched_prob)
        matched_prob = matched_cls_prob * matched_box_prob
        weight = 1 / torch.clamp(1 - matched_prob, 1e-12, None)
        weight /= weight.sum(dim=1).unsqueeze(dim=-1)
        bag_prob = (weight * matched_prob).sum(dim=1)
        # positive_bag_loss = -self.alpha * log(bag_prob)
        bag_prob = bag_prob.clamp(0, 1)  # to avoid bug of BCE, check
        return self.alpha * F.binary_cross_entropy(
            bag_prob, torch.ones_like(bag_prob), reduction='none')

    def negative_bag_loss(self, cls_prob: Tensor, box_prob: Tensor) -> Tensor:
        """Generate negative bag loss.

        Args:
            cls_prob (torch.Tensor): Classification probability
                of negative samples.
            box_prob (torch.Tensor): Bounding box probability
                of negative samples.

        Returns:
            torch.Tensor: Loss of negative samples.
        """
        prob = cls_prob * (1 - box_prob)
        prob = prob.clamp(0, 1)  # to avoid bug of BCE, check
        negative_bag_loss = prob**self.gamma * F.binary_cross_entropy(
            prob, torch.zeros_like(prob), reduction='none')
        return (1 - self.alpha) * negative_bag_loss
