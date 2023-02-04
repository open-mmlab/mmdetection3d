# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple

import torch
from mmdet.models.utils import multi_apply
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor
from torch import nn as nn

from mmdet3d.models.layers import nms_bev, nms_normal_bev
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures import xywhr2xyxyr
from mmdet3d.structures.bbox_3d import (BaseInstance3DBoxes,
                                        DepthInstance3DBoxes,
                                        LiDARInstance3DBoxes)
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils.typing_utils import InstanceList


@MODELS.register_module()
class PointRPNHead(BaseModule):
    """RPN module for PointRCNN.

    Args:
        num_classes (int): Number of classes.
        train_cfg (dict): Train configs.
        test_cfg (dict): Test configs.
        pred_layer_cfg (dict, optional): Config of classification and
            regression prediction layers. Defaults to None.
        enlarge_width (float, optional): Enlarge bbox for each side to ignore
            close points. Defaults to 0.1.
        cls_loss (dict, optional): Config of direction classification loss.
            Defaults to None.
        bbox_loss (dict, optional): Config of localization loss.
            Defaults to None.
        bbox_coder (dict, optional): Config dict of box coders.
            Defaults to None.
        init_cfg (dict, optional): Config of initialization. Defaults to None.
    """

    def __init__(self,
                 num_classes: int,
                 train_cfg: dict,
                 test_cfg: dict,
                 pred_layer_cfg: Optional[dict] = None,
                 enlarge_width: float = 0.1,
                 cls_loss: Optional[dict] = None,
                 bbox_loss: Optional[dict] = None,
                 bbox_coder: Optional[dict] = None,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.enlarge_width = enlarge_width

        # build loss function
        self.bbox_loss = MODELS.build(bbox_loss)
        self.cls_loss = MODELS.build(cls_loss)

        # build box coder
        self.bbox_coder = TASK_UTILS.build(bbox_coder)

        # build pred conv
        self.cls_layers = self._make_fc_layers(
            fc_cfg=pred_layer_cfg.cls_linear_channels,
            input_channels=pred_layer_cfg.in_channels,
            output_channels=self._get_cls_out_channels())

        self.reg_layers = self._make_fc_layers(
            fc_cfg=pred_layer_cfg.reg_linear_channels,
            input_channels=pred_layer_cfg.in_channels,
            output_channels=self._get_reg_out_channels())

    def _make_fc_layers(self, fc_cfg: dict, input_channels: int,
                        output_channels: int) -> nn.Sequential:
        """Make fully connect layers.

        Args:
            fc_cfg (dict): Config of fully connect.
            input_channels (int): Input channels for fc_layers.
            output_channels (int): Input channels for fc_layers.

        Returns:
            nn.Sequential: Fully connect layers.
        """
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
        # torch.cos(yaw) (1), torch.sin(yaw) (1)
        return self.bbox_coder.code_size

    def forward(self, feat_dict: dict) -> Tuple[List[Tensor]]:
        """Forward pass.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            tuple[list[torch.Tensor]]: Predicted boxes and classification
                scores.
        """
        point_features = feat_dict['fp_features']
        point_features = point_features.permute(0, 2, 1).contiguous()
        batch_size = point_features.shape[0]
        feat_cls = point_features.view(-1, point_features.shape[-1])
        feat_reg = point_features.view(-1, point_features.shape[-1])

        point_cls_preds = self.cls_layers(feat_cls).reshape(
            batch_size, -1, self._get_cls_out_channels())
        point_box_preds = self.reg_layers(feat_reg).reshape(
            batch_size, -1, self._get_reg_out_channels())
        return point_box_preds, point_cls_preds

    def loss_by_feat(
            self,
            bbox_preds: List[Tensor],
            cls_preds: List[Tensor],
            points: List[Tensor],
            batch_gt_instances_3d: InstanceList,
            batch_input_metas: Optional[List[dict]] = None,
            batch_gt_instances_ignore: Optional[InstanceList] = None) -> Dict:
        """Compute loss.

        Args:
            bbox_preds (list[torch.Tensor]): Predictions from forward of
                PointRCNN RPN_Head.
            cls_preds (list[torch.Tensor]): Classification from forward of
                PointRCNN RPN_Head.
            points (list[torch.Tensor]): Input points.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances_3d. It usually includes ``bboxes_3d`` and
                ``labels_3d`` attributes.
            batch_input_metas (list[dict]): Contain pcd and img's meta info.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict: Losses of PointRCNN RPN module.
        """
        targets = self.get_targets(points, batch_gt_instances_3d)
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
        # for ignore, but now we do not have ignored label
        semantic_loss_weight = negative_mask.float() + positive_mask.float()
        semantic_loss = self.cls_loss(semantic_points,
                                      semantic_points_label.reshape(-1),
                                      semantic_loss_weight.reshape(-1))
        semantic_loss /= positive_mask.float().sum()
        losses = dict(bbox_loss=bbox_loss, semantic_loss=semantic_loss)

        return losses

    def get_targets(self, points: List[Tensor],
                    batch_gt_instances_3d: InstanceList) -> Tuple[Tensor]:
        """Generate targets of PointRCNN RPN head.

        Args:
            points (list[torch.Tensor]): Points in one batch.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances_3d. It usually includes ``bboxes_3d`` and
                ``labels_3d`` attributes.

        Returns:
            tuple[torch.Tensor]: Targets of PointRCNN RPN head.
        """
        gt_labels_3d = [
            instances.labels_3d for instances in batch_gt_instances_3d
        ]
        gt_bboxes_3d = [
            instances.bboxes_3d for instances in batch_gt_instances_3d
        ]

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

    def get_targets_single(self, points: Tensor,
                           gt_bboxes_3d: BaseInstance3DBoxes,
                           gt_labels_3d: Tensor) -> Tuple[Tensor]:
        """Generate targets of PointRCNN RPN head for single batch.

        Args:
            points (torch.Tensor): Points of each batch.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): Ground truth
                boxes of each batch.
            gt_labels_3d (torch.Tensor): Labels of each batch.

        Returns:
            tuple[torch.Tensor]: Targets of ssd3d head.
        """
        gt_bboxes_3d = gt_bboxes_3d.to(points.device)

        valid_gt = gt_labels_3d != -1
        gt_bboxes_3d = gt_bboxes_3d[valid_gt]
        gt_labels_3d = gt_labels_3d[valid_gt]

        # transform the bbox coordinate to the point cloud coordinate
        gt_bboxes_3d_tensor = gt_bboxes_3d.tensor.clone()
        gt_bboxes_3d_tensor[..., 2] += gt_bboxes_3d_tensor[..., 5] / 2

        points_mask, assignment = self._assign_targets_by_points_inside(
            gt_bboxes_3d, points)
        gt_bboxes_3d_tensor = gt_bboxes_3d_tensor[assignment]
        mask_targets = gt_labels_3d[assignment]

        bbox_targets = self.bbox_coder.encode(gt_bboxes_3d_tensor,
                                              points[..., 0:3], mask_targets)

        positive_mask = (points_mask.max(1)[0] > 0)
        # add ignore_mask
        extend_gt_bboxes_3d = gt_bboxes_3d.enlarged_box(self.enlarge_width)
        points_mask, _ = self._assign_targets_by_points_inside(
            extend_gt_bboxes_3d, points)
        negative_mask = (points_mask.max(1)[0] == 0)

        point_targets = points[..., 0:3]
        return (bbox_targets, mask_targets, positive_mask, negative_mask,
                point_targets)

    def predict_by_feat(self, points: Tensor, bbox_preds: List[Tensor],
                        cls_preds: List[Tensor], batch_input_metas: List[dict],
                        cfg: Optional[dict]) -> InstanceList:
        """Generate bboxes from RPN head predictions.

        Args:
            points (torch.Tensor): Input points.
            bbox_preds (list[tensor]): Regression predictions from PointRCNN
                head.
            cls_preds (list[tensor]): Class scores predictions from PointRCNN
                head.
            batch_input_metas (list[dict]): Batch inputs meta info.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each sample
            after the post process.
            Each item usually contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (BaseInstance3DBoxes): Prediction of bboxes,
              contains a tensor with shape (num_instances, C), where
              C >= 7.
            - cls_preds (torch.Tensor): Class score of each bbox.
        """
        sem_scores = cls_preds.sigmoid()
        obj_scores = sem_scores.max(-1)[0]
        object_class = sem_scores.argmax(dim=-1)

        batch_size = sem_scores.shape[0]
        results = list()
        for b in range(batch_size):
            bbox3d = self.bbox_coder.decode(bbox_preds[b], points[b, ..., :3],
                                            object_class[b])
            mask = ~bbox3d.sum(dim=1).isinf()
            bbox_selected, score_selected, labels, cls_preds_selected = \
                self.class_agnostic_nms(obj_scores[b][mask],
                                        sem_scores[b][mask, :],
                                        bbox3d[mask, :],
                                        points[b, ..., :3][mask, :],
                                        batch_input_metas[b],
                                        cfg.nms_cfg)
            bbox_selected = batch_input_metas[b]['box_type_3d'](
                bbox_selected, box_dim=bbox_selected.shape[-1])
            result = InstanceData()
            result.bboxes_3d = bbox_selected
            result.scores_3d = score_selected
            result.labels_3d = labels
            result.cls_preds = cls_preds_selected
            results.append(result)
        return results

    def class_agnostic_nms(self, obj_scores: Tensor, sem_scores: Tensor,
                           bbox: Tensor, points: Tensor, input_meta: Dict,
                           nms_cfg: Dict) -> Tuple[Tensor]:
        """Class agnostic nms.

        Args:
            obj_scores (torch.Tensor): Objectness score of bounding boxes.
            sem_scores (torch.Tensor): Semantic class score of bounding boxes.
            bbox (torch.Tensor): Predicted bounding boxes.
            points (torch.Tensor): Input points.
            input_meta (dict): Contain pcd and img's meta info.
            nms_cfg (dict): NMS config dict.

        Returns:
            tuple[torch.Tensor]: Bounding boxes, scores and labels.
        """
        if nms_cfg.use_rotate_nms:
            nms_func = nms_bev
        else:
            nms_func = nms_normal_bev

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

        bbox = bbox[nonempty_box_mask]

        if nms_cfg.score_thr is not None:
            score_thr = nms_cfg.score_thr
            keep = (obj_scores >= score_thr)
            obj_scores = obj_scores[keep]
            sem_scores = sem_scores[keep]
            bbox = bbox.tensor[keep]

        if bbox.tensor.shape[0] > 0:
            topk = min(nms_cfg.nms_pre, obj_scores.shape[0])
            obj_scores_nms, indices = torch.topk(obj_scores, k=topk)
            bbox_for_nms = xywhr2xyxyr(bbox[indices].bev)
            sem_scores_nms = sem_scores[indices]

            keep = nms_func(bbox_for_nms, obj_scores_nms, nms_cfg.iou_thr)
            keep = keep[:nms_cfg.nms_post]

            bbox_selected = bbox.tensor[indices][keep]
            score_selected = obj_scores_nms[keep]
            cls_preds = sem_scores_nms[keep]
            labels = torch.argmax(cls_preds, -1)
            if bbox_selected.shape[0] > nms_cfg.nms_post:
                _, inds = score_selected.sort(descending=True)
                inds = inds[:score_selected.nms_post]
                bbox_selected = bbox_selected[inds, :]
                labels = labels[inds]
                score_selected = score_selected[inds]
                cls_preds = cls_preds[inds, :]
        else:
            bbox_selected = bbox.tensor
            score_selected = obj_scores.new_zeros([0])
            labels = obj_scores.new_zeros([0])
            cls_preds = obj_scores.new_zeros([0, sem_scores.shape[-1]])
        return bbox_selected, score_selected, labels, cls_preds

    def _assign_targets_by_points_inside(self, bboxes_3d: BaseInstance3DBoxes,
                                         points: Tensor) -> Tuple[Tensor]:
        """Compute assignment by checking whether point is inside bbox.

        Args:
            bboxes_3d (:obj:`BaseInstance3DBoxes`): Instance of bounding boxes.
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

    def predict(self, feats_dict: Dict,
                batch_data_samples: SampleList) -> InstanceList:
        """Perform forward propagation of the 3D detection head and predict
        detection results on the features of the upstream network.

        Args:
            feats_dict (dict): Contains features from the first stage.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each sample
            after the post process.
            Each item usually contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (BaseInstance3DBoxes): Prediction of bboxes,
              contains a tensor with shape (num_instances, C), where
              C >= 7.
        """
        batch_input_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        raw_points = feats_dict.pop('raw_points')
        bbox_preds, cls_preds = self(feats_dict)
        proposal_cfg = self.test_cfg

        proposal_list = self.predict_by_feat(
            raw_points,
            bbox_preds,
            cls_preds,
            cfg=proposal_cfg,
            batch_input_metas=batch_input_metas)
        feats_dict['points_cls_preds'] = cls_preds
        return proposal_list

    def loss_and_predict(self,
                         feats_dict: Dict,
                         batch_data_samples: SampleList,
                         proposal_cfg: Optional[dict] = None,
                         **kwargs) -> Tuple[dict, InstanceList]:
        """Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples.

        Args:
            feats_dict (dict): Contains features from the first stage.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.
            proposal_cfg (ConfigDict, optional): Proposal config.

        Returns:
            tuple: the return value is a tuple contains:

            - losses: (dict[str, Tensor]): A dictionary of loss components.
            - predictions (list[:obj:`InstanceData`]): Detection
              results of each sample after the post process.
        """
        batch_gt_instances_3d = []
        batch_gt_instances_ignore = []
        batch_input_metas = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
            batch_gt_instances_ignore.append(
                data_sample.get('ignored_instances', None))
        raw_points = feats_dict.pop('raw_points')
        bbox_preds, cls_preds = self(feats_dict)

        loss_inputs = (bbox_preds, cls_preds,
                       raw_points) + (batch_gt_instances_3d, batch_input_metas,
                                      batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)

        predictions = self.predict_by_feat(
            raw_points,
            bbox_preds,
            cls_preds,
            batch_input_metas=batch_input_metas,
            cfg=proposal_cfg)
        feats_dict['points_cls_preds'] = cls_preds
        if predictions[0].bboxes_3d.tensor.isinf().any():
            print(predictions)
        return losses, predictions
