# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

import torch
from mmdet.models.task_modules import AssignResult
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures import bbox3d2roi
from mmdet3d.utils.typing import InstanceList, SampleList
from .base_3droi_head import Base3DRoIHead


@MODELS.register_module()
class PointRCNNRoIHead(Base3DRoIHead):
    """RoI head for PointRCNN.

    Args:
        bbox_head (dict): Config of bbox_head.
        bbox_roi_extractor (dict): Config of RoI extractor.
        train_cfg (dict): Train configs.
        test_cfg (dict): Test configs.
        depth_normalizer (float): Normalize depth feature.
            Defaults to 70.0.
        init_cfg (dict, optional): Config of initialization. Defaults to None.
    """

    def __init__(self,
                 bbox_head: dict,
                 bbox_roi_extractor: dict,
                 train_cfg: dict,
                 test_cfg: dict,
                 depth_normalizer: dict = 70.0,
                 init_cfg: Optional[dict] = None) -> None:
        super(PointRCNNRoIHead, self).__init__(
            bbox_head=bbox_head,
            bbox_roi_extractor=bbox_roi_extractor,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        self.depth_normalizer = depth_normalizer

        self.init_assigner_sampler()

    def init_mask_head(self):
        """Initialize maek head."""
        pass

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            if isinstance(self.train_cfg.assigner, dict):
                self.bbox_assigner = TASK_UTILS.build(self.train_cfg.assigner)
            elif isinstance(self.train_cfg.assigner, list):
                self.bbox_assigner = [
                    TASK_UTILS.build(res) for res in self.train_cfg.assigner
                ]
            self.bbox_sampler = TASK_UTILS.build(self.train_cfg.sampler)

    def loss(self, feats_dict: Dict, rpn_results_list: InstanceList,
             batch_data_samples: SampleList, **kwargs) -> dict:
        """Perform forward propagation and loss calculation of the detection
        roi on the features of the upstream network.

        Args:
            feats_dict (dict): Contains features from the first stage.
            rpn_results_list (List[:obj:`InstanceData`]): Detection results
                of rpn head.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        features = feats_dict['fp_features']
        fp_points = feats_dict['fp_points']
        point_cls_preds = feats_dict['points_cls_preds']
        sem_scores = point_cls_preds.sigmoid()
        point_scores = sem_scores.max(-1)[0]
        batch_gt_instances_3d = []
        batch_gt_instances_ignore = []
        for data_sample in batch_data_samples:
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
            if 'ignored_instances' in data_sample:
                batch_gt_instances_ignore.append(data_sample.ignored_instances)
            else:
                batch_gt_instances_ignore.append(None)
        sample_results = self._assign_and_sample(rpn_results_list,
                                                 batch_gt_instances_3d,
                                                 batch_gt_instances_ignore)

        # concat the depth, semantic features and backbone features
        features = features.transpose(1, 2).contiguous()
        point_depths = fp_points.norm(dim=2) / self.depth_normalizer - 0.5
        features_list = [
            point_scores.unsqueeze(2),
            point_depths.unsqueeze(2), features
        ]
        features = torch.cat(features_list, dim=2)

        bbox_results = self._bbox_forward_train(features, fp_points,
                                                sample_results)
        losses = dict()
        losses.update(bbox_results['loss_bbox'])

        return losses

    def predict(self,
                feats_dict: Dict,
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList,
                rescale: bool = False,
                **kwargs) -> InstanceList:
        """Perform forward propagation of the roi head and predict detection
        results on the features of the upstream network.

        Args:
            feats_dict (dict): Contains features from the first stage.
            rpn_results_list (List[:obj:`InstanceData`]): Detection results
                of rpn head.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

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
        rois = bbox3d2roi(
            [res['bboxes_3d'].tensor for res in rpn_results_list])
        labels_3d = [res['labels_3d'] for res in rpn_results_list]
        batch_input_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        fp_features = feats_dict['fp_features']
        fp_points = feats_dict['fp_points']
        point_cls_preds = feats_dict['points_cls_preds']
        sem_scores = point_cls_preds.sigmoid()
        point_scores = sem_scores.max(-1)[0]

        features = fp_features.transpose(1, 2).contiguous()
        point_depths = fp_points.norm(dim=2) / self.depth_normalizer - 0.5
        features_list = [
            point_scores.unsqueeze(2),
            point_depths.unsqueeze(2), features
        ]

        features = torch.cat(features_list, dim=2)
        batch_size = features.shape[0]
        bbox_results = self._bbox_forward(features, fp_points, batch_size,
                                          rois)
        object_score = bbox_results['cls_score'].sigmoid()
        bbox_list = self.bbox_head.get_results(
            rois,
            object_score,
            bbox_results['bbox_pred'],
            labels_3d,
            batch_input_metas,
            cfg=self.test_cfg)

        return bbox_list

    def _bbox_forward_train(self, features: Tensor, points: Tensor,
                            sampling_results: SampleList) -> dict:
        """Forward training function of roi_extractor and bbox_head.

        Args:
            features (torch.Tensor): Backbone features with depth and \
                semantic features.
            points (torch.Tensor): Point cloud.
            sampling_results (:obj:`SamplingResult`): Sampled results used
                for training.

        Returns:
            dict: Forward results including losses and predictions.
        """
        rois = bbox3d2roi([res.bboxes for res in sampling_results])
        batch_size = features.shape[0]
        bbox_results = self._bbox_forward(features, points, batch_size, rois)
        bbox_targets = self.bbox_head.get_targets(sampling_results,
                                                  self.train_cfg)

        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _bbox_forward(self, features: Tensor, points: Tensor, batch_size: int,
                      rois: Tensor) -> dict:
        """Forward function of roi_extractor and bbox_head used in both
        training and testing.

        Args:
            features (torch.Tensor): Backbone features with depth and
                semantic features.
            points (torch.Tensor): Point cloud.
            batch_size (int): Batch size.
            rois (torch.Tensor): RoI boxes.

        Returns:
            dict: Contains predictions of bbox_head and
                features of roi_extractor.
        """
        pooled_point_feats = self.bbox_roi_extractor(features, points,
                                                     batch_size, rois)

        cls_score, bbox_pred = self.bbox_head(pooled_point_feats)
        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred)
        return bbox_results

    def _assign_and_sample(
            self, rpn_results_list: InstanceList,
            batch_gt_instances_3d: InstanceList,
            batch_gt_instances_ignore: InstanceList) -> SampleList:
        """Assign and sample proposals for training.

        Args:
            rpn_results_list (List[:obj:`InstanceData`]): Detection results
                of rpn head.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances. It usually includes ``bboxes_3d`` and
                ``labels_3d`` attributes.
            batch_gt_instances_ignore (list[:obj:`InstanceData`]): Ignore
                instances of gt bboxes.

        Returns:
            list[:obj:`SamplingResult`]: Sampled results of each training
                sample.
        """
        sampling_results = []
        # bbox assign
        for batch_idx in range(len(rpn_results_list)):
            cur_proposal_list = rpn_results_list[batch_idx]
            cur_boxes = cur_proposal_list['bboxes_3d']
            cur_labels_3d = cur_proposal_list['labels_3d']
            cur_gt_instances_3d = batch_gt_instances_3d[batch_idx]
            cur_gt_instances_3d.bboxes_3d = cur_gt_instances_3d.\
                bboxes_3d.tensor
            cur_gt_instances_ignore = batch_gt_instances_ignore[batch_idx]
            cur_gt_bboxes = cur_gt_instances_3d.bboxes_3d.to(cur_boxes.device)
            cur_gt_labels = cur_gt_instances_3d.labels_3d
            batch_num_gts = 0
            # 0 is bg
            batch_gt_indis = cur_gt_labels.new_full((len(cur_boxes), ), 0)
            batch_max_overlaps = cur_boxes.tensor.new_zeros(len(cur_boxes))
            # -1 is bg
            batch_gt_labels = cur_gt_labels.new_full((len(cur_boxes), ), -1)

            # each class may have its own assigner
            if isinstance(self.bbox_assigner, list):
                for i, assigner in enumerate(self.bbox_assigner):
                    gt_per_cls = (cur_gt_labels == i)
                    pred_per_cls = (cur_labels_3d == i)
                    cur_assign_res = assigner.assign(
                        cur_proposal_list[pred_per_cls],
                        cur_gt_instances_3d[gt_per_cls],
                        cur_gt_instances_ignore)
                    # gather assign_results in different class into one result
                    batch_num_gts += cur_assign_res.num_gts
                    # gt inds (1-based)
                    gt_inds_arange_pad = gt_per_cls.nonzero(
                        as_tuple=False).view(-1) + 1
                    # pad 0 for indice unassigned
                    gt_inds_arange_pad = F.pad(
                        gt_inds_arange_pad, (1, 0), mode='constant', value=0)
                    # pad -1 for indice ignore
                    gt_inds_arange_pad = F.pad(
                        gt_inds_arange_pad, (1, 0), mode='constant', value=-1)
                    # convert to 0~gt_num+2 for indices
                    gt_inds_arange_pad += 1
                    # now 0 is bg, >1 is fg in batch_gt_indis
                    batch_gt_indis[pred_per_cls] = gt_inds_arange_pad[
                        cur_assign_res.gt_inds + 1] - 1
                    batch_max_overlaps[
                        pred_per_cls] = cur_assign_res.max_overlaps
                    batch_gt_labels[pred_per_cls] = cur_assign_res.labels

                assign_result = AssignResult(batch_num_gts, batch_gt_indis,
                                             batch_max_overlaps,
                                             batch_gt_labels)
            else:  # for single class
                assign_result = self.bbox_assigner.assign(
                    cur_proposal_list, cur_gt_instances_3d,
                    cur_gt_instances_ignore)

            # sample boxes
            sampling_result = self.bbox_sampler.sample(assign_result,
                                                       cur_boxes.tensor,
                                                       cur_gt_bboxes,
                                                       cur_gt_labels)
            sampling_results.append(sampling_result)
        return sampling_results
