# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

from mmdet.models.task_modules import AssignResult, SamplingResult
from mmengine import ConfigDict
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.registry import MODELS
from mmdet3d.structures import bbox3d2roi
from mmdet3d.utils import InstanceList
from ...structures.det3d_data_sample import SampleList
from .base_3droi_head import Base3DRoIHead


@MODELS.register_module()
class PartAggregationROIHead(Base3DRoIHead):
    """Part aggregation roi head for PartA2.

    Args:
        semantic_head (ConfigDict): Config of semantic head.
        num_classes (int): The number of classes.
        seg_roi_extractor (ConfigDict): Config of seg_roi_extractor.
        bbox_roi_extractor (ConfigDict): Config of part_roi_extractor.
        bbox_head (ConfigDict): Config of bbox_head.
        train_cfg (ConfigDict): Training config.
        test_cfg (ConfigDict): Testing config.
    """

    def __init__(self,
                 semantic_head: dict,
                 num_classes: int = 3,
                 seg_roi_extractor: dict = None,
                 bbox_head: dict = None,
                 bbox_roi_extractor: dict = None,
                 train_cfg: dict = None,
                 test_cfg: dict = None,
                 init_cfg: dict = None) -> None:
        super(PartAggregationROIHead, self).__init__(
            bbox_head=bbox_head,
            bbox_roi_extractor=bbox_roi_extractor,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        self.num_classes = num_classes
        assert semantic_head is not None
        self.init_seg_head(seg_roi_extractor, semantic_head)

    def init_seg_head(self, seg_roi_extractor: dict,
                      semantic_head: dict) -> None:
        """Initialize semantic head and seg roi extractor.

        Args:
            seg_roi_extractor (dict): Config of seg
                roi extractor.
            semantic_head (dict): Config of semantic head.
        """
        self.semantic_head = MODELS.build(semantic_head)
        self.seg_roi_extractor = MODELS.build(seg_roi_extractor)

    @property
    def with_semantic(self):
        """bool: whether the head has semantic branch"""
        return hasattr(self,
                       'semantic_head') and self.semantic_head is not None

    def _bbox_forward_train(self, feats_dict: Dict, voxels_dict: Dict,
                            sampling_results: List[SamplingResult]) -> Dict:
        """Forward training function of roi_extractor and bbox_head.

        Args:
            feats_dict (dict): Contains features from the first stage.
            voxels_dict (dict): Contains information of voxels.
            sampling_results (:obj:`SamplingResult`): Sampled results used
                for training.

        Returns:
            dict: Forward results including losses and predictions.
        """
        rois = bbox3d2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(feats_dict, voxels_dict, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results,
                                                  self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _assign_and_sample(
            self, rpn_results_list: InstanceList,
            batch_gt_instances_3d: InstanceList,
            batch_gt_instances_ignore: InstanceList) -> List[SamplingResult]:
        """Assign and sample proposals for training.

        Args:
            rpn_results_list (List[:obj:`InstanceData`]): Detection results
                of rpn head.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances. It usually includes ``bboxes_3d`` and
                ``labels_3d`` attributes.
            batch_gt_instances_ignore (list): Ignore instances of gt bboxes.

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
            cur_gt_instances_ignore = batch_gt_instances_ignore[batch_idx]
            cur_gt_instances_3d.bboxes_3d = cur_gt_instances_3d.\
                bboxes_3d.tensor
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

    def _semantic_forward_train(self, feats_dict: dict, voxel_dict: dict,
                                batch_gt_instances_3d: InstanceList) -> Dict:
        """Train semantic head.

        Args:
            feats_dict (dict): Contains features from the first stage.
            voxel_dict (dict): Contains information of voxels.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances. It usually includes ``bboxes_3d`` and
                ``labels_3d`` attributes.

        Returns:
            dict: Segmentation results including losses
        """
        semantic_results = self.semantic_head(feats_dict['seg_features'])
        semantic_targets = self.semantic_head.get_targets(
            voxel_dict, batch_gt_instances_3d)
        loss_semantic = self.semantic_head.loss(semantic_results,
                                                semantic_targets)
        semantic_results.update(loss_semantic=loss_semantic)
        return semantic_results

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
        assert self.with_bbox, 'Bbox head must be implemented in PartA2.'
        assert self.with_semantic, 'Semantic head must be implemented' \
                                   ' in PartA2.'

        batch_input_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        voxels_dict = feats_dict.pop('voxels_dict')
        # TODO: Split predict semantic and bbox
        results_list = self.predict_bbox(feats_dict, voxels_dict,
                                         batch_input_metas, rpn_results_list,
                                         self.test_cfg)
        return results_list

    def predict_bbox(self, feats_dict: Dict, voxel_dict: Dict,
                     batch_input_metas: List[dict],
                     rpn_results_list: InstanceList,
                     test_cfg: ConfigDict) -> InstanceList:
        """Perform forward propagation of the bbox head and predict detection
        results on the features of the upstream network.

        Args:
            feats_dict (dict): Contains features from the first stage.
            voxel_dict (dict): Contains information of voxels.
            batch_input_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            rpn_results_list (List[:obj:`InstanceData`]): Detection results
                of rpn head.
            test_cfg (Config): Test config.

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
        semantic_results = self.semantic_head(feats_dict['seg_features'])
        feats_dict.update(semantic_results)
        rois = bbox3d2roi(
            [res['bboxes_3d'].tensor for res in rpn_results_list])
        labels_3d = [res['labels_3d'] for res in rpn_results_list]
        cls_preds = [res['cls_preds'] for res in rpn_results_list]
        bbox_results = self._bbox_forward(feats_dict, voxel_dict, rois)

        bbox_list = self.bbox_head.get_results(rois, bbox_results['cls_score'],
                                               bbox_results['bbox_pred'],
                                               labels_3d, cls_preds,
                                               batch_input_metas, test_cfg)
        return bbox_list

    def _bbox_forward(self, feats_dict: Dict, voxel_dict: Dict,
                      rois: Tensor) -> Dict:
        """Forward function of roi_extractor and bbox_head used in both
        training and testing.

        Args:
            feats_dict (dict): Contains features from the first stage.
            voxel_dict (dict): Contains information of voxels.
            rois (Tensor): Roi boxes.

        Returns:
            dict: Contains predictions of bbox_head and
                features of roi_extractor.
        """
        pooled_seg_feats = self.seg_roi_extractor(feats_dict['seg_features'],
                                                  voxel_dict['voxel_centers'],
                                                  voxel_dict['coors'][...,
                                                                      0], rois)
        pooled_part_feats = self.bbox_roi_extractor(
            feats_dict['part_feats'], voxel_dict['voxel_centers'],
            voxel_dict['coors'][..., 0], rois)
        cls_score, bbox_pred = self.bbox_head(pooled_seg_feats,
                                              pooled_part_feats)

        bbox_results = dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            pooled_seg_feats=pooled_seg_feats,
            pooled_part_feats=pooled_part_feats)
        return bbox_results

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
        assert len(rpn_results_list) == len(batch_data_samples)
        losses = dict()
        batch_gt_instances_3d = []
        batch_gt_instances_ignore = []
        voxels_dict = feats_dict.pop('voxels_dict')
        for data_sample in batch_data_samples:
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
            if 'ignored_instances' in data_sample:
                batch_gt_instances_ignore.append(data_sample.ignored_instances)
            else:
                batch_gt_instances_ignore.append(None)
        if self.with_semantic:
            semantic_results = self._semantic_forward_train(
                feats_dict, voxels_dict, batch_gt_instances_3d)
            losses.update(semantic_results.pop('loss_semantic'))

        sample_results = self._assign_and_sample(rpn_results_list,
                                                 batch_gt_instances_3d,
                                                 batch_gt_instances_ignore)
        if self.with_bbox:
            feats_dict.update(semantic_results)
            bbox_results = self._bbox_forward_train(feats_dict, voxels_dict,
                                                    sample_results)
            losses.update(bbox_results['loss_bbox'])

        return losses

    def _forward(self, feats_dict: dict,
                 rpn_results_list: InstanceList) -> Tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            feats_dict (dict): Contains features from the first stage.
            rpn_results_list (List[:obj:`InstanceData`]): Detection results
                of rpn head.

        Returns:
            tuple: A tuple of results from roi head.
        """
        voxel_dict = feats_dict.pop('voxel_dict')
        semantic_results = self.semantic_head(feats_dict['seg_features'])
        feats_dict.update(semantic_results)
        rois = bbox3d2roi([res['bbox_3d'].tensor for res in rpn_results_list])
        bbox_results = self._bbox_forward(feats_dict, voxel_dict, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        return cls_score, bbox_pred
