# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.models.task_modules import AssignResult
from mmengine.structures import InstanceData

from mmdet3d.registry import TASK_UTILS
from ..samplers import BaseSampler, SamplingResult


@TASK_UTILS.register_module()
class PseudoSampler(BaseSampler):
    """A pseudo sampler that does not do sampling actually."""

    # TODO: This is a temporary pseudo sampler.

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        """Sample positive samples."""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative samples."""
        raise NotImplementedError

    def sample(self, assign_result: AssignResult, pred_instances: InstanceData,
               gt_instances: InstanceData, *args, **kwargs):
        """Directly returns the positive and negative indices  of samples.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            pred_instances (:obj:`InstaceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors, points, or bboxes predicted by the model,
                shape(n, 4).
            gt_instances (:obj:`InstaceData`): Ground truth of instance
                annotations. It usually includes ``bboxes`` and ``labels``
                attributes.

        Returns:
            :obj:`SamplingResult`: sampler results
        """
        gt_bboxes = gt_instances.bboxes_3d
        priors = pred_instances.priors

        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()

        gt_flags = priors.new_zeros(priors.shape[0], dtype=torch.uint8)
        sampling_result = SamplingResult(
            pos_inds=pos_inds,
            neg_inds=neg_inds,
            priors=priors,
            gt_bboxes=gt_bboxes,
            assign_result=assign_result,
            gt_flags=gt_flags,
            avg_factor_with_neg=False)
        return sampling_result
