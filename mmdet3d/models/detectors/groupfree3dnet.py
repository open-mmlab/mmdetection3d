# Copyright (c) OpenMMLab. All rights reserved.

from mmdet3d.registry import MODELS
from ...structures.det3d_data_sample import SampleList
from .single_stage import SingleStage3DDetector


@MODELS.register_module()
class GroupFree3DNet(SingleStage3DDetector):
    """`Group-Free 3D <https://arxiv.org/abs/2104.00678>`_."""

    def __init__(self,
                 backbone,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super(GroupFree3DNet, self).__init__(
            backbone=backbone,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            **kwargs)

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> dict:
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'imgs' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_pts_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs_dict)
        points = batch_inputs_dict['points']
        losses = self.bbox_head.loss(points, x, batch_data_samples, **kwargs)
        return losses

    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'imgs' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_pts_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input images. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instance, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bboxes_3d (Tensor): Contains a tensor with shape
                (num_instances, C) where C >=7.
        """
        x = self.extract_feat(batch_inputs_dict)
        points = batch_inputs_dict['points']
        results_list = self.bbox_head.predict(points, x, batch_data_samples,
                                              **kwargs)
        predictions = self.add_pred_to_datasample(batch_data_samples,
                                                  results_list)
        return predictions
