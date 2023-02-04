# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple, Union

import torch
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from ...structures.det3d_data_sample import OptSampleList, SampleList
from .base import Base3DDetector


@MODELS.register_module()
class SingleStage3DDetector(Base3DDetector):
    """SingleStage3DDetector.

    This class serves as a base class for single-stage 3D detectors which
    directly and densely predict 3D bounding boxes on the output features
    of the backbone+neck.


    Args:
        backbone (dict): Config dict of detector's backbone.
        neck (dict, optional): Config dict of neck. Defaults to None.
        bbox_head (dict, optional): Config dict of box head. Defaults to None.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> Union[dict, list]:
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'img' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs_dict)
        losses = self.bbox_head.loss(x, batch_data_samples, **kwargs)
        return losses

    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'img' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample usually contain
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
        results_list = self.bbox_head.predict(x, batch_data_samples, **kwargs)
        predictions = self.add_pred_to_datasample(batch_data_samples,
                                                  results_list)
        return predictions

    def _forward(self,
                 batch_inputs_dict: dict,
                 data_samples: OptSampleList = None,
                 **kwargs) -> Tuple[List[torch.Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'img' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x = self.extract_feat(batch_inputs_dict)
        results = self.bbox_head.forward(x)
        return results

    def extract_feat(
        self, batch_inputs_dict: Dict[str, Tensor]
    ) -> Union[Tuple[torch.Tensor], Dict[str, Tensor]]:
        """Directly extract features from the backbone+neck.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'img' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.

        Returns:
            tuple[Tensor] | dict:  For outside 3D object detection, we
                typically obtain a tuple of features from the backbone + neck,
                and for inside 3D object detection, usually a dict containing
                features will be obtained.
        """
        points = batch_inputs_dict['points']
        stack_points = torch.stack(points)
        x = self.backbone(stack_points)
        if self.with_neck:
            x = self.neck(x)
        return x
