# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from abc import ABCMeta, abstractmethod
from typing import List, Optional

from mmcv.runner import BaseModule
from mmengine.config import ConfigDict
from torch import Tensor

from mmdet3d.core import Det3DDataSample


class BaseMono3DDenseHead(BaseModule, metaclass=ABCMeta):
    """Base class for Monocular 3D DenseHeads."""

    def __init__(self, init_cfg: Optional[dict] = None) -> None:
        super(BaseMono3DDenseHead, self).__init__(init_cfg=init_cfg)

    @abstractmethod
    def loss(self, **kwargs):
        """Compute losses of the head."""
        pass

    def get_bboxes(self, *args, **kwargs):
        warnings.warn('`get_bboxes` is deprecated and will be removed in '
                      'the future. Please use `get_results` instead.')
        return self.get_results(*args, **kwargs)

    @abstractmethod
    def get_results(self, *args, **kwargs):
        """Transform network outputs of a batch into 3D bbox results."""
        pass

    def forward_train(self,
                      x: List[Tensor],
                      batch_data_samples: List[Det3DDataSample],
                      proposal_cfg: Optional[ConfigDict] = None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            batch_data_samples (list[:obj:`Det3DDataSample`]): Each item
                contains the meta information of each image and corresponding
                annotations.
            proposal_cfg (mmengine.Config, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.

        Returns:
            tuple or Tensor: When `proposal_cfg` is None, the detector is a \
            normal one-stage detector, The return value is the losses.

            - losses: (dict[str, Tensor]): A dictionary of loss components.

            When the `proposal_cfg` is not None, the head is used as a
            `rpn_head`, the return value is a tuple contains:

            - losses: (dict[str, Tensor]): A dictionary of loss components.
            - results_list (list[:obj:`InstanceData`]): Detection
              results of each image after the post process.
              Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (:obj:`BaseInstance3DBoxes`): Contains a tensor
                  with shape (num_instances, C), the last dimension C of a
                  3D box is (x, y, z, x_size, y_size, z_size, yaw, ...), where
                  C >= 7. C = 7 for kitti and C = 9 for nuscenes with extra 2
                  dims of velocity.
        """

        outs = self(x)
        batch_gt_instances_3d = []
        batch_gt_instances_ignore = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
            if 'ignored_instances' in data_sample:
                batch_gt_instances_ignore.append(data_sample.ignored_instances)
            else:
                batch_gt_instances_ignore.append(None)

        loss_inputs = outs + (batch_gt_instances_3d, batch_img_metas,
                              batch_gt_instances_ignore)
        losses = self.loss(*loss_inputs)

        if proposal_cfg is None:
            return losses
        else:
            batch_img_metas = [
                data_sample.metainfo for data_sample in batch_data_samples
            ]
            results_list = self.get_results(
                *outs, batch_img_metas=batch_img_metas, cfg=proposal_cfg)
            return losses, results_list
