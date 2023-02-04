# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple

from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from torch import Tensor

from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import InstanceList, OptMultiConfig


class BaseMono3DDenseHead(BaseModule, metaclass=ABCMeta):
    """Base class for Monocular 3D DenseHeads.

    1. The ``loss`` method is used to calculate the loss of densehead,
    which includes two steps: (1) the densehead model performs forward
    propagation to obtain the feature maps (2) The ``loss_by_feat`` method
    is called based on the feature maps to calculate the loss.

    .. code:: text

    loss(): forward() -> loss_by_feat()

    2. The ``predict`` method is used to predict detection results,
    which includes two steps: (1) the densehead model performs forward
    propagation to obtain the feature maps (2) The ``predict_by_feat`` method
    is called based on the feature maps to predict detection results including
    post-processing.

    .. code:: text

    predict(): forward() -> predict_by_feat()

    3. The ``loss_and_predict`` method is used to return loss and detection
    results at the same time. It will call densehead's ``forward``,
    ``loss_by_feat`` and ``predict_by_feat`` methods in order.  If one-stage is
    used as RPN, the densehead needs to return both losses and predictions.
    This predictions is used as the proposal of roihead.

    .. code:: text

    loss_and_predict(): forward() -> loss_by_feat() -> predict_by_feat()
    """

    def __init__(self, init_cfg: OptMultiConfig = None) -> None:
        super(BaseMono3DDenseHead, self).__init__(init_cfg=init_cfg)

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             **kwargs) -> dict:
        """
        Args:
            x (list[Tensor]): Features from FPN.
            batch_data_samples (list[:obj:`Det3DDataSample`]): Each item
                contains the meta information of each image and corresponding
                annotations.

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
        batch_gt_instances = []
        batch_gt_instances_ignore = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
            batch_gt_instances.append(data_sample.gt_instances)
            batch_gt_instances_ignore.append(
                data_sample.get('ignored_instances', None))

        loss_inputs = outs + (batch_gt_instances_3d, batch_gt_instances,
                              batch_img_metas, batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)

        return losses

    @abstractmethod
    def loss_by_feat(self, **kwargs) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head."""
        pass

    def loss_and_predict(self,
                         x: Tuple[Tensor],
                         batch_data_samples: SampleList,
                         proposal_cfg: Optional[ConfigDict] = None,
                         **kwargs) -> Tuple[dict, InstanceList]:
        """Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples.

        Args:
            x (tuple[Tensor]): Features from FPN.
            batch_data_samples (list[:obj:`Det3DDataSample`]): Each item
                contains the meta information of each image and
                corresponding annotations.
            proposal_cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.

        Returns:
            tuple: the return value is a tuple contains:

            - losses: (dict[str, Tensor]): A dictionary of loss components.
            - predictions (list[:obj:`InstanceData`]): Detection
                results of each image after the post process.
        """
        batch_gt_instances_3d = []
        batch_gt_instances = []
        batch_gt_instances_ignore = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
            batch_gt_instances.append(data_sample.gt_instances)
            batch_gt_instances_ignore.append(
                data_sample.get('ignored_instances', None))

        outs = self(x)

        loss_inputs = outs + (batch_gt_instances_3d, batch_gt_instances,
                              batch_img_metas, batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, cfg=proposal_cfg)

        return losses, predictions

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_pts_panoptic_seg` and `gt_pts_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs = self(x)
        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)

        return predictions

    @abstractmethod
    def predict_by_feat(self, **kwargs) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results."""
        pass
