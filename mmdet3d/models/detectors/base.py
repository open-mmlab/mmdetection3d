# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
from mmengine.data import InstanceData
from torch.optim import Optimizer

from mmdet3d.core import Det3DDataSample
from mmdet3d.registry import MODELS
from mmdet.core.utils import stack_batch
from mmdet.models.detectors import BaseDetector


@MODELS.register_module()
class Base3DDetector(BaseDetector):
    """Base class for 3D detectors.

    Args:
        preprocess_cfg (dict, optional): Model preprocessing config
            for processing the input data. it usually includes
            ``to_rgb``, ``pad_size_divisor``, ``pad_value``,
            ``mean`` and ``std``. Default to None.
       init_cfg (dict, optional): the config to control the
           initialization. Default to None.
    """

    def __init__(self,
                 preprocess_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None) -> None:
        super(Base3DDetector, self).__init__(
            preprocess_cfg=preprocess_cfg, init_cfg=init_cfg)

    def forward_simple_test(self, batch_inputs_dict: Dict[List, torch.Tensor],
                            batch_data_samples: List[Det3DDataSample],
                            **kwargs) -> List[Det3DDataSample]:
        """
        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'img' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            list(obj:`Det3DDataSample`): Detection results of the
            input images. Each DetDataSample usually contains
            ``pred_instances_3d`` or ``pred_panoptic_seg_3d`` or
            ``pred_sem_seg_3d``.
        """
        batch_size = len(batch_data_samples)
        batch_input_metas = []
        if batch_size != len(batch_inputs_dict['points']):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(batch_inputs_dict['points']), len(batch_input_metas)))

        for batch_index in range(batch_size):
            metainfo = batch_data_samples[batch_index].metainfo
            batch_input_metas.append(metainfo)
        for var, name in [(batch_inputs_dict['points'], 'points'),
                          (batch_input_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        if batch_size == 1:
            return self.simple_test(
                batch_inputs_dict, batch_input_metas, rescale=True, **kwargs)
        else:
            return self.aug_test(
                batch_inputs_dict, batch_input_metas, rescale=True, **kwargs)

    def forward(self,
                data: List[dict],
                optimizer: Optional[Union[Optimizer, dict]] = None,
                return_loss: bool = False,
                **kwargs):
        """The iteration step during training and testing. This method defines
        an iteration step during training and testing, except for the back
        propagation and optimizer updating during training, which are done in
        an optimizer scheduler.

        Args:
            data (list[dict]): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer`, dict, Optional): The
                optimizer of runner. This argument is unused and reserved.
                Default to None.
            return_loss (bool): Whether to return loss. In general,
                it will be set to True during training and False
                during testing. Default to False.

        Returns:
            during training
                dict: It should contain at least 3 keys: ``loss``,
                ``log_vars``, ``num_samples``.

                    - ``loss`` is a tensor for back propagation, which can be a
                      weighted sum of multiple losses.
                    - ``log_vars`` contains all the variables to be sent to the
                        logger.
                    - ``num_samples`` indicates the batch size (when the model
                        is DDP, it means the batch size on each GPU), which is
                        used for averaging the logs.

            during testing
                list(obj:`Det3DDataSample`): Detection results of the
                input samples. Each DetDataSample usually contains
                ``pred_instances_3d`` or ``pred_panoptic_seg_3d`` or
                ``pred_sem_seg_3d``.
        """

        batch_inputs_dict, batch_data_samples = self.preprocess_data(data)
        if return_loss:
            losses = self.forward_train(batch_inputs_dict, batch_data_samples,
                                        **kwargs)
            loss, log_vars = self._parse_losses(losses)

            outputs = dict(
                loss=loss,
                log_vars=log_vars,
                num_samples=len(batch_data_samples))
            return outputs
        else:
            return self.forward_simple_test(batch_inputs_dict,
                                            batch_data_samples, **kwargs)

    def preprocess_data(self, data: List[dict]) -> tuple:
        """ Process input data during training and simple testing phases.
        Args:
            data (list[dict]): The data to be processed, which
                comes from dataloader.

        Returns:
            tuple:  It should contain 2 item.

                 - batch_inputs_dict (dict): The model input dict which include
                    'points', 'img' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.

                 - batch_data_samples (list[:obj:`Det3DDataSample`]): The Data
                     Samples. It usually includes information such as
                    `gt_instance_3d` , `gt_instances`.
        """
        batch_data_samples = [
            data_['data_sample'].to(self.device) for data_ in data
        ]
        if 'points' in data[0]['inputs'].keys():
            points = [
                data_['inputs']['points'].to(self.device) for data_ in data
            ]
        else:
            raise KeyError(
                "Model input dict needs to include the 'points' key.")
        if 'img' in data[0]['inputs'].keys():
            imgs = [data_['inputs']['img'].to(self.device) for data_ in data]
        else:
            imgs = None
        if self.preprocess_cfg is None:
            batch_inputs_dict = {
                'points': points,
                'imgs': stack_batch(imgs).float() if imgs is not None else None
            }
            return batch_inputs_dict, batch_data_samples

        if self.to_rgb and imgs[0].size(0) == 3:
            imgs = [_img[[2, 1, 0], ...] for _img in imgs]
        imgs = [(_img - self.pixel_mean) / self.pixel_std for _img in imgs]
        batch_img = stack_batch(imgs, self.pad_size_divisor, self.pad_value)
        batch_inputs_dict = {'points': points, 'imgs': batch_img}
        return batch_inputs_dict, batch_data_samples

    def postprocess_result(self, results_list: List[InstanceData]) \
            -> List[Det3DDataSample]:
        """ Convert results list to `Det3DDataSample`.
        Args:
            results_list (list[:obj:`InstanceData`]): Detection results of
                each sample.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3dd`` usually
            contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                    (num_instances, )
                - labels_3d (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                    contains a tensor with shape (num_instances, 7).
            """
        for i in range(len(results_list)):
            result = Det3DDataSample()
            result.pred_instances_3d = results_list[i]
            results_list[i] = result
        return results_list

    def show_results(self, data, result, out_dir, show=False, score_thr=None):
        # TODO
        pass
