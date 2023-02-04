# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

from mmengine.structures import InstanceData
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from ..test_time_augs import merge_aug_bboxes_3d
from .single_stage import SingleStage3DDetector


@MODELS.register_module()
class VoteNet(SingleStage3DDetector):
    r"""`VoteNet <https://arxiv.org/pdf/1904.09664.pdf>`_ for 3D detection.

    Args:
        backbone (dict): Config dict of detector's backbone.
        bbox_head (dict, optional): Config dict of box head. Defaults to None.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        init_cfg (dict, optional): the config to control the
           initialization. Default to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
            ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
    """

    def __init__(self,
                 backbone: dict,
                 bbox_head: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 **kwargs):
        super(VoteNet, self).__init__(
            backbone=backbone,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor,
            **kwargs)

    def loss(self, batch_inputs_dict: Dict[str, Union[List, Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        """
        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        feat_dict = self.extract_feat(batch_inputs_dict)
        points = batch_inputs_dict['points']
        losses = self.bbox_head.loss(points, feat_dict, batch_data_samples,
                                     **kwargs)
        return losses

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                    (num_instances, )
                - labels_3d (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                    contains a tensor with shape (num_instances, 7).
        """
        feats_dict = self.extract_feat(batch_inputs_dict)
        points = batch_inputs_dict['points']
        results_list = self.bbox_head.predict(points, feats_dict,
                                              batch_data_samples, **kwargs)
        data_3d_samples = self.add_pred_to_datasample(batch_data_samples,
                                                      results_list)
        return data_3d_samples

    def aug_test(self, aug_inputs_list: List[dict],
                 aug_data_samples: List[List[dict]], **kwargs):
        """Test with augmentation.

        Batch size always is 1 when do the augtest.

        Args:
            aug_inputs_list (List[dict]): The list indicate same data
                under differecnt augmentation.
            aug_data_samples (List[List[dict]]): The outer list
                indicate different augmentation, and the inter
                list indicate the batch size.
        """
        num_augs = len(aug_inputs_list)
        if num_augs == 1:
            return self.predict(aug_inputs_list[0], aug_data_samples[0])

        batch_size = len(aug_data_samples[0])
        assert batch_size == 1
        multi_aug_results = []
        for aug_id in range(num_augs):
            batch_inputs_dict = aug_inputs_list[aug_id]
            batch_data_samples = aug_data_samples[aug_id]
            feats_dict = self.extract_feat(batch_inputs_dict)
            points = batch_inputs_dict['points']
            results_list = self.bbox_head.predict(points, feats_dict,
                                                  batch_data_samples, **kwargs)
            multi_aug_results.append(results_list[0])
        aug_input_metas_list = []
        for aug_index in range(num_augs):
            metainfo = aug_data_samples[aug_id][0].metainfo
            aug_input_metas_list.append(metainfo)

        aug_results_list = [item.to_dict() for item in multi_aug_results]
        # after merging, bboxes will be rescaled to the original image size
        merged_results_dict = merge_aug_bboxes_3d(aug_results_list,
                                                  aug_input_metas_list,
                                                  self.bbox_head.test_cfg)

        merged_results = InstanceData(**merged_results_dict)
        data_3d_samples = self.add_pred_to_datasample(batch_data_samples,
                                                      [merged_results])
        return data_3d_samples
