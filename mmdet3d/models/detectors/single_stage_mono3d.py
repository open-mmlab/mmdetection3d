# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

from torch import Tensor

from mmdet3d.core import Det3DDataSample, InstanceList
from mmdet3d.core.utils import SampleList
from mmdet3d.registry import MODELS
from mmdet.models.detectors.single_stage import SingleStageDetector


@MODELS.register_module()
class SingleStageMono3DDetector(SingleStageDetector):
    """Base class for monocular 3D single-stage detectors.

    Monocular 3D single-stage detectors directly and densely predict bounding
    boxes on the output features of the backbone+neck.
    """

    def convert_to_datasample(self, results_list: InstanceList) -> SampleList:
        """ Convert results list to `Det3DDataSample`.
        Args:
            results_list (list[:obj:`InstanceData`]):Detection results
            of each image. For each image, it could contains two results
            format:
                1. pred_instances_3d
                2. (pred_instances_3d, pred_instances)

        Returns:
            list[:obj:`Det3DDataSample`]: 3D Detection results of the
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
        out_results_list = []
        for i in range(len(results_list)):
            result = Det3DDataSample()
            if len(results_list[i]) == 2:
                result.pred_instances_3d = results_list[i][0]
                result.pred_instances = results_list[i][1]
            else:
                result.pred_instances_3d = results_list[i]
            out_results_list.append(result)
        return out_results_list

    def extract_feat(self, batch_inputs_dict: dict) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs_dict (dict): Contains 'img' key
                with image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        batch_imgs = batch_inputs_dict['imgs']
        x = self.backbone(batch_imgs)
        if self.with_neck:
            x = self.neck(x)
        return x

    # TODO: Support test time augmentation
    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation."""
        pass
