# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from ...structures.det3d_data_sample import SampleList
from .single_stage import SingleStage3DDetector


@MODELS.register_module()
class SASSD(SingleStage3DDetector):
    r"""`SASSD <https://github.com/skyhehe123/SA-SSD>` _ for 3D detection."""

    def __init__(self,
                 voxel_encoder: ConfigType,
                 middle_encoder: ConfigType,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super(SASSD, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        self.voxel_encoder = MODELS.build(voxel_encoder)
        self.middle_encoder = MODELS.build(middle_encoder)

    def extract_feat(
        self,
        batch_inputs_dict: dict,
        test_mode: bool = True
    ) -> Union[Tuple[Tuple[Tensor], Tuple], Tuple[Tensor]]:
        """Extract features from points.

        Args:
            batch_inputs_dict (dict): The batch inputs.
            test_mode (bool, optional): Whether test mode. Defaults to True.

        Returns:
            Union[Tuple[Tuple[Tensor], Tuple], Tuple[Tensor]]: In test mode, it
            returns the features of points from multiple levels. In training
            mode, it returns the features of points from multiple levels and a
            tuple containing the mean features of points and the targets of
            clssification and regression.
        """
        voxel_dict = batch_inputs_dict['voxels']
        voxel_features = self.voxel_encoder(voxel_dict['voxels'],
                                            voxel_dict['num_points'],
                                            voxel_dict['coors'])
        batch_size = voxel_dict['coors'][-1, 0].item() + 1
        # `point_misc` is a tuple containing the mean features of points and
        # the targets of clssification and regression. It's only used for
        # calculating auxiliary loss in training mode.
        x, point_misc = self.middle_encoder(voxel_features,
                                            voxel_dict['coors'], batch_size,
                                            test_mode)
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)

        return (x, point_misc) if not test_mode else x

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> dict:
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.
                    - points (list[torch.Tensor]): Point cloud of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            dict: A dictionary of loss components.
        """
        x, point_misc = self.extract_feat(batch_inputs_dict, test_mode=False)
        batch_gt_bboxes_3d = [
            data_sample.gt_instances_3d.bboxes_3d
            for data_sample in batch_data_samples
        ]
        aux_loss = self.middle_encoder.aux_loss(*point_misc,
                                                batch_gt_bboxes_3d)
        losses = self.bbox_head.loss(x, batch_data_samples)
        losses.update(aux_loss)
        return losses
