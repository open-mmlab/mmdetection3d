# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

from mmengine.structures import InstanceData
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from .base_3droi_head import Base3DRoIHead


@MODELS.register_module()
class H3DRoIHead(Base3DRoIHead):
    """H3D roi head for H3DNet.

    Args:
        primitive_list (List): Configs of primitive heads.
        bbox_head (ConfigDict): Config of bbox_head.
        train_cfg (ConfigDict): Training config.
        test_cfg (ConfigDict): Testing config.
    """

    def __init__(self,
                 primitive_list: List[dict],
                 bbox_head: dict = None,
                 train_cfg: dict = None,
                 test_cfg: dict = None,
                 init_cfg: dict = None):
        super(H3DRoIHead, self).__init__(
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        # Primitive module
        assert len(primitive_list) == 3
        self.primitive_z = MODELS.build(primitive_list[0])
        self.primitive_xy = MODELS.build(primitive_list[1])
        self.primitive_line = MODELS.build(primitive_list[2])

    def init_mask_head(self):
        """Initialize mask head, skip since ``H3DROIHead`` does not have
        one."""
        pass

    def init_bbox_head(self, dummy_args, bbox_head):
        """Initialize box head.

        Args:
            dummy_args (optional): Just to compatible with
                the interface in base class
            bbox_head (dict): Config for bbox head.
        """
        bbox_head['train_cfg'] = self.train_cfg
        bbox_head['test_cfg'] = self.test_cfg
        self.bbox_head = MODELS.build(bbox_head)

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        pass

    def loss(self, points: List[Tensor], feats_dict: dict,
             batch_data_samples: List[Det3DDataSample], **kwargs):
        """Training forward function of PartAggregationROIHead.

        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
            feats_dict (dict): Dict of feature.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            dict: losses from each head.
        """
        losses = dict()

        primitive_loss_inputs = (points, feats_dict, batch_data_samples)
        # note the feats_dict would be added new key and value in each head.
        loss_z = self.primitive_z.loss(*primitive_loss_inputs)
        loss_xy = self.primitive_xy.loss(*primitive_loss_inputs)
        loss_line = self.primitive_line.loss(*primitive_loss_inputs)

        losses.update(loss_z)
        losses.update(loss_xy)
        losses.update(loss_line)

        targets = feats_dict.pop('targets')

        bbox_loss = self.bbox_head.loss(
            points,
            feats_dict,
            rpn_targets=targets,
            batch_data_samples=batch_data_samples)
        losses.update(bbox_loss)
        return losses

    def predict(self,
                points: List[Tensor],
                feats_dict: Dict[str, Tensor],
                batch_data_samples: List[Det3DDataSample],
                suffix='_optimized',
                **kwargs) -> List[InstanceData]:
        """
        Args:
            points (list[tensor]): Point clouds of multiple samples.
            feats_dict (dict): Features from FPN or backbone..
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes meta information of data.

        Returns:
            list[:obj:`InstanceData`]: List of processed predictions. Each
            InstanceData contains 3d Bounding boxes and corresponding
            scores and labels.
        """

        result_z = self.primitive_z(feats_dict)
        feats_dict.update(result_z)

        result_xy = self.primitive_xy(feats_dict)
        feats_dict.update(result_xy)

        result_line = self.primitive_line(feats_dict)
        feats_dict.update(result_line)

        bbox_preds = self.bbox_head(feats_dict)
        feats_dict.update(bbox_preds)
        results_list = self.bbox_head.predict(
            points, feats_dict, batch_data_samples, suffix=suffix)

        return results_list
