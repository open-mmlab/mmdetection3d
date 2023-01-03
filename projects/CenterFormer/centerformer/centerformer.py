# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

import torch
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample


@MODELS.register_module()
class CenterFormer(Base3DDetector):
    """Base class of center-based 3D detector.

    Args:
        voxel_encoder (dict, optional): Point voxelization
            encoder layer. Defaults to None.
        middle_encoder (dict, optional): Middle encoder layer
            of points cloud modality. Defaults to None.
        pts_fusion_layer (dict, optional): Fusion layer.
            Defaults to None.
        backbone (dict, optional): Backbone of extracting
            points features. Defaults to None.
        neck (dict, optional): Neck of extracting
            points features. Defaults to None.
        bbox_head (dict, optional): Bboxes head of
            point cloud modality. Defaults to None.
        train_cfg (dict, optional): Train config of model.
            Defaults to None.
        test_cfg (dict, optional): Train config of model.
            Defaults to None.
        init_cfg (dict, optional): Initialize config of
            model. Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`Det3DDataPreprocessor`. Defaults to None.
    """

    def __init__(self,
                 voxel_encoder: Optional[dict] = None,
                 middle_encoder: Optional[dict] = None,
                 backbone: Optional[dict] = None,
                 neck: Optional[dict] = None,
                 bbox_head: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 **kwargs):
        super(CenterFormer, self).__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor, **kwargs)

        if voxel_encoder:
            self.voxel_encoder = MODELS.build(voxel_encoder)
        if middle_encoder:
            self.middle_encoder = MODELS.build(middle_encoder)
        if backbone:
            backbone.update(train_cfg=train_cfg, test_cfg=test_cfg)
            self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        if bbox_head:
            bbox_head.update(train_cfg=train_cfg, test_cfg=test_cfg)
            self.bbox_head = MODELS.build(bbox_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, _BatchNorm):
                torch.nn.init.uniform_(m.weight)

    @property
    def with_bbox(self):
        """bool: Whether the detector has a 3D box head."""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_backbone(self):
        """bool: Whether the detector has a 3D backbone."""
        return hasattr(self, 'backbone') and self.backbone is not None

    @property
    def with_voxel_encoder(self):
        """bool: Whether the detector has a voxel encoder."""
        return hasattr(self,
                       'voxel_encoder') and self.voxel_encoder is not None

    @property
    def with_middle_encoder(self):
        """bool: Whether the detector has a middle encoder."""
        return hasattr(self,
                       'middle_encoder') and self.middle_encoder is not None

    def _forward(self):
        pass

    def extract_feat(self, batch_inputs_dict: dict,
                     batch_input_metas: List[dict]) -> tuple:
        """Extract features from images and points.
        Args:
            batch_inputs_dict (dict): Dict of batch inputs. It
                contains
                - points (List[tensor]):  Point cloud of multiple inputs.
                - imgs (tensor): Image tensor with shape (B, C, H, W).
            batch_input_metas (list[dict]): Meta information of multiple inputs
                in a batch.
        Returns:
             tuple: Two elements in tuple arrange as
             image features and point cloud features.
        """
        voxel_dict = batch_inputs_dict.get('voxels', None)
        voxel_features, feature_coors = self.voxel_encoder(
            voxel_dict['voxels'], voxel_dict['coors'])
        batch_size = voxel_dict['coors'][-1, 0].item() + 1
        x = self.middle_encoder(voxel_features, feature_coors, batch_size)

        return x

    def loss(self, batch_inputs_dict: Dict[List, torch.Tensor],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        """
        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' and `imgs` keys.
                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor): Tensor of batch images, has shape
                  (B, C, H ,W)
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, .
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        batch_input_metas = [item.metainfo for item in batch_data_samples]
        pts_feats = self.extract_feat(batch_inputs_dict, batch_input_metas)
        preds, batch_tatgets = self.backbone(pts_feats, batch_data_samples)
        preds = self.bbox_head(preds)
        losses = dict()
        losses.update(self.bbox_head.loss(preds, batch_tatgets))
        return losses
        # return self.bbox_head.predict(preds, batch_tatgets)

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
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        pts_feats = self.extract_feat(batch_inputs_dict, batch_input_metas)
        preds, _ = self.backbone(pts_feats, batch_data_samples)

        preds = self.bbox_head(preds)
        results_list_3d = self.bbox_head.predict(preds, batch_input_metas)

        detsamples = self.add_pred_to_datasample(batch_data_samples,
                                                 results_list_3d)
        return detsamples
