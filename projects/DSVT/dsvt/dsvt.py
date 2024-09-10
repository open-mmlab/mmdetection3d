from typing import Dict, List, Optional

import torch
from torch import Tensor

from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample


@MODELS.register_module()
class DSVT(Base3DDetector):
    """DSVT detector."""

    def __init__(self,
                 voxel_encoder: Optional[dict] = None,
                 middle_encoder: Optional[dict] = None,
                 backbone: Optional[dict] = None,
                 neck: Optional[dict] = None,
                 map2bev: Optional[dict] = None,
                 bbox_head: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 **kwargs):
        super(DSVT, self).__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor, **kwargs)

        if voxel_encoder:
            self.voxel_encoder = MODELS.build(voxel_encoder)
        if middle_encoder:
            self.middle_encoder = MODELS.build(middle_encoder)
        if backbone:
            self.backbone = MODELS.build(backbone)
        self.map2bev = MODELS.build(map2bev)
        if neck is not None:
            self.neck = MODELS.build(neck)
        if bbox_head:
            bbox_head.update(train_cfg=train_cfg, test_cfg=test_cfg)
            self.bbox_head = MODELS.build(bbox_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

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

    def extract_feat(self, batch_inputs_dict: dict) -> tuple:
        """Extract features from images and points.
        Args:
            batch_inputs_dict (dict): Dict of batch inputs. It
                contains
                - points (List[tensor]):  Point cloud of multiple inputs.
                - imgs (tensor): Image tensor with shape (B, C, H, W).
        Returns:
             tuple: Two elements in tuple arrange as
             image features and point cloud features.
        """
        batch_out_dict = self.voxel_encoder(batch_inputs_dict)
        batch_out_dict = self.middle_encoder(batch_out_dict)
        batch_out_dict = self.map2bev(batch_out_dict)
        multi_feats = self.backbone(batch_out_dict['spatial_features'])
        feats = self.neck(multi_feats)

        return feats

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
        pts_feats = self.extract_feat(batch_inputs_dict)
        losses = dict()
        loss = self.bbox_head.loss(pts_feats, batch_data_samples)
        losses.update(loss)
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
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        pts_feats = self.extract_feat(batch_inputs_dict)
        results_list_3d = self.bbox_head.predict(pts_feats, batch_data_samples)

        detsamples = self.add_pred_to_datasample(batch_data_samples,
                                                 results_list_3d)
        return detsamples
