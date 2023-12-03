# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.utils import OptConfigType
from ...structures.det3d_data_sample import OptSampleList, SampleList
from .encoder_decoder import EncoderDecoder3D


@MODELS.register_module()
class VoxelSegmentor(EncoderDecoder3D):
    """Voxel-Based Methods for LiDAR Segmentation.

    Args:
        voxel_encoder (dict or :obj:`ConfigDict`, optional): The config for the
            points2voxel encoder of segmentor. Defaults to None.
    """

    def __init__(self, voxel_encoder: OptConfigType = None, **kwargs) -> None:
        super(VoxelSegmentor, self).__init__(**kwargs)

        if voxel_encoder is not None:
            self.voxel_encoder = MODELS.build(voxel_encoder)

    @property
    def with_voxel_encoder(self) -> bool:
        """bool: Whether the segmentor has voxel_encoder."""
        return hasattr(self, 'voxel_encoder') and \
            self.voxel_encoder is not None

    def extract_feat(self, batch_inputs_dict: dict) -> dict:
        """Extract features from voxels.

        Args:
            batch_inputs_dict (dict): Input sample dict which includes 'points'
                and 'voxels' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - voxels (dict): Voxel feature and coords after voxelization.

        Returns:
            dict: The dict containing features.
        """
        voxel_dict = batch_inputs_dict['voxels'].copy()
        if self.with_voxel_encoder:
            voxel_dict = self.voxel_encoder(voxel_dict)
        voxel_dict = self.backbone(voxel_dict)
        if self.with_neck:
            voxel_dict = self.neck(voxel_dict)
        return voxel_dict

    def loss(self, batch_inputs_dict: dict,
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """

        # extract features using backbone
        voxel_dict = self.extract_feat(batch_inputs_dict)
        losses = dict()
        loss_decode = self._decode_head_forward_train(voxel_dict,
                                                      batch_data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                voxel_dict, batch_data_samples)
            losses.update(loss_aux)

        return losses

    def predict(self,
                batch_inputs_dict: dict,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Simple test with single scene.

        Args:
            batch_inputs_dict (dict): Input sample dict which includes 'points'
                and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.
            rescale (bool): Whether transform to original number of points.
                Will be used for voxelization based segmentors.
                Defaults to True.

        Returns:
            List[:obj:`Det3DDataSample`]: Segmentation results of the input
            points. Each Det3DDataSample usually contains:

            - ``pred_pts_seg`` (PointData): Prediction of 3D semantic
              segmentation.
            - ``pts_seg_logits`` (PointData): Predicted logits of 3D semantic
              segmentation before normalization.
        """
        # 3D segmentation requires per-point prediction, so it's impossible
        # to use down-sampling to get a batch of scenes with same num_points
        # therefore, we only support testing one scene every time
        batch_input_metas = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)

        voxel_dict = self.extract_feat(batch_inputs_dict)
        seg_logits_list = self.decode_head.predict(voxel_dict,
                                                   batch_input_metas,
                                                   self.test_cfg)
        for i in range(len(seg_logits_list)):
            seg_logits_list[i] = seg_logits_list[i].transpose(0, 1)

        return self.postprocess_result(seg_logits_list, batch_data_samples)

    def _forward(self,
                 batch_inputs_dict: dict,
                 batch_data_samples: OptSampleList = None) -> dict:
        """Network forward process.

        Args:
            batch_inputs_dict (dict): Input sample dict which includes 'points'
                and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            dict: The dict containing forward output of model without any
            post-processes.
        """
        voxel_dict = self.extract_feat(batch_inputs_dict)
        return self.decode_head.forward(voxel_dict)
