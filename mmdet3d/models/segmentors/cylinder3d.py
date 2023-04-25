# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from ...structures.det3d_data_sample import SampleList
from .encoder_decoder import EncoderDecoder3D


@MODELS.register_module()
class Cylinder3D(EncoderDecoder3D):
    """`Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR
    Segmentation.

        <https://arxiv.org/abs/2011.10033>`_.

    Args:
        voxel_encoder (dict or :obj:`ConfigDict`): The config for the
            points2voxel encoder of segmentor.
        backbone (dict or :obj:`ConfigDict`): The config for the backnone of
            segmentor.
        decode_head (dict or :obj:`ConfigDict`): The config for the decode
            head of segmentor.
        neck (dict or :obj:`ConfigDict`, optional): The config for the neck of
            segmentor. Defaults to None.
        auxiliary_head (dict or :obj:`ConfigDict` or List[dict or
            :obj:`ConfigDict`], optional): The config for the auxiliary head of
            segmentor. Defaults to None.
        loss_regularization (dict or :obj:`ConfigDict` or List[dict or
            :obj:`ConfigDict`], optional): The config for the regularization
            loass. Defaults to None.
        train_cfg (dict or :obj:`ConfigDict`, optional): The config for
            training. Defaults to None.
        test_cfg (dict or :obj:`ConfigDict`, optional): The config for testing.
            Defaults to None.
        data_preprocessor (dict or :obj:`ConfigDict`, optional): The
            pre-process config of :class:`BaseDataPreprocessor`.
            Defaults to None.
        init_cfg (dict or :obj:`ConfigDict` or List[dict or :obj:`ConfigDict`],
            optional): The weight initialized config for :class:`BaseModule`.
            Defaults to None.
    """

    def __init__(self,
                 voxel_encoder: ConfigType,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 loss_regularization: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super(Cylinder3D, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            loss_regularization=loss_regularization,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        self.voxel_encoder = MODELS.build(voxel_encoder)

    def extract_feat(self, batch_inputs: dict) -> Tensor:
        """Extract features from points."""
        encoded_feats = self.voxel_encoder(batch_inputs['voxels']['voxels'],
                                           batch_inputs['voxels']['coors'])
        batch_inputs['voxels']['voxel_coors'] = encoded_feats[1]
        x = self.backbone(encoded_feats[0], encoded_feats[1],
                          len(batch_inputs['points']))
        if self.with_neck:
            x = self.neck(x)
        return x

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
        x = self.extract_feat(batch_inputs_dict)
        losses = dict()
        loss_decode = self._decode_head_forward_train(x, batch_data_samples)
        losses.update(loss_decode)

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
        x = self.extract_feat(batch_inputs_dict)
        seg_logits_list = self.decode_head.predict(x, batch_inputs_dict,
                                                   batch_data_samples)
        for i in range(len(seg_logits_list)):
            seg_logits_list[i] = seg_logits_list[i].transpose(0, 1)

        return self.postprocess_result(seg_logits_list, batch_data_samples)
