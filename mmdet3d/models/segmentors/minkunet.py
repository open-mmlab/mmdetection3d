# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

from torch import Tensor

from mmdet3d.models.layers.torchsparse import IS_TORCHSPARSE_AVAILABLE
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import OptSampleList, SampleList
from .encoder_decoder import EncoderDecoder3D


@MODELS.register_module()
class MinkUNet(EncoderDecoder3D):
    """Base class for 3D segmentors.

    Args:
        data_preprocessor (dict, optional): Model preprocessing config
            for processing the input data. it usually includes
            ``to_rgb``, ``pad_size_divisor``, ``pad_val``,
            ``mean`` and ``std``. Default to None.
       init_cfg (dict, optional): the config to control the
           initialization. Default to None.
    """

    def __init__(self, **kwargs) -> None:
        if not IS_TORCHSPARSE_AVAILABLE:
            raise ImportError(
                'Please follow `getting_started.md` to install Torchsparse.`'
            )  # noqa: E501
        super().__init__(**kwargs)

    def loss(self, inputs: dict, data_samples: SampleList):
        """"""
        x = self.extract_feat(inputs)
        losses = self.decode_head.loss(x, data_samples)
        return losses

    def predict(self, inputs: dict, data_samples: SampleList) -> SampleList:
        """Simple test with single scene.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'imgs' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor, optional): Image tensor has shape
                    (B, C, H, W).
            batch_data_samples (list[:obj:`Det3DDataSample`]): The det3d
                data samples. It usually includes information such
                as `metainfo` and `gt_pts_sem_seg`.
            rescale (bool): Whether transform to original number of points.
                Will be used for voxelization based segmentors.
                Defaults to True.

        Returns:
            list[dict]: The output prediction result with following keys:

                - semantic_mask (Tensor): Segmentation mask of shape [N].
        """
        x = self.extract_feat(inputs)
        preds = self.decode_head.predict(x, data_samples)

        return self.postprocess_result(preds, data_samples)

    def _forward(self,
                 batch_inputs_dict: dict,
                 batch_data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'imgs' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor, optional): Image tensor has shape
                  (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_pts_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(batch_inputs_dict)
        return self.decode_head.forward(x)

    def extract_feat(self, batch_inputs_dict: dict) -> Tuple[Tensor]:
        """Extract features from points."""
        voxel_dict = batch_inputs_dict['voxels']
        x = self.backbone(voxel_dict['voxels'], voxel_dict['coors'])
        if self.with_neck:
            x = self.neck(x)
        return x
