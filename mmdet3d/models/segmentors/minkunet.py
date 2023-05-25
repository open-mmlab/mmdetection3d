# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import OptSampleList, SampleList
from .encoder_decoder import EncoderDecoder3D


@MODELS.register_module()
class MinkUNet(EncoderDecoder3D):
    r"""MinkUNet is the implementation of `4D Spatio-Temporal ConvNets.
    <https://arxiv.org/abs/1904.08755>`_ with TorchSparse backend.

    Refer to `implementation code <https://github.com/mit-han-lab/spvnas>`_.

    Args:
        kwargs (dict): Arguments are the same as those in
            :class:`EncoderDecoder3D`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def loss(self, inputs: dict, data_samples: SampleList):
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'voxels' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - voxels (dict): Voxel feature and coords after voxelization.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.extract_feat(inputs)
        losses = self.decode_head.loss(x, data_samples, self.train_cfg)
        return losses

    def predict(self, inputs: dict,
                batch_data_samples: SampleList) -> SampleList:
        """Simple test with single scene.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'voxels' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - voxels (dict): Voxel feature and coords after voxelization.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            List[:obj:`Det3DDataSample`]: Segmentation results of the input
            points. Each Det3DDataSample usually contains:

            - ``pred_pts_seg`` (PointData): Prediction of 3D semantic
              segmentation.
            - ``pts_seg_logits`` (PointData): Predicted logits of 3D semantic
              segmentation before normalization.
        """
        x = self.extract_feat(inputs)
        seg_logits_list = self.decode_head.predict(x, batch_data_samples)
        for i in range(len(seg_logits_list)):
            seg_logits_list[i] = seg_logits_list[i].transpose(0, 1)

        return self.postprocess_result(seg_logits_list, batch_data_samples)

    def _forward(self,
                 batch_inputs_dict: dict,
                 batch_data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'voxels' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - voxels (dict): Voxel feature and coords after voxelization.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`. Defaults to None.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(batch_inputs_dict)
        return self.decode_head.forward(x)

    def extract_feat(self, batch_inputs_dict: dict) -> Tensor:
        """Extract features from voxels.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'voxels' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - voxels (dict): Voxel feature and coords after voxelization.

        Returns:
            SparseTensor: voxels with features.
        """
        voxel_dict = batch_inputs_dict['voxels']
        x = self.backbone(voxel_dict['voxels'], voxel_dict['coors'])
        if self.with_neck:
            x = self.neck(x)
        return x
