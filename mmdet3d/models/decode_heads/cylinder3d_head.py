# Copyright (c) OpenMMLab. All rights reserved.

import torch
from mmcv.ops import SparseConvTensor, SparseModule, SubMConv3d

from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import OptMultiConfig
from mmdet3d.utils.typing_utils import ConfigType
from .decode_head import Base3DDecodeHead


@MODELS.register_module()
class Cylinder3DHead(Base3DDecodeHead):
    """Cylinder3D decoder head.

    Decoder head used in `Cylinder3D <https://arxiv.org/abs/2011.10033>`_.
    Refer to the
    `official code <https://https://github.com/xinge008/Cylinder3D>`_.

    Args:
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Defaults to 0.
        conv_cfg (dict or :obj:`ConfigDict`): Config of conv layers.
            Defaults to dict(type='Conv1d').
        norm_cfg (dict or :obj:`ConfigDict`): Config of norm layers.
            Defaults to dict(type='BN1d').
        act_cfg (dict or :obj:`ConfigDict`): Config of activation layers.
            Defaults to dict(type='ReLU').
        loss_ce (dict or :obj:`ConfigDict`): Config of CrossEntropy loss.
            Defaults to dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=False,
                     class_weight=None,
                     loss_weight=1.0).
        loss_lovasz (dict or :obj:`ConfigDict`): Config of Lovasz loss.
            Defaults to dict(type='LovaszLoss', loss_weight=1.0).
        conv_seg_kernel_size (int): The kernel size used in conv_seg.
            Defaults to 3.
        ignore_index (int): The label index to be ignored. When using masked
            BCE loss, ignore_index should be set to None. Defaults to 19.
        init_cfg (dict or :obj:`ConfigDict` or list[dict or :obj:`ConfigDict`],
            optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 channels: int,
                 num_classes: int,
                 dropout_ratio: float = 0,
                 conv_cfg: ConfigType = dict(type='Conv1d'),
                 norm_cfg: ConfigType = dict(type='BN1d'),
                 act_cfg: ConfigType = dict(type='ReLU'),
                 loss_ce: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=False,
                     class_weight=None,
                     loss_weight=1.0),
                 loss_lovasz: ConfigType = dict(
                     type='LovaszLoss', loss_weight=1.0),
                 conv_seg_kernel_size: int = 3,
                 ignore_index: int = 19,
                 init_cfg: OptMultiConfig = None) -> None:
        super(Cylinder3DHead, self).__init__(
            channels=channels,
            num_classes=num_classes,
            dropout_ratio=dropout_ratio,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            conv_seg_kernel_size=conv_seg_kernel_size,
            init_cfg=init_cfg)

        self.loss_lovasz = MODELS.build(loss_lovasz)
        self.loss_ce = MODELS.build(loss_ce)
        self.ignore_index = ignore_index

    def build_conv_seg(self, channels: int, num_classes: int,
                       kernel_size: int) -> SparseModule:
        return SubMConv3d(
            channels,
            num_classes,
            indice_key='logit',
            kernel_size=kernel_size,
            stride=1,
            padding=1,
            bias=True)

    def forward(self, sparse_voxels: SparseConvTensor) -> SparseConvTensor:
        """Forward function."""
        sparse_logits = self.cls_seg(sparse_voxels)
        return sparse_logits

    def loss_by_feat(self, seg_logit: SparseConvTensor,
                     batch_data_samples: SampleList) -> dict:
        """Compute semantic segmentation loss.

        Args:
            seg_logit (SparseConvTensor): Predicted per-voxel
                segmentation logits of shape [num_voxels, num_classes]
                stored in SparseConvTensor.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """

        gt_semantic_segs = [
            data_sample.gt_pts_seg.voxel_semantic_mask
            for data_sample in batch_data_samples
        ]
        seg_label = torch.cat(gt_semantic_segs)
        seg_logit_feat = seg_logit.features
        loss = dict()
        loss['loss_ce'] = self.loss_ce(
            seg_logit_feat, seg_label, ignore_index=self.ignore_index)
        loss['loss_lovasz'] = self.loss_lovasz(
            seg_logit_feat, seg_label, ignore_index=self.ignore_index)

        return loss

    def predict(
        self,
        inputs: SparseConvTensor,
        batch_inputs_dict: dict,
        batch_data_samples: SampleList,
    ) -> torch.Tensor:
        """Forward function for testing.

        Args:
            inputs (SparseConvTensor): Feature from backbone.
            batch_inputs_dict (dict): Input sample dict which includes 'points'
                and 'voxels' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - voxels (dict): Dict of voxelized voxels and the corresponding
                coordinates.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`. We use `point2voxel_map` in this function.

        Returns:
            List[torch.Tensor]: List of point-wise segmentation logits.
        """
        seg_logits = self.forward(inputs).features

        seg_pred_list = []
        coors = batch_inputs_dict['voxels']['voxel_coors']
        for batch_idx in range(len(batch_data_samples)):
            seg_logits_sample = seg_logits[coors[:, 0] == batch_idx]
            point2voxel_map = batch_data_samples[
                batch_idx].point2voxel_map.long()
            point_seg_predicts = seg_logits_sample[point2voxel_map]
            seg_pred_list.append(point_seg_predicts)

        return seg_pred_list
