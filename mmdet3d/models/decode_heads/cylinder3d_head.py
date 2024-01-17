# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import torch
import torch.nn.functional as F
from mmcv.ops import SparseModule, SubMConv3d
from torch import Tensor

from mmdet3d.models.data_preprocessors.voxelize import dynamic_scatter_3d
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import OptConfigType
from .decode_head import Base3DDecodeHead


@MODELS.register_module()
class Cylinder3DHead(Base3DDecodeHead):
    """Cylinder3D decoder head.

    Decoder head used in `Cylinder3D <https://arxiv.org/abs/2011.10033>`_.
    Refer to the
    `official code <https://https://github.com/xinge008/Cylinder3D>`_.

    Args:
        loss_lovasz (dict or :obj:`ConfigDict`, optional): Config of Lovasz
            loss. Defaults to None.
    """

    def __init__(self, loss_lovasz: OptConfigType = None, **kwargs) -> None:
        super(Cylinder3DHead, self).__init__(**kwargs)

        if loss_lovasz is not None:
            self.loss_lovasz = MODELS.build(loss_lovasz)
        else:
            self.loss_lovasz = None

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

    def forward(self, feat_dict: dict) -> dict:
        """Forward function."""
        sparse_logits = self.cls_seg(feat_dict['voxel_feats'])
        # put logits into `feat_dict` for voxel2point mapping.
        feat_dict['logits'] = sparse_logits.features
        return feat_dict

    def loss_by_feat(self, feat_dict: dict,
                     batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """Compute semantic segmentation loss.

        Args:
            feat_dict (dict): The dict may contain `logits`, `point2voxel_map`.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """

        voxel_semantic_segs = []
        coors = feat_dict['coors']
        for batch_idx, data_sample in enumerate(batch_data_samples):
            pts_semantic_mask = data_sample.gt_pts_seg.pts_semantic_mask
            batch_mask = coors[:, 0] == batch_idx
            this_coors = coors[batch_mask, 1:]
            voxel_semantic_mask, _, _ = dynamic_scatter_3d(
                F.one_hot(pts_semantic_mask.long()).float(), this_coors,
                'mean')
            voxel_semantic_mask = torch.argmax(voxel_semantic_mask, dim=-1)
            voxel_semantic_segs.append(voxel_semantic_mask)
        seg_label = torch.cat(voxel_semantic_segs)
        seg_logit_feat = feat_dict['logits']
        loss = dict()
        loss['loss_ce'] = self.loss_ce(
            seg_logit_feat, seg_label, ignore_index=self.ignore_index)
        if self.loss_lovasz is not None:
            loss['loss_lovasz'] = self.loss_lovasz(
                seg_logit_feat, seg_label, ignore_index=self.ignore_index)

        return loss

    def predict(
        self,
        feat_dict: dict,
        batch_data_samples: SampleList,
    ) -> List[Tensor]:
        """Forward function for testing.

        Args:
            feat_dict (dict): Features from backbone.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`. We use `point2voxel_map` in this function.

        Returns:
            List[Tensor]: List of point-wise segmentation logits.
        """
        feat_dict = self.forward(feat_dict)
        seg_pred_list = self.predict_by_feat(feat_dict, batch_data_samples)
        return seg_pred_list

    def predict_by_feat(self, feat_dict: dict,
                        batch_data_samples: SampleList) -> List[Tensor]:
        """Predict function.

        Args:
            feat_dict (dict): The dict may contain `logits`, `point2voxel_map`.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            List[Tensor]: List of point-wise segmentation logits.
        """
        seg_logits = feat_dict['logits']

        seg_pred_list = []
        coors = feat_dict['voxel_coors']
        for batch_idx in range(len(batch_data_samples)):
            batch_mask = coors[:, 0] == batch_idx
            seg_logits_sample = seg_logits[batch_mask]
            point2voxel_map = feat_dict['point2voxel_maps'][batch_idx].long()
            point_seg_predicts = seg_logits_sample[point2voxel_map]
            seg_pred_list.append(point_seg_predicts)

        return seg_pred_list
