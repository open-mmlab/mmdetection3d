# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence

from torch import Tensor

from mmdet3d.registry import MODELS
from .mvx_two_stage import MVXTwoStageDetector


@MODELS.register_module()
class MVXFasterRCNN(MVXTwoStageDetector):
    """Multi-modality VoxelNet using Faster R-CNN."""

    def __init__(self, **kwargs):
        super(MVXFasterRCNN, self).__init__(**kwargs)


@MODELS.register_module()
class DynamicMVXFasterRCNN(MVXTwoStageDetector):
    """Multi-modality VoxelNet using Faster R-CNN and dynamic voxelization."""

    def __init__(self, **kwargs):
        super(DynamicMVXFasterRCNN, self).__init__(**kwargs)

    def extract_pts_feat(
            self,
            voxel_dict: Dict[str, Tensor],
            points: Optional[List[Tensor]] = None,
            img_feats: Optional[Sequence[Tensor]] = None,
            batch_input_metas: Optional[List[dict]] = None
    ) -> Sequence[Tensor]:
        """Extract features of points.

        Args:
            voxel_dict(Dict[str, Tensor]): Dict of voxelization infos.
            points (List[tensor], optional):  Point cloud of multiple inputs.
            img_feats (list[Tensor], tuple[tensor], optional): Features from
                image backbone.
            batch_input_metas (list[dict], optional): The meta information
                of multiple samples. Defaults to True.

        Returns:
            Sequence[tensor]: points features of multiple inputs
            from backbone or neck.
        """
        if not self.with_pts_bbox:
            return None
        voxel_features, feature_coors = self.pts_voxel_encoder(
            voxel_dict['voxels'], voxel_dict['coors'], points, img_feats,
            batch_input_metas)
        batch_size = voxel_dict['coors'][-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, feature_coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x
