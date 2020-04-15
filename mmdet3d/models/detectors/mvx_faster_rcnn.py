import torch
import torch.nn.functional as F

from mmdet.models import DETECTORS
from .mvx_two_stage import MVXTwoStageDetector


@DETECTORS.register_module
class DynamicMVXFasterRCNN(MVXTwoStageDetector):

    def __init__(self, **kwargs):
        super(DynamicMVXFasterRCNN, self).__init__(**kwargs)

    def extract_pts_feat(self, points, img_feats, img_meta):
        if not self.with_pts_bbox:
            return None
        voxels, coors = self.voxelize(points)
        # adopt an early fusion strategy
        if self.with_fusion:
            voxels = self.pts_fusion_layer(img_feats, points, voxels, img_meta)
        voxel_features, feature_coors = self.pts_voxel_encoder(voxels, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, feature_coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    @torch.no_grad()
    def voxelize(self, points):
        coors = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            res_coors = self.pts_voxel_layer(res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch


@DETECTORS.register_module
class DynamicMVXFasterRCNNV2(DynamicMVXFasterRCNN):

    def __init__(self, **kwargs):
        super(DynamicMVXFasterRCNNV2, self).__init__(**kwargs)

    def extract_pts_feat(self, points, img_feats, img_meta):
        if not self.with_pts_bbox:
            return None
        voxels, coors = self.voxelize(points)
        voxel_features, feature_coors = self.pts_voxel_encoder(
            voxels, coors, points, img_feats, img_meta)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, feature_coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x


@DETECTORS.register_module
class MVXFasterRCNNV2(MVXTwoStageDetector):

    def __init__(self, **kwargs):
        super(MVXFasterRCNNV2, self).__init__(**kwargs)

    def extract_pts_feat(self, pts, img_feats, img_meta):
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                img_feats, img_meta)

        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)

        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x


@DETECTORS.register_module
class DynamicMVXFasterRCNNV3(DynamicMVXFasterRCNN):

    def __init__(self, **kwargs):
        super(DynamicMVXFasterRCNNV3, self).__init__(**kwargs)

    def extract_pts_feat(self, points, img_feats, img_meta):
        if not self.with_pts_bbox:
            return None
        voxels, coors = self.voxelize(points)
        voxel_features, feature_coors = self.pts_voxel_encoder(voxels, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, feature_coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x, coors, points, img_feats, img_meta)
        return x
