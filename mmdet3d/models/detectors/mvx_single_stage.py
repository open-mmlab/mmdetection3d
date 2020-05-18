import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.core import bbox3d2result
from mmdet3d.ops import Voxelization
from mmdet.models import DETECTORS
from .. import builder
from .base import BaseDetector


@DETECTORS.register_module()
class MVXSingleStageDetector(BaseDetector):

    def __init__(self,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 fusion_layer,
                 img_backbone,
                 pts_backbone,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(MVXSingleStageDetector, self).__init__()
        self.voxel_layer = Voxelization(**voxel_layer)
        self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        self.middle_encoder = builder.build_middle_encoder(middle_encoder)
        self.pts_backbone = builder.build_backbone(pts_backbone)

        if fusion_layer:
            self.fusion_layer = builder.build_fusion_layer(fusion_layer)
        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)

        pts_bbox_head.update(train_cfg=train_cfg)
        pts_bbox_head.update(test_cfg=test_cfg)
        self.pts_bbox_head = builder.build_head(pts_bbox_head)
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)
        if pts_neck is not None:
            self.pts_neck = builder.build_neck(pts_neck)
        if img_bbox_head is not None:
            self.img_bbox_head = builder.build_head(img_bbox_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(MVXSingleStageDetector, self).init_weights(pretrained)
        if self.with_img_backbone:
            self.img_backbone.init_weights(pretrained=pretrained)
        if self.with_img_neck:
            if isinstance(self.img_neck, nn.Sequential):
                for m in self.img_neck:
                    m.init_weights()
            else:
                self.img_neck.init_weights()
        if self.with_img_bbox:
            self.img_bbox_head.init_weights()
        if self.with_pts_bbox:
            self.pts_bbox_head.init_weights()

    @property
    def with_pts_bbox(self):
        return hasattr(self,
                       'pts_bbox_head') and self.pts_bbox_head is not None

    @property
    def with_img_bbox(self):
        return hasattr(self,
                       'img_bbox_head') and self.img_bbox_head is not None

    @property
    def with_img_backbone(self):
        return hasattr(self, 'img_backbone') and self.img_backbone is not None

    @property
    def with_fusion(self):
        return hasattr(self, 'fusion_layer') and self.fusion_layer is not None

    @property
    def with_img_neck(self):
        return hasattr(self, 'img_neck') and self.img_neck is not None

    @property
    def with_pts_neck(self):
        return hasattr(self, 'pts_neck') and self.pts_neck is not None

    def extract_feat(self, points, img, img_meta):
        if self.with_img_backbone:
            img_feats = self.img_backbone(img)
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_neck:
            x = self.pts_neck(x)
        return x

    @torch.no_grad()
    def voxelize(self, points):
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train(self,
                      points,
                      img_meta,
                      gt_bboxes_3d,
                      gt_labels,
                      img=None,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(points, img=img, img_meta=img_meta)
        outs = self.pts_bbox_head(x)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels, img_meta)
        losses = self.pts_bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def forward_test(self, **kwargs):
        return self.simple_test(**kwargs)

    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def simple_test(self,
                    points,
                    img_meta,
                    img=None,
                    gt_bboxes_3d=None,
                    rescale=False):
        x = self.extract_feat(points, img, img_meta)
        outs = self.pts_bbox_head(x)
        bbox_list = self.pts_bbox_head.get_bboxes(
            *outs, img_meta, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results[0]

    def aug_test(self, points, imgs, img_metas, rescale=False):
        raise NotImplementedError


@DETECTORS.register_module()
class DynamicMVXNet(MVXSingleStageDetector):

    def __init__(self,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 pts_backbone,
                 fusion_layer=None,
                 img_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(DynamicMVXNet, self).__init__(
            voxel_layer=voxel_layer,
            voxel_encoder=voxel_encoder,
            middle_encoder=middle_encoder,
            img_backbone=img_backbone,
            fusion_layer=fusion_layer,
            pts_backbone=pts_backbone,
            pts_neck=pts_neck,
            img_neck=img_neck,
            img_bbox_head=img_bbox_head,
            pts_bbox_head=pts_bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
        )

    def extract_feat(self, points, img, img_meta):
        if self.with_img_backbone:
            img_feats = self.img_backbone(img)
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        voxels, coors = self.voxelize(points)
        # adopt an early fusion strategy
        if self.with_fusion:
            voxels = self.fusion_layer(img_feats, points, voxels, img_meta)

        voxel_features, feature_coors = self.voxel_encoder(voxels, coors)
        batch_size = coors[-1, 0] + 1
        x = self.middle_encoder(voxel_features, feature_coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    @torch.no_grad()
    def voxelize(self, points):
        coors = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            res_coors = self.voxel_layer(res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch


@DETECTORS.register_module()
class DynamicMVXNetV2(DynamicMVXNet):

    def __init__(self,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 pts_backbone,
                 fusion_layer=None,
                 img_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(DynamicMVXNetV2, self).__init__(
            voxel_layer=voxel_layer,
            voxel_encoder=voxel_encoder,
            middle_encoder=middle_encoder,
            img_backbone=img_backbone,
            fusion_layer=fusion_layer,
            pts_backbone=pts_backbone,
            pts_neck=pts_neck,
            img_neck=img_neck,
            img_bbox_head=img_bbox_head,
            pts_bbox_head=pts_bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
        )

    def extract_feat(self, points, img, img_meta):
        if self.with_img_backbone:
            img_feats = self.img_backbone(img)
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        voxels, coors = self.voxelize(points)

        voxel_features, feature_coors = self.voxel_encoder(
            voxels, coors, points, img_feats, img_meta)
        batch_size = coors[-1, 0] + 1
        x = self.middle_encoder(voxel_features, feature_coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x


@DETECTORS.register_module()
class DynamicMVXNetV3(DynamicMVXNet):

    def __init__(self,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 pts_backbone,
                 fusion_layer=None,
                 img_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(DynamicMVXNetV3, self).__init__(
            voxel_layer=voxel_layer,
            voxel_encoder=voxel_encoder,
            middle_encoder=middle_encoder,
            img_backbone=img_backbone,
            fusion_layer=fusion_layer,
            pts_backbone=pts_backbone,
            pts_neck=pts_neck,
            img_neck=img_neck,
            img_bbox_head=img_bbox_head,
            pts_bbox_head=pts_bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
        )

    def extract_feat(self, points, img, img_meta):
        if self.with_img_backbone:
            img_feats = self.img_backbone(img)
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        voxels, coors = self.voxelize(points)
        voxel_features, feature_coors = self.voxel_encoder(voxels, coors)
        batch_size = coors[-1, 0] + 1
        x = self.middle_encoder(voxel_features, feature_coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x, coors, points, img_feats, img_meta)
        return x
