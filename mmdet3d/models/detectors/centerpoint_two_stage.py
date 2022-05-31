# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet3d.ops import Voxelization
from mmdet.models import DETECTORS
from .. import builder
from .two_stage import TwoStage3DDetector


@DETECTORS.register_module()
class CenterPointTwoStage(TwoStage3DDetector):
    """Two-stage center point.

    Args:
        voxel_layer (dict): Config of voxelization layer.
        voxel_encoder (dict): Config of voxelization encoder.
        middle_encoder (dict): Config of middle encoder.
        backbone (dict): Config of one-stage backbone.
        neck (dict): Config of one-stage neck.
        rpn_head (dict): Config of one-stage head.
        roi_head (dict): Config of two-stage head.
        train_cfg (dict, optional): Config of the training. Defaults to None.
            - rpn (dict): Training config for RPN head.
            - rcnn (dict): Training config for two-stage roi head.
        test_cfg (dict, optional): Config of the testing. Defaults to None.
            - rpn (dict): Training config for RPN head.
            - rcnn (dict): Training config for two-stage roi head.
        init_cfg (dict, optional): Initialization config dict. Defaults to None
    """

    def __init__(self,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(CenterPointTwoStage, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.voxel_layer = Voxelization(**voxel_layer)
        self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        self.middle_encoder = builder.build_middle_encoder(middle_encoder)

    def extract_feat(self, points):
        """Extract features from images and points.

        Args:
            points (list[torch.Tensor]): Points of each sample. The outer
                list indicates point cloud in a batch. The shape of inside
                point cloud is [N, C]

        Returns:
            list[torch.Tensor]: Multi-level feature maps. The shape of each
                feature map is [B, C_i, H_i, W_i]
        """
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        batch_size = len(points)
        x = self.middle_encoder(voxel_features, coors, batch_size)
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, points, img_metas, gt_bboxes_3d, gt_labels_3d):
        """Training forward function of CenterPointTwoStage.

        Args:
            points (list[torch.Tensor]): Points of each sample. The outer
                list indicates point cloud in a batch. The shape of inside
                point cloud is [N, C]
            imput_metas (list[dict]): Meta info of each input.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]):
                GT bboxes of each sample. The bboxes are encapsulated
                by 3D box structures.
            gt_labels_3d (list[LongTensor]): GT labels of each sample.

        Returns:
            dict: Losses from CenterPoint Two-Stage head.
                - loss_bbox (torch.Tensor): Loss of bboxes.
        """
        losses = dict()
        # - extract feature
        bev_feature = self.extract_feat(points)  # list[torch.Tensor]
        # - one-stage loss
        one_stage_head_outs = self.rpn_head(bev_feature)
        ont_stage_loss_inputs = [
            gt_bboxes_3d, gt_labels_3d, one_stage_head_outs
        ]
        one_stage_losses = self.rpn_head.loss(*ont_stage_loss_inputs)
        losses.update(one_stage_losses)
        # - get one-stage rois
        rois = self.rpn_head.get_bboxes(one_stage_head_outs, img_metas)
        # - two-stage loss
        roi_losses = self.roi_head.forward_train(bev_feature, img_metas, rois,
                                                 gt_bboxes_3d, gt_labels_3d)
        losses.update(roi_losses)

        return losses

    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        """Test without augmentation.

        Args:
            points (list[torch.Tensor]): Points of each sample. The outer
                list indicates point cloud in a batch. The shape of inside
                point cloud is [N, C]
            imput_metas (list[dict]): Meta info of each input.
            imgs (list[torch.Tensor], optional): Images of each sample.
                Default to None.
            rescale (bool, optional): Whether to rescale results. Default
                to False.

        Returns:
            list[dict[str, torch.Tensor]]: Bounding box results in cpu mode.
                - boxes_3d (torch.Tensor): 3D boxes.
                - scores_3d (torch.Tensor): Prediction scores.
                - labels_3d (torch.Tensor): Box labels.
                - attrs_3d (torch.Tensor, optional): Box attributes.
        """

        assert self.with_bbox, 'Bbox head must be implemented.'
        # extract feature
        bev_feature = self.extract_feat(points)
        one_stage_head_outs = self.rpn_head(bev_feature)
        # get one-stage rois
        rois = self.rpn_head.get_bboxes(one_stage_head_outs, img_metas)
        # get two-stage proposals
        bbox_results = self.roi_head.simple_test(bev_feature, img_metas, rois)
        return bbox_results

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample. The outer
                list indicates point cloud in a batch. The shape of inside
                point cloud is [N, C]

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
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
            # first channel pad the number of the batch
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch
