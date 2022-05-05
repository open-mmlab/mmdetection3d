# Copyright (c) OpenMMLab. All rights reserved.
from math import pi

import torch
from mmcv import Config

from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.core.bbox.samplers import IoUNegPiecewiseSampler
from mmdet3d.models import CenterPointBBoxHead
from mmdet3d.models.roi_heads.roi_extractors import BEVFeatureExtractor


class TestCenterPointBBoxHead:
    # def __init__(self):
    bbox_head = CenterPointBBoxHead(
        input_channels=128 * 3 * 5,
        shared_fc=[256, 256],
        cls_fc=[256, 256],
        reg_fc=[256, 256],
        dp_ratio=0.3,
        code_size=7,
        num_classes=1,
        loss_reg=dict(type='L1Loss', reduction='none', loss_weight=1.0),
        loss_cls=dict(
            type='CrossEntropyLoss', reduction='none', loss_weight=1.0),
        init_cfg=None)

    bev_feature_extractor_cfg = dict(
        pc_start=[-61.2, -61.2],
        voxel_size=[0.2, 0.2],
        downsample_stride=1,
    )

    bev_feature_extractor = BEVFeatureExtractor(**bev_feature_extractor_cfg)
    C, H, W = 128 * 3, 612, 612
    bev_feats = [torch.rand((1, C, H, W))]

    rois_tensor = torch.tensor([[0, 0, 0, 3.2, 1.6, 1.5, 0],
                                [0, 0, 0, 3.2, 1.6, 1.5, pi / 2],
                                [0, 0, 0, 3.2, 1.6, 1.5, pi]])
    rois = [[
        LiDARInstance3DBoxes(rois_tensor),
        torch.ones(rois_tensor.shape[0]),
        torch.zeros(rois_tensor.shape[0])
    ]]

    roi_features = bev_feature_extractor.forward(bev_feats, rois)  # [[3,5*C]]

    def test_forward(self):
        pred_res = self.bbox_head(self.roi_features)
        # assert batch size
        assert len(pred_res) == len(self.rois) == len(self.roi_features)
        # assert head shape
        assert pred_res[0]['cls'].shape == torch.Size(
            [3, self.bbox_head.num_classes])
        assert pred_res[0]['reg'].shape == torch.Size(
            [3, self.bbox_head.code_size])

    def test_get_bboxes(self):
        final_bboxes = self.bbox_head.get_bboxes(self.roi_features, None,
                                                 self.rois)
        # assert batch size
        assert len(final_bboxes) == len(self.rois) == len(self.roi_features)
        # assert the number of bboxes
        assert len(final_bboxes[0][0]) == len(self.rois[0][0])
        assert final_bboxes[0][1].shape == self.rois[0][1].shape
        assert final_bboxes[0][2].shape == self.rois[0][2].shape

    def test_get_targets(self):

        train_cfg = Config({
            'cls_pos_thr': 0.75,
            'cls_neg_thr': 0.25,
            'reg_pos_thr': 0.55
        })

        sampling_result = IoUNegPiecewiseSampler(
            1,
            pos_fraction=0.55,
            neg_piece_fractions=[0.8, 0.2],
            neg_iou_piece_thrs=[0.55, 0.1],
            return_iou=True)

        sampling_result.pos_bboxes = torch.Tensor(
            [[8.1517, 0.0384, -1.9496, 1.5271, 4.1131, 1.4879, 1.2076]])
        sampling_result.pos_gt_bboxes = torch.Tensor(
            [[7.8417, -0.1405, -1.9652, 1.6122, 3.2838, 1.5331, -2.0835]])
        sampling_result.iou = torch.Tensor(
            [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

        label, bbox_targets, pos_gt_bboxes, reg_mask, label_weights, \
            bbox_weights\
            = self.bbox_head.get_targets([sampling_result], train_cfg)

        label, bbox_targets, pos_gt_bboxes, reg_mask, label_weights, \
            bbox_weights\
            = label[0], bbox_targets[0], pos_gt_bboxes[0], reg_mask[0], \
            label_weights[0], bbox_weights[0]

        expected_label = torch.Tensor([1, 0.9, 0.7, 0.5, 0.3, 0.1, 0, 0, 0])
        expected_bbox_targets = torch.Tensor(
            [-0.27736, 0.22622, -0.0156, 0.0851, -0.8293, 0.0452, -0.1495])
        expected_pos_gt_bboxes = torch.Tensor(
            [7.8417, -0.1405, -1.9652, 1.6122, 3.2838, 1.5331, -2.0835])
        expected_reg_mask = torch.Tensor([1, 1, 1, 0, 0, 0, 0, 0, 0])
        expected_label_weights = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1])
        expected_bbox_weights = torch.Tensor([1, 1, 1, 0, 0, 0, 0, 0, 0])

        assert torch.allclose(label, expected_label, 1e-2)
        assert torch.allclose(bbox_targets, expected_bbox_targets, 1e-2)
        assert torch.allclose(pos_gt_bboxes, expected_pos_gt_bboxes)
        assert torch.all(reg_mask == expected_reg_mask)
        assert torch.allclose(label_weights, expected_label_weights, 1e-2)
        assert torch.allclose(bbox_weights, expected_bbox_weights)

    def test_loss(self):
        train_cfg = Config({
            'cls_pos_thr': 0.75,
            'cls_neg_thr': 0.25,
            'reg_pos_thr': 0.55
        })

        sampling_result = IoUNegPiecewiseSampler(
            1,
            pos_fraction=0.55,
            neg_piece_fractions=[0.8, 0.2],
            neg_iou_piece_thrs=[0.55, 0.1],
            return_iou=True)

        sampling_result.pos_bboxes = torch.Tensor(
            [[8.1517, 0.0384, -1.9496, 1.5271, 4.1131, 1.4879, 1.2076],
             [8.1517, 0.0384, -1.9496, 1.5271, 4.1131, 1.4879, 1.2076],
             [8.1517, 0.0384, -1.9496, 1.5271, 4.1131, 1.4879, 1.2076]])
        sampling_result.pos_gt_bboxes = torch.Tensor(
            [[7.8417, -0.1405, -1.9652, 1.6122, 3.2838, 1.5331, -2.0835],
             [7.8417, -0.1405, -1.9652, 1.6122, 3.2838, 1.5331, -2.0835],
             [7.8417, -0.1405, -1.9652, 1.6122, 3.2838, 1.5331, -2.0835]])
        sampling_result.iou = torch.Tensor([0.8, 0.7, 0.6])

        roi_features_sampled = self.roi_features

        loss = self.bbox_head.loss(roi_features_sampled, [sampling_result],
                                   train_cfg)
        assert loss['loss_cls'].shape == torch.Size([3, 3])
        assert loss['loss_reg'].shape == torch.Size([3, 7])
