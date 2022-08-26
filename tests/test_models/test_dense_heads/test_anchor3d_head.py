# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine import Config
from mmengine.structures import InstanceData

from mmdet3d import *  # noqa
from mmdet3d.models.dense_heads import Anchor3DHead
from mmdet3d.structures import Box3DMode, LiDARInstance3DBoxes


class TestAnchor3DHead(TestCase):

    def test_anchor3d_head_loss(self):
        """Test anchor head loss when truth is empty and non-empty."""

        cfg = Config(
            dict(
                assigner=[
                    dict(  # for Pedestrian
                        type='Max3DIoUAssigner',
                        iou_calculator=dict(type='BboxOverlapsNearest3D'),
                        pos_iou_thr=0.35,
                        neg_iou_thr=0.2,
                        min_pos_iou=0.2,
                        ignore_iof_thr=-1),
                    dict(  # for Cyclist
                        type='Max3DIoUAssigner',
                        iou_calculator=dict(type='BboxOverlapsNearest3D'),
                        pos_iou_thr=0.35,
                        neg_iou_thr=0.2,
                        min_pos_iou=0.2,
                        ignore_iof_thr=-1),
                    dict(  # for Car
                        type='Max3DIoUAssigner',
                        iou_calculator=dict(type='BboxOverlapsNearest3D'),
                        pos_iou_thr=0.6,
                        neg_iou_thr=0.45,
                        min_pos_iou=0.45,
                        ignore_iof_thr=-1),
                ],
                allowed_border=0,
                pos_weight=-1,
                debug=False))

        anchor3d_head = Anchor3DHead(
            num_classes=3,
            in_channels=512,
            feat_channels=512,
            use_direction_classifier=True,
            anchor_generator=dict(
                type='Anchor3DRangeGenerator',
                ranges=[
                    [0, -40.0, -0.6, 70.4, 40.0, -0.6],
                    [0, -40.0, -0.6, 70.4, 40.0, -0.6],
                    [0, -40.0, -1.78, 70.4, 40.0, -1.78],
                ],
                sizes=[[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]],
                rotations=[0, 1.57],
                reshape_out=False),
            diff_rad_by_sin=True,
            bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
            loss_cls=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(
                type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
            loss_dir=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=0.2),
            train_cfg=cfg)

        # Anchor head expects a multiple levels of features per image
        feats = (torch.rand([1, 512, 200, 176], dtype=torch.float32), )
        (cls_scores, bbox_preds, dir_cls_preds) = anchor3d_head.forward(feats)

        self.assertEqual(cls_scores[0].shape, torch.Size([1, 18, 200, 176]))
        self.assertEqual(bbox_preds[0].shape, torch.Size([1, 42, 200, 176]))
        self.assertEqual(dir_cls_preds[0].shape, torch.Size([1, 12, 200, 176]))

        # # Test that empty ground truth encourages the network to
        # # predict background
        gt_instances = InstanceData()
        gt_bboxes_3d = LiDARInstance3DBoxes(torch.empty((0, 7)))
        gt_labels_3d = torch.tensor([])
        input_metas = dict(sample_idx=1234)
        # fake input_metas
        gt_instances.bboxes_3d = gt_bboxes_3d
        gt_instances.labels_3d = gt_labels_3d

        empty_gt_losses = anchor3d_head.loss_by_feat(cls_scores, bbox_preds,
                                                     dir_cls_preds,
                                                     [gt_instances],
                                                     [input_metas])

        # When there is no truth, the cls loss should be nonzero but
        # there should be no box and dir loss.
        self.assertGreater(empty_gt_losses['loss_cls'][0], 0,
                           'cls loss should be non-zero')
        self.assertEqual(
            empty_gt_losses['loss_bbox'][0], 0,
            'there should be no box loss when there are no true boxes')
        self.assertEqual(
            empty_gt_losses['loss_dir'][0], 0,
            'there should be no dir loss when there are no true dirs')

        # When truth is non-empty then both cls and box loss
        # should be nonzero for random inputs
        gt_instances = InstanceData()
        gt_bboxes_3d = LiDARInstance3DBoxes(
            torch.tensor(
                [[6.4118, -3.4305, -1.7291, 1.7033, 3.4693, 1.6197, -0.9091]],
                dtype=torch.float32))
        gt_labels_3d = torch.tensor([1], dtype=torch.int64)
        gt_instances.bboxes_3d = gt_bboxes_3d
        gt_instances.labels_3d = gt_labels_3d

        gt_losses = anchor3d_head.loss_by_feat(cls_scores, bbox_preds,
                                               dir_cls_preds, [gt_instances],
                                               [input_metas])

        self.assertGreater(gt_losses['loss_cls'][0], 0,
                           'cls loss should be non-zero')
        self.assertGreater(gt_losses['loss_bbox'][0], 0,
                           'box loss should be non-zero')
        self.assertGreater(gt_losses['loss_dir'][0], 0,
                           'dir loss should be none-zero')

    def test_anchor3d_head_predict(self):

        cfg = Config(
            dict(
                use_rotate_nms=True,
                nms_across_levels=False,
                nms_thr=0.01,
                score_thr=0.1,
                min_bbox_size=0,
                nms_pre=100,
                max_num=50))

        anchor3d_head = Anchor3DHead(
            num_classes=3,
            in_channels=512,
            feat_channels=512,
            use_direction_classifier=True,
            anchor_generator=dict(
                type='Anchor3DRangeGenerator',
                ranges=[
                    [0, -40.0, -0.6, 70.4, 40.0, -0.6],
                    [0, -40.0, -0.6, 70.4, 40.0, -0.6],
                    [0, -40.0, -1.78, 70.4, 40.0, -1.78],
                ],
                sizes=[[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]],
                rotations=[0, 1.57],
                reshape_out=False),
            diff_rad_by_sin=True,
            bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
            loss_cls=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(
                type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
            loss_dir=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=0.2),
            test_cfg=cfg)

        feats = (torch.rand([2, 512, 200, 176], dtype=torch.float32), )
        (cls_scores, bbox_preds, dir_cls_preds) = anchor3d_head.forward(feats)
        # fake input_metas
        input_metas = [{
            'sample_idx': 1234,
            'box_type_3d': LiDARInstance3DBoxes,
            'box_mode_3d': Box3DMode.LIDAR
        }, {
            'sample_idx': 2345,
            'box_type_3d': LiDARInstance3DBoxes,
            'box_mode_3d': Box3DMode.LIDAR
        }]
        # test get_boxes
        cls_scores[0] -= 1.5  # too many positive samples may cause cuda oom
        results = anchor3d_head.predict_by_feat(cls_scores, bbox_preds,
                                                dir_cls_preds, input_metas)
        pred_instances = results[0]
        scores_3d = pred_instances.scores_3d

        assert (scores_3d > 0.3).all()
