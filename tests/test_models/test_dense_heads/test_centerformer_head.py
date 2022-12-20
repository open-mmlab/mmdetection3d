# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine import Config
from mmengine.structures import InstanceData

from mmdet3d.registry import MODELS
from mmdet3d.structures import Box3DMode, Det3DDataSample, LiDARInstance3DBoxes
from mmdet3d.utils import register_all_modules


class TestCenterFormerHead(TestCase):

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(
        'configs/centerformer/centerformer_voxel01_second_4xb4-cyclic-20e_waymoD5-3d-3class.py'  # noqa
    ).model
    cfg.bbox_head.update(train_cfg=cfg.train_cfg)
    cfg.bbox_head.update(test_cfg=cfg.test_cfg)
    centerformer_head = MODELS.build(cfg.bbox_head)

    feats = (torch.randn([1, 256, 188, 188], dtype=torch.float32).cuda(),
             torch.randn([1, 256, 94,
                          94]).cuda(), torch.randn([1, 256, 376, 376]).cuda())

    gt_instances = InstanceData()
    gt_bboxes_3d = LiDARInstance3DBoxes(torch.rand([3, 7]).cuda(), box_dim=7)
    gt_labels_3d = torch.randint(0, 10, [3]).cuda()
    gt_instances.bboxes_3d = gt_bboxes_3d
    gt_instances.labels_3d = gt_labels_3d

    data_sample = Det3DDataSample(
        metainfo=dict(box_type_3d=LiDARInstance3DBoxes))
    data_sample.gt_instances_3d = gt_instances

    def test_centerformer_head_loss(self):
        """Test centerformer head loss when truth is empty and non-empty."""

        if torch.cuda.is_available():
            centerformer_head = self.centerformer_head.cuda()

            with torch.no_grad():
                losses = centerformer_head.loss(self.feats, [self.data_sample])
            self.assertGreater(losses['task0.loss_bbox'], 0,
                               'bbox loss should be non-zero')
            self.assertGreater(losses['task0.loss_heatmap'], 0,
                               'heatmap loss should be non-zero')
            self.assertGreater(losses['task0.loss_corner'], 0,
                               'corner loss should be non-zero')
            self.assertGreater(losses['task0.loss_iou'], 0,
                               'iou loss should be non-zero')

    def test_centerformer_head_predict(self):
        if torch.cuda.is_available():
            centerformer_head = self.centerformer_head.cuda()
            with torch.no_grad():
                preds = centerformer_head.predict(self.feats,
                                                  [self.data_sample])
            self.assertEqual(len(preds), 1)
            self.assertIn('bboxes_3d', preds[0])
            self.assertIn('scores_3d', preds[0])
            self.assertIn('labels_3d', preds[0])
