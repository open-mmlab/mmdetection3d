# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch
from mmengine.structures import BaseDataElement

from mmdet3d.evaluation.metrics import SegMetric
from mmdet3d.structures import Det3DDataSample, PointData


class TestSegMetric(unittest.TestCase):

    def _demo_mm_model_output(self):
        """Create a superset of inputs needed to run test or train batches."""
        pred_pts_semantic_mask = torch.LongTensor([
            0, 0, 1, 0, 0, 2, 1, 3, 1, 2, 1, 0, 2, 2, 2, 2, 1, 3, 0, 3, 3, 3, 3
        ])
        pred_pts_seg_data = dict(pts_semantic_mask=pred_pts_semantic_mask)
        data_sample = Det3DDataSample()
        data_sample.pred_pts_seg = PointData(**pred_pts_seg_data)

        gt_pts_semantic_mask = torch.LongTensor(([
            0, 0, 0, 4, 0, 0, 1, 1, 1, 4, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4
        ]))
        gt_pts_seg_data = dict(pts_semantic_mask=gt_pts_semantic_mask)
        data_sample.gt_pts_seg = PointData(**gt_pts_seg_data)

        batch_data_samples = [data_sample]

        predictions = []
        for pred in batch_data_samples:
            if isinstance(pred, BaseDataElement):
                pred = pred.to_dict()
            predictions.append(pred)

        return predictions

    def test_evaluate(self):
        data_batch = {}
        predictions = self._demo_mm_model_output()
        dataset_meta = dict(classes=('car', 'bicyle', 'motorcycle', 'truck'))
        seg_metric = SegMetric(ignore_index=len(dataset_meta['classes']))
        seg_metric.dataset_meta = dataset_meta
        seg_metric.process(data_batch, predictions)
        res = seg_metric.evaluate(1)
        self.assertIsInstance(res, dict)
