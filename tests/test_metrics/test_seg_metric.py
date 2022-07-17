# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch
from mmengine.data import BaseDataElement

from mmdet3d.core.data_structures import Det3DDataSample, PointData
from mmdet3d.metrics import SegMetric


class TestSegMetric(unittest.TestCase):

    def _demo_mm_inputs(self):
        """Create a superset of inputs needed to run test or train batches."""
        packed_inputs = []
        mm_inputs = dict()
        data_sample = Det3DDataSample()
        pts_semantic_mask = torch.Tensor([
            0, 0, 0, 255, 0, 0, 1, 1, 1, 255, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3,
            3, 255
        ])
        gt_pts_seg_data = dict(pts_semantic_mask=pts_semantic_mask)
        data_sample.gt_pts_seg = PointData(**gt_pts_seg_data)
        mm_inputs['data_sample'] = data_sample.to_dict()
        packed_inputs.append(mm_inputs)

        return packed_inputs

    def _demo_mm_model_output(self):
        """Create a superset of inputs needed to run test or train batches."""
        results_dict = dict()
        pts_seg_pred = torch.Tensor([
            0, 0, 1, 0, 0, 2, 1, 3, 1, 2, 1, 0, 2, 2, 2, 2, 1, 3, 0, 3, 3, 3, 3
        ])
        results_dict['pts_semantic_mask'] = pts_seg_pred
        data_sample = Det3DDataSample()
        data_sample['pred_pts_seg'] = results_dict
        batch_data_samples = [data_sample]

        predictions = []
        for pred in batch_data_samples:
            if isinstance(pred, BaseDataElement):
                pred = pred.to_dict()
            predictions.append(pred)

        return predictions

    def test_evaluate(self):
        data_batch = self._demo_mm_inputs()
        predictions = self._demo_mm_model_output()
        label2cat = {
            0: 'car',
            1: 'bicycle',
            2: 'motorcycle',
            3: 'truck',
        }
        dataset_meta = dict(label2cat=label2cat, ignore_index=255)
        seg_metric = SegMetric()
        seg_metric.dataset_meta = dataset_meta
        seg_metric.process(data_batch, predictions)
        res = seg_metric.evaluate(0)
        self.assertIsInstance(res, dict)
