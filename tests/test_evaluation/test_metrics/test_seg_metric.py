# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch
from mmengine.data import BaseDataElement

from mmdet3d.evaluation.metrics import SegMetric
from mmdet3d.structures import Det3DDataSample, PointData


class TestSegMetric(unittest.TestCase):

    def _demo_mm_inputs(self):
        """Create a superset of inputs needed to run test or train batches."""
        packed_inputs = []
        mm_inputs = dict()
        pts_semantic_mask = torch.Tensor([
            0, 0, 0, 255, 0, 0, 1, 1, 1, 255, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3,
            3, 255
        ])
        ann_info_data = dict(pts_semantic_mask=pts_semantic_mask)
        mm_inputs['data_sample'] = dict(eval_ann_info=ann_info_data)
        packed_inputs.append(mm_inputs)

        return packed_inputs

    def _demo_mm_model_output(self):
        """Create a superset of inputs needed to run test or train batches."""
        pts_semantic_mask = torch.Tensor([
            0, 0, 1, 0, 0, 2, 1, 3, 1, 2, 1, 0, 2, 2, 2, 2, 1, 3, 0, 3, 3, 3, 3
        ])
        pred_pts_seg_data = dict(pts_semantic_mask=pts_semantic_mask)
        data_sample = Det3DDataSample()
        data_sample.pred_pts_seg = PointData(**pred_pts_seg_data)
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
        res = seg_metric.evaluate(1)
        self.assertIsInstance(res, dict)
