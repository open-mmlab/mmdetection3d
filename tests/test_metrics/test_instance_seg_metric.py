# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np
import torch
from mmengine.data import BaseDataElement

from mmdet3d.core.data_structures import Det3DDataSample, PointData
from mmdet3d.metrics import InstanceSegMetric


class TestInstanceSegMetric(unittest.TestCase):

    def _demo_mm_inputs(self):
        """Create a superset of inputs needed to run test or train batches."""
        packed_inputs = []
        results_dict = dict()
        mm_inputs = dict()
        n_points = 3300
        gt_labels = [0, 0, 0, 0, 0, 0, 14, 14, 2, 1]
        gt_instance_mask = np.ones(n_points, dtype=np.int) * -1
        gt_semantic_mask = np.ones(n_points, dtype=np.int) * -1
        for i, gt_label in enumerate(gt_labels):
            begin = i * 300
            end = begin + 300
            gt_instance_mask[begin:end] = i
            gt_semantic_mask[begin:end] = gt_label

        results_dict['pts_instance_mask'] = torch.tensor(gt_instance_mask)
        results_dict['pts_semantic_mask'] = torch.tensor(gt_semantic_mask)

        data_sample = Det3DDataSample()
        data_sample.gt_pts_seg = PointData(**results_dict)
        mm_inputs['data_sample'] = data_sample.to_dict()
        packed_inputs.append(mm_inputs)

        return packed_inputs

    def _demo_mm_model_output(self):
        """Create a superset of inputs needed to run test or train batches."""
        results_dict = dict()
        n_points = 3300
        gt_labels = [0, 0, 0, 0, 0, 0, 14, 14, 2, 1]
        pred_instance_mask = np.ones(n_points, dtype=np.int) * -1
        labels = []
        scores = []
        for i, gt_label in enumerate(gt_labels):
            begin = i * 300
            end = begin + 300
            pred_instance_mask[begin:end] = i
            labels.append(gt_label)
            scores.append(.99)

        results_dict['pts_instance_mask'] = torch.tensor(pred_instance_mask)
        results_dict['instance_labels'] = torch.tensor(labels)
        results_dict['instance_scores'] = torch.tensor(scores)
        data_sample = Det3DDataSample()
        data_sample.pred_pts_seg = PointData(**results_dict)
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
        valid_class_ids = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33,
                           34, 36, 39)
        class_labels = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                        'window', 'bookshelf', 'picture', 'counter', 'desk',
                        'curtain', 'refrigerator', 'showercurtrain', 'toilet',
                        'sink', 'bathtub', 'garbagebin')
        dataset_meta = dict(
            VALID_CLASS_IDS=valid_class_ids, CLASSES=class_labels)
        instance_seg_metric = InstanceSegMetric()
        instance_seg_metric.dataset_meta = dataset_meta
        instance_seg_metric.process(data_batch, predictions)
        res = instance_seg_metric.evaluate(6)
        self.assertIsInstance(res, dict)
