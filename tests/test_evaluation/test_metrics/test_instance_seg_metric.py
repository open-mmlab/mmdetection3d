# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np
import torch
from mmengine.structures import BaseDataElement

from mmdet3d.evaluation.metrics import InstanceSegMetric
from mmdet3d.structures import Det3DDataSample, PointData


class TestInstanceSegMetric(unittest.TestCase):

    def _demo_mm_model_output(self):
        """Create a superset of inputs needed to run test or train batches."""

        n_points = 3300
        gt_labels = [0, 0, 0, 0, 0, 0, 14, 14, 2, 1]
        gt_instance_mask = np.ones(n_points, dtype=np.int64) * -1
        gt_semantic_mask = np.ones(n_points, dtype=np.int64) * -1
        for i, gt_label in enumerate(gt_labels):
            begin = i * 300
            end = begin + 300
            gt_instance_mask[begin:end] = i
            gt_semantic_mask[begin:end] = gt_label

        ann_info_data = dict()
        ann_info_data['pts_instance_mask'] = torch.tensor(gt_instance_mask)
        ann_info_data['pts_semantic_mask'] = torch.tensor(gt_semantic_mask)

        results_dict = dict()
        n_points = 3300
        gt_labels = [0, 0, 0, 0, 0, 0, 14, 14, 2, 1]
        pred_instance_mask = np.ones(n_points, dtype=np.int64) * -1
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
        data_sample.eval_ann_info = ann_info_data
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
        seg_valid_class_ids = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28,
                               33, 34, 36, 39)
        class_labels = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                        'window', 'bookshelf', 'picture', 'counter', 'desk',
                        'curtain', 'refrigerator', 'showercurtrain', 'toilet',
                        'sink', 'bathtub', 'garbagebin')
        dataset_meta = dict(
            seg_valid_class_ids=seg_valid_class_ids, classes=class_labels)
        instance_seg_metric = InstanceSegMetric()
        instance_seg_metric.dataset_meta = dataset_meta
        instance_seg_metric.process(data_batch, predictions)
        res = instance_seg_metric.evaluate(1)
        self.assertIsInstance(res, dict)
