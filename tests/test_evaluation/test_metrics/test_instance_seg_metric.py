# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np

from mmdet3d.evaluation.metrics import InstanceSegMetric

seg_valid_class_ids = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34,
                       36, 39)
class_labels = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                'bookshelf', 'picture', 'counter', 'desk', 'curtain',
                'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
                'garbagebin')
dataset_meta = dict(
    seg_valid_class_ids=seg_valid_class_ids, classes=class_labels)


class TestInstanceSegMetric(unittest.TestCase):

    def _demo_mm_model_output(self):
        """Create a superset of inputs needed to run test or train batches."""

        n_points_list = [3300, 3000]
        gt_labels_list = [[0, 0, 0, 0, 0, 0, 14, 14, 2, 2, 2],
                          [13, 13, 2, 1, 3, 3, 0, 0, 0, 0]]
        predictions = []

        for idx, points_num in enumerate(n_points_list):
            points = np.ones(points_num) * -1
            gt = np.ones(points_num)
            info = {}
            pred_mask_3d = []
            pred_score_3d = []
            pred_label_3d = []
            for ii, i in enumerate(gt_labels_list[idx]):
                i = seg_valid_class_ids[i]
                points[ii * 300:(ii + 1) * 300] = ii
                gt[ii * 300:(ii + 1) * 300] = i * 1000 + ii
                pred_mask_3d.append(np.expand_dims(points == ii, axis=0))
                pred_label_3d.append(i)
                pred_score_3d.append(0.99)
            pred_mask_3d = np.concatenate(pred_mask_3d, 0)
            pred_score_3d = np.array(pred_score_3d)
            pred_label_3d = np.array(pred_label_3d)
            info['pred'] = {
                'pred_mask_3d': pred_mask_3d,
                'pred_label_3d': pred_label_3d,
                'pred_score_3d': pred_score_3d,
            }
            info['scene_id'] = idx
            info['groundtruth'] = gt
            predictions.append(info)

        return predictions

    def _demo_mm_model_wrong_output(self):
        """Create a superset of inputs needed to run test or train batches."""

        n_points_list = [3300, 3000]
        gt_labels_list = [[0, 0, 0, 0, 0, 0, 14, 14, 2, 2, 2],
                          [13, 13, 2, 1, 3, 3, 0, 0, 0, 0]]
        predictions = []

        for idx, points_num in enumerate(n_points_list):
            points = np.ones(points_num) * -1
            gt = np.ones(points_num)
            info = {}
            pred_mask_3d = []
            pred_score_3d = []
            pred_label_3d = []
            for ii, i in enumerate(gt_labels_list[idx]):
                i = seg_valid_class_ids[i]
                points[ii * 300:(ii + 1) * 300] = i
                gt[ii * 300:(ii + 1) * 300] = i * 1000 + ii
                pred_mask_3d.append(np.expand_dims(points == i, axis=0))
                pred_label_3d.append(i)
                pred_score_3d.append(0.99)
            pred_mask_3d = np.concatenate(pred_mask_3d, 0)
            pred_score_3d = np.array(pred_score_3d)
            pred_label_3d = np.array(pred_label_3d)
            info['pred'] = {
                'pred_mask_3d': pred_mask_3d,
                'pred_label_3d': pred_label_3d,
                'pred_score_3d': pred_score_3d,
            }
            info['scene_id'] = idx
            info['groundtruth'] = gt
            predictions.append(info)

        return predictions

    def test_evaluate(self):
        data_batch = {}
        predictions = self._demo_mm_model_output()
        instance_seg_metric = InstanceSegMetric(dataset_meta=dataset_meta)
        instance_seg_metric.process(data_batch, predictions)
        res = instance_seg_metric.evaluate(1)
        self.assertIsInstance(res, dict)

        predictions = self._demo_mm_model_wrong_output()
        instance_seg_metric.reset()
        instance_seg_metric.process(data_batch, predictions)
        res = instance_seg_metric.evaluate(1)
        self.assertIsInstance(res, dict)
