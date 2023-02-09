# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np
import torch
from mmengine.structures import BaseDataElement

from mmdet3d.evaluation.metrics import PanopticSegMetric
from mmdet3d.structures import Det3DDataSample, PointData


class TestPanopticSegMetric(unittest.TestCase):

    def _demo_mm_model_output(self):
        """Create a superset of inputs needed to run test or train batches."""
        # generate ground truth and prediction
        semantic_preds = []
        instance_preds = []
        gt_semantic = []
        gt_instance = []

        # some ignore stuff
        num_ignore = 50
        semantic_preds.extend([0 for i in range(num_ignore)])
        instance_preds.extend([0 for i in range(num_ignore)])
        gt_semantic.extend([0 for i in range(num_ignore)])
        gt_instance.extend([0 for i in range(num_ignore)])

        # grass segment
        num_grass = 50
        num_grass_pred = 40  # rest is sky
        semantic_preds.extend([1 for i in range(num_grass_pred)])  # grass
        semantic_preds.extend([2 for i in range(num_grass - num_grass_pred)
                               ])  # sky
        instance_preds.extend([0 for i in range(num_grass)])
        gt_semantic.extend([1 for i in range(num_grass)])  # grass
        gt_instance.extend([0 for i in range(num_grass)])

        # sky segment
        num_sky = 50
        num_sky_pred = 40  # rest is grass
        semantic_preds.extend([2 for i in range(num_sky_pred)])  # sky
        semantic_preds.extend([1 for i in range(num_sky - num_sky_pred)
                               ])  # grass
        instance_preds.extend([0 for i in range(num_sky)])  # first instance
        gt_semantic.extend([2 for i in range(num_sky)])  # sky
        gt_instance.extend([0 for i in range(num_sky)])  # first instance

        # wrong dog as person prediction
        num_dog = 50
        num_person = num_dog
        semantic_preds.extend([3 for i in range(num_person)])
        instance_preds.extend([35 for i in range(num_person)])
        gt_semantic.extend([4 for i in range(num_dog)])
        gt_instance.extend([22 for i in range(num_dog)])

        # two persons in prediction, but three in gt
        num_person = 50
        semantic_preds.extend([3 for i in range(6 * num_person)])
        instance_preds.extend([8 for i in range(4 * num_person)])
        instance_preds.extend([95 for i in range(2 * num_person)])
        gt_semantic.extend([3 for i in range(6 * num_person)])
        gt_instance.extend([33 for i in range(3 * num_person)])
        gt_instance.extend([42 for i in range(num_person)])
        gt_instance.extend([11 for i in range(2 * num_person)])

        # gt and pred to numpy
        semantic_preds = np.array(semantic_preds, dtype=int).reshape(1, -1)
        instance_preds = np.array(instance_preds, dtype=int).reshape(1, -1)
        gt_semantic = np.array(gt_semantic, dtype=int).reshape(1, -1)
        gt_instance = np.array(gt_instance, dtype=int).reshape(1, -1)

        pred_pts_semantic_mask = torch.Tensor(semantic_preds)
        pred_pts_instance_mask = torch.Tensor(instance_preds)
        pred_pts_seg_data = dict(
            pts_semantic_mask=pred_pts_semantic_mask,
            pts_instance_mask=pred_pts_instance_mask)
        data_sample = Det3DDataSample()
        data_sample.pred_pts_seg = PointData(**pred_pts_seg_data)

        ann_info_data = dict(
            pts_semantic_mask=gt_semantic, pts_instance_mask=gt_instance)
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

        classes = ['unlabeled', 'person', 'dog', 'grass', 'sky']
        label2cat = {
            0: 'unlabeled',
            1: 'person',
            2: 'dog',
            3: 'grass',
            4: 'sky',
        }

        ignore_index = [0]  # only ignore ignore class
        min_num_points = 1  # for this example we care about all points
        id_offset = 2**16

        dataset_meta = dict(
            label2cat=label2cat, ignore_index=ignore_index, classes=classes)
        panoptic_seg_metric = PanopticSegMetric(
            thing_class_inds=[0, 1],
            stuff_class_inds=[2, 3],
            min_num_points=min_num_points,
            id_offset=id_offset,
        )
        panoptic_seg_metric.dataset_meta = dataset_meta
        panoptic_seg_metric.process(data_batch, predictions)
        res = panoptic_seg_metric.evaluate(1)
        self.assertIsInstance(res, dict)
