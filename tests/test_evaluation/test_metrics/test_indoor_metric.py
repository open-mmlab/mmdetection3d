import unittest
from io import StringIO
from unittest.mock import patch

import numpy as np
import torch

from mmdet3d.evaluation.metrics import IndoorMetric
from mmdet3d.structures import DepthInstance3DBoxes


class TestIndoorMetric(unittest.TestCase):

    @patch('sys.stdout', new_callable=StringIO)
    def test_process(self, stdout):
        indoor_metric = IndoorMetric()
        eval_ann_info = {
            'gt_bboxes_3d':
            DepthInstance3DBoxes(
                torch.tensor([
                    [2.3578, 1.7841, -0.0987, 0.5532, 0.4948, 0.6474, 0.0000],
                    [-0.2773, -2.1403, 0.0615, 0.4786, 0.5170, 0.3842, 0.0000],
                    [0.0259, -2.7954, -0.0157, 0.3869, 0.4361, 0.5229, 0.0000],
                    [-2.3968, 1.1040, 0.0945, 2.5563, 1.5989, 0.9322, 0.0000],
                    [
                        -0.3173, -2.7770, -0.0134, 0.5473, 0.8569, 0.5577,
                        0.0000
                    ],
                    [-2.4882, -1.4437, 0.0987, 1.2199, 0.4859, 0.6461, 0.0000],
                    [-3.4702, -0.1315, 0.2463, 1.3137, 0.8022, 0.4765, 0.0000],
                    [1.9786, 3.0196, -0.0934, 1.6129, 0.5834, 1.4662, 0.0000],
                    [2.3835, 2.2691, -0.1376, 0.5197, 0.5099, 0.6896, 0.0000],
                    [2.5986, -0.5313, 1.4269, 0.0696, 0.2933, 0.3104, 0.0000],
                    [0.4555, -3.1278, -0.0637, 2.0247, 0.1292, 0.2419, 0.0000],
                    [0.4655, -3.1941, 0.3769, 2.1132, 0.3536, 1.9803, 0.0000]
                ])),
            'gt_labels_3d':
            np.array([2, 2, 2, 3, 4, 17, 4, 7, 2, 8, 17, 11])
        }

        pred_instances_3d = dict()
        pred_instances_3d['scores_3d'] = torch.ones(
            len(eval_ann_info['gt_bboxes_3d']))
        pred_instances_3d['bboxes_3d'] = eval_ann_info['gt_bboxes_3d']
        pred_instances_3d['labels_3d'] = torch.Tensor(
            eval_ann_info['gt_labels_3d'])
        pred_dict = dict()
        pred_dict['pred_instances_3d'] = pred_instances_3d
        pred_dict['eval_ann_info'] = eval_ann_info

        indoor_metric.dataset_meta = {
            'classes': ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                        'window', 'bookshelf', 'picture', 'counter', 'desk',
                        'curtain', 'refrigerator', 'showercurtrain', 'toilet',
                        'sink', 'bathtub', 'garbagebin'),
            'box_type_3d':
            'Depth',
        }

        indoor_metric.process({}, [pred_dict])

        eval_results = indoor_metric.evaluate(1)
        for v in eval_results.values():
            # map == 1
            self.assertEqual(1, v)
