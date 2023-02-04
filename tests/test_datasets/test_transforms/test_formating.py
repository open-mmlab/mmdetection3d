# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch
from mmengine.testing import assert_allclose

from mmdet3d.datasets.transforms.formating import Pack3DDetInputs
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.testing import create_data_info_after_loading


class TestPack3DDetInputs(unittest.TestCase):

    def test_packinputs(self):
        ori_data_info = create_data_info_after_loading()
        pack_input = Pack3DDetInputs(
            keys=['points', 'gt_labels_3d', 'gt_bboxes_3d'])
        packed_results = pack_input(ori_data_info)
        inputs = packed_results['inputs']

        # annotations
        gt_instances = packed_results['data_samples'].gt_instances_3d
        self.assertIn('points', inputs)
        self.assertIsInstance(inputs['points'], torch.Tensor)
        assert_allclose(inputs['points'].sum(), torch.tensor(13062.6436))
        # assert to_tensor
        self.assertIsInstance(inputs['points'], torch.Tensor)
        self.assertIn('labels_3d', gt_instances)
        assert_allclose(gt_instances.labels_3d, torch.tensor([1]))
        # assert to_tensor
        self.assertIsInstance(gt_instances.labels_3d, torch.Tensor)

        self.assertIn('bboxes_3d', gt_instances)
        self.assertIsInstance(gt_instances.bboxes_3d, LiDARInstance3DBoxes)
        assert_allclose(gt_instances.bboxes_3d.tensor.sum(),
                        torch.tensor(7.2650))
