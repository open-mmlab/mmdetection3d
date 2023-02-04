# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor
from mmdet3d.structures import Det3DDataSample


class TestDet3DDataPreprocessor(TestCase):

    def test_init(self):
        # test mean is None
        processor = Det3DDataPreprocessor()
        self.assertTrue(not hasattr(processor, 'mean'))
        self.assertTrue(processor._enable_normalize is False)

        # test mean is not None
        processor = Det3DDataPreprocessor(mean=[0, 0, 0], std=[1, 1, 1])
        self.assertTrue(hasattr(processor, 'mean'))
        self.assertTrue(hasattr(processor, 'std'))
        self.assertTrue(processor._enable_normalize)

        # please specify both mean and std
        with self.assertRaises(AssertionError):
            Det3DDataPreprocessor(mean=[0, 0, 0])

        # bgr2rgb and rgb2bgr cannot be set to True at the same time
        with self.assertRaises(AssertionError):
            Det3DDataPreprocessor(bgr_to_rgb=True, rgb_to_bgr=True)

    def test_forward(self):
        processor = Det3DDataPreprocessor(mean=[0, 0, 0], std=[1, 1, 1])

        points = torch.randn((5000, 3))
        image = torch.randint(0, 256, (3, 11, 10)).float()
        inputs_dict = dict(points=[points], img=[image])

        data = {'inputs': inputs_dict, 'data_samples': [Det3DDataSample()]}
        out_data = processor(data)
        batch_inputs, batch_data_samples = out_data['inputs'], out_data[
            'data_samples']

        self.assertEqual(batch_inputs['imgs'].shape, (1, 3, 11, 10))
        self.assertEqual(len(batch_inputs['points']), 1)
        self.assertEqual(len(batch_data_samples), 1)

        # test image channel_conversion
        processor = Det3DDataPreprocessor(
            mean=[0., 0., 0.], std=[1., 1., 1.], bgr_to_rgb=True)
        out_data = processor(data)
        batch_inputs, batch_data_samples = out_data['inputs'], out_data[
            'data_samples']
        self.assertEqual(batch_inputs['imgs'].shape, (1, 3, 11, 10))
        self.assertEqual(len(batch_data_samples), 1)

        # test image padding
        data = {
            'inputs': {
                'points': [torch.randn((5000, 3)),
                           torch.randn((5000, 3))],
                'img': [
                    torch.randint(0, 256, (3, 10, 11)),
                    torch.randint(0, 256, (3, 9, 14))
                ]
            }
        }
        processor = Det3DDataPreprocessor(
            mean=[0., 0., 0.], std=[1., 1., 1.], bgr_to_rgb=True)
        out_data = processor(data)
        batch_inputs, batch_data_samples = out_data['inputs'], out_data[
            'data_samples']
        self.assertEqual(batch_inputs['imgs'].shape, (2, 3, 10, 14))
        self.assertIsNone(batch_data_samples)

        # test pad_size_divisor
        data = {
            'inputs': {
                'points': [torch.randn((5000, 3)),
                           torch.randn((5000, 3))],
                'img': [
                    torch.randint(0, 256, (3, 10, 11)),
                    torch.randint(0, 256, (3, 9, 24))
                ]
            },
            'data_samples': [Det3DDataSample()] * 2
        }
        processor = Det3DDataPreprocessor(
            mean=[0., 0., 0.], std=[1., 1., 1.], pad_size_divisor=5)
        out_data = processor(data)
        batch_inputs, batch_data_samples = out_data['inputs'], out_data[
            'data_samples']
        self.assertEqual(batch_inputs['imgs'].shape, (2, 3, 10, 25))
        self.assertEqual(len(batch_data_samples), 2)
        for data_sample, expected_shape in zip(batch_data_samples, [(10, 15),
                                                                    (10, 25)]):
            self.assertEqual(data_sample.pad_shape, expected_shape)
