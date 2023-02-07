# Copyright (c) OpenMMLab. All rights reserved.
import random
from unittest import TestCase

import numpy as np
import pytest
import torch

from mmdet3d.structures import PointData


class TestPointData(TestCase):

    def setup_data(self):
        metainfo = dict(sample_idx=random.randint(0, 100))
        points = torch.rand((5, 3))
        point_data = PointData(metainfo=metainfo, points=points)
        return point_data

    def test_set_data(self):
        point_data = self.setup_data()

        # test set '_metainfo_fields' or '_data_fields'
        with self.assertRaises(AttributeError):
            point_data._metainfo_fields = 1
        with self.assertRaises(AttributeError):
            point_data._data_fields = 1

        point_data.keypoints = torch.rand((5, 2))
        assert 'keypoints' in point_data

    def test_getitem(self):
        point_data = PointData()
        # length must be greater than 0
        with self.assertRaises(IndexError):
            point_data[1]

        point_data = self.setup_data()
        assert len(point_data) == 5
        slice_point_data = point_data[:2]
        assert len(slice_point_data) == 2
        slice_point_data = point_data[1]
        assert len(slice_point_data) == 1
        # assert the index should in 0 ~ len(point_data) - 1
        with pytest.raises(IndexError):
            point_data[5]

        # isinstance(str, slice, int, torch.LongTensor, torch.BoolTensor)
        item = torch.Tensor([1, 2, 3, 4])  # float
        with pytest.raises(AssertionError):
            point_data[item]

        # when input is a bool tensor, The shape of
        # the input at index 0 should equal to
        # the value length in instance_data_field
        with pytest.raises(AssertionError):
            point_data[item.bool()]

        # test LongTensor
        long_tensor = torch.randint(5, (2, ))
        long_index_point_data = point_data[long_tensor]
        assert len(long_index_point_data) == len(long_tensor)

        # test BoolTensor
        bool_tensor = torch.rand(5) > 0.5
        bool_index_point_data = point_data[bool_tensor]
        assert len(bool_index_point_data) == bool_tensor.sum()
        bool_tensor = torch.rand(5) > 1
        empty_point_data = point_data[bool_tensor]
        assert len(empty_point_data) == bool_tensor.sum()

        # test list index
        list_index = [1, 2]
        list_index_point_data = point_data[list_index]
        assert len(list_index_point_data) == len(list_index)

        # test list bool
        list_bool = [True, False, True, False, False]
        list_bool_point_data = point_data[list_bool]
        assert len(list_bool_point_data) == 2

        # test numpy
        long_numpy = np.random.randint(5, size=2)
        long_numpy_point_data = point_data[long_numpy]
        assert len(long_numpy_point_data) == len(long_numpy)

        bool_numpy = np.random.rand(5) > 0.5
        bool_numpy_point_data = point_data[bool_numpy]
        assert len(bool_numpy_point_data) == bool_numpy.sum()

    def test_len(self):
        point_data = self.setup_data()
        assert len(point_data) == 5
        point_data = PointData()
        assert len(point_data) == 0
