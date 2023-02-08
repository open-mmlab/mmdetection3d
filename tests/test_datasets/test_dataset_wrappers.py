# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import numpy as np
import pytest
from mmcv.transforms.base import BaseTransform
from mmengine.structures import InstanceData

from mmdet3d.datasets import CBGSDataset, NuScenesDataset
from mmdet3d.registry import DATASETS, TRANSFORMS
from mmdet3d.structures import Det3DDataSample


def is_equal(dict_a, dict_b):
    for key in dict_a:
        if key not in dict_b:
            return False
        if isinstance(dict_a[key], dict):
            return is_equal(dict_a[key], dict_b[key])
        elif isinstance(dict_a[key], np.ndarray):
            if not (dict_a[key] == dict_b[key]).any():
                return False
        else:
            if not (dict_a[key] == dict_b[key]):
                return False
    return True


@TRANSFORMS.register_module()
class Identity(BaseTransform):

    def transform(self, info):
        packed_input = dict(data_samples=Det3DDataSample())
        if 'ann_info' in info:
            packed_input['data_samples'].gt_instances_3d = InstanceData()
            packed_input['data_samples'].gt_instances_3d.labels_3d = info[
                'ann_info']['gt_labels_3d']
        return packed_input


@DATASETS.register_module()
class CustomDataset(NuScenesDataset):
    pass


class TestCBGSDataset:

    def setup(self):
        dataset = NuScenesDataset
        self.dataset = dataset(
            data_root=osp.join(osp.dirname(__file__), '../data/nuscenes'),
            ann_file='nus_info.pkl',
            data_prefix=dict(
                pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP'),
            pipeline=[dict(type=Identity)])

        self.sample_indices = [0, 0, 1, 1, 1]
        # test init
        self.cbgs_datasets = CBGSDataset(dataset=self.dataset)
        self.cbgs_datasets.sample_indices = self.sample_indices

    def test_init(self):
        # Test build dataset from cfg
        dataset_cfg = dict(
            type=CustomDataset,
            data_root=osp.join(osp.dirname(__file__), '../data/nuscenes'),
            ann_file='nus_info.pkl',
            data_prefix=dict(
                pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP'),
            pipeline=[dict(type=Identity)])
        cbgs_datasets = CBGSDataset(dataset=dataset_cfg)
        cbgs_datasets.sample_indices = self.sample_indices
        cbgs_datasets.dataset.pipeline = self.dataset.pipeline
        assert len(cbgs_datasets) == len(self.cbgs_datasets)
        for i in range(len(cbgs_datasets)):
            assert is_equal(
                cbgs_datasets.get_data_info(i),
                self.cbgs_datasets.get_data_info(i))
            assert (cbgs_datasets[i]['data_samples'].gt_instances_3d.labels_3d
                    == self.cbgs_datasets[i]
                    ['data_samples'].gt_instances_3d.labels_3d).any()

        with pytest.raises(TypeError):
            CBGSDataset(dataset=[0])

    def test_full_init(self):
        self.cbgs_datasets.full_init()
        self.cbgs_datasets.sample_indices = self.sample_indices
        assert len(self.cbgs_datasets) == len(self.sample_indices)
        # Reinit `sample_indices`
        self.cbgs_datasets._fully_initialized = False
        self.cbgs_datasets.sample_indices = self.sample_indices
        assert len(self.cbgs_datasets) != len(self.sample_indices)

        with pytest.raises(NotImplementedError):
            self.cbgs_datasets.get_subset_(1)

        with pytest.raises(NotImplementedError):
            self.cbgs_datasets.get_subset(1)

    def test_metainfo(self):
        assert self.cbgs_datasets.metainfo == self.dataset.metainfo

    def test_length(self):
        assert len(self.cbgs_datasets) == len(self.sample_indices)

    def test_getitem(self):
        for i in range(len(self.sample_indices)):
            assert (self.cbgs_datasets[i]['data_samples'].gt_instances_3d.
                    labels_3d == self.dataset[self.sample_indices[i]]
                    ['data_samples'].gt_instances_3d.labels_3d).any()

    def test_get_data_info(self):
        for i in range(len(self.sample_indices)):
            assert is_equal(
                self.cbgs_datasets.get_data_info(i),
                self.dataset.get_data_info(self.sample_indices[i]))

    def test_get_cat_ids(self):
        for i in range(len(self.sample_indices)):
            assert self.cbgs_datasets.get_cat_ids(
                i) == self.dataset.get_cat_ids(self.sample_indices[i])
