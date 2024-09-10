# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import Mock

from mmengine.dataset import BaseDataset

from mmdet3d.datasets.transforms import ObjectSample
from mmdet3d.engine.hooks import DisableObjectSampleHook


class TestDisableObjectSampleHook(TestCase):

    runner = Mock()
    runner.train_dataloader = Mock()
    runner.train_dataloader.dataset = Mock(spec=BaseDataset)
    runner.train_dataloader.dataset.pipeline = Mock()
    runner.train_dataloader._DataLoader__initialized = True
    runner.train_dataloader.dataset.pipeline.transforms = [
        ObjectSample(
            db_sampler=dict(
                data_root='tests/data/waymo/kitti_format',
                info_path=  # noqa
                'tests/data/waymo/kitti_format/waymo_dbinfos_train.pkl',
                rate=1.0,
                prepare=dict(
                    filter_by_difficulty=[-1],
                    filter_by_min_points=dict(Car=5)),
                classes=['Car'],
                sample_groups=dict(Car=15),
            ))
    ]

    def test_is_model_wrapper_and_persistent_workers_on(self):
        self.runner.train_dataloader.dataset.pipeline.transforms[
            0].disabled = False
        self.runner.train_dataloader.persistent_workers = True
        hook = DisableObjectSampleHook(disable_after_epoch=15)
        self.runner.epoch = 14
        hook.before_train_epoch(self.runner)
        self.assertFalse(self.runner.train_dataloader.dataset.pipeline.
                         transforms[0].disabled)  # noqa: E501

        self.runner.epoch = 15
        hook.before_train_epoch(self.runner)
        self.assertTrue(self.runner.train_dataloader.dataset.pipeline.
                        transforms[0].disabled)  # noqa: E501
        self.assertTrue(hook._restart_dataloader)
        self.assertFalse(self.runner.train_dataloader._DataLoader__initialized)

        self.runner.epoch = 16
        hook.before_train_epoch(self.runner)
        self.assertTrue(self.runner.train_dataloader._DataLoader__initialized)
        self.assertTrue(self.runner.train_dataloader.dataset.pipeline.
                        transforms[0].disabled)  # noqa: E501

    def test_not_model_wrapper_and_persistent_workers_off(self):
        self.runner.train_dataloader.dataset.pipeline.transforms[
            0].disabled = False
        self.runner.train_dataloader.persistent_workers = False
        hook = DisableObjectSampleHook(disable_after_epoch=15)
        self.runner.epoch = 14
        hook.before_train_epoch(self.runner)
        self.assertFalse(self.runner.train_dataloader.dataset.pipeline.
                         transforms[0].disabled)  # noqa: E501

        self.runner.epoch = 15
        hook.before_train_epoch(self.runner)
        self.assertTrue(self.runner.train_dataloader.dataset.pipeline.
                        transforms[0].disabled)  # noqa: E501
        self.assertFalse(hook._restart_dataloader)
        self.assertTrue(self.runner.train_dataloader._DataLoader__initialized)

        self.runner.epoch = 16
        hook.before_train_epoch(self.runner)
        self.assertTrue(self.runner.train_dataloader._DataLoader__initialized)
        self.assertTrue(self.runner.train_dataloader.dataset.pipeline.
                        transforms[0].disabled)  # noqa: E501
