# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import BaseDataset
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from mmdet3d.datasets.transforms import ObjectSample
from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class DisableObjectSampleHook(Hook):
    """The hook of disabling augmentations during training.

    Args:
        disable_after_epoch (int): The number of epochs after which
            the ``ObjectSample`` will be closed in the training.
            Defaults to 15.
    """

    def __init__(self, disable_after_epoch: int = 15):
        self.disable_after_epoch = disable_after_epoch
        self._restart_dataloader = False

    def before_train_epoch(self, runner: Runner):
        """Close augmentation.

        Args:
            runner (Runner): The runner.
        """
        epoch = runner.epoch
        train_loader = runner.train_dataloader
        model = runner.model
        # TODO: refactor after mmengine using model wrapper
        if is_model_wrapper(model):
            model = model.module
        if epoch == self.disable_after_epoch:
            runner.logger.info('Disable ObjectSample')
            dataset = runner.train_dataloader.dataset
            # handle dataset wrapper
            if not isinstance(dataset, BaseDataset):
                dataset = dataset.dataset
            for transform in dataset.pipeline.transforms:  # noqa: E501
                if isinstance(transform, ObjectSample):
                    assert hasattr(transform, 'disabled')
                    transform.disabled = True
            # The dataset pipeline cannot be updated when persistent_workers
            # is True, so we need to force the dataloader's multi-process
            # restart. This is a very hacky approach.
            if hasattr(train_loader, 'persistent_workers'
                       ) and train_loader.persistent_workers is True:
                train_loader._DataLoader__initialized = False
                train_loader._iterator = None
                self._restart_dataloader = True
        else:
            # Once the restart is complete, we need to restore
            # the initialization flag.
            if self._restart_dataloader:
                train_loader._DataLoader__initialized = True
