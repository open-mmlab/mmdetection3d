# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from mmengine.dataset import BaseDataset
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class DisableAugHook(Hook):
    """The hook of disabling augmentations during training.

    Args:
        disable_after_epoch (int): The number of epochs after which
            the data augmentation will be closed in the training.
            Defaults to 15.
        disable_aug_list (list): the list of data augmentation will
            be closed in the training. Defaults to [].
    """

    def __init__(self,
                 disable_after_epoch: int = 15,
                 disable_aug_list: List = []):
        self.disable_after_epoch = disable_after_epoch
        self.disable_aug_list = disable_aug_list
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

            dataset = runner.train_dataloader.dataset
            # handle dataset wrapper
            if not isinstance(dataset, BaseDataset):
                dataset = dataset.dataset
            new_transforms = []
            for transform in dataset.pipeline.transforms:  # noqa: E501
                if transform.__class__.__name__ not in self.disable_aug_list:
                    new_transforms.append(transform)
                else:
                    runner.logger.info(
                        f'Disable {transform.__class__.__name__}')
            dataset.pipeline.transforms = new_transforms
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
