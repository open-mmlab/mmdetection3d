# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class WorkerInitFnHook(Hook):
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
        train_loader.worker_init_fn = lambda worker_id: np.random.seed(
            self.seed + (epoch) * self.num_workers + worker_id)
