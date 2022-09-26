# Copyright (c) OpenMMLab. All rights reserved.

from mmengine.hooks import Hook

from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class BenchmarkHook(Hook):
    """A hook that logs the time spent during iteration.

    E.g. ``data_time`` for loading data and ``time`` for a model train step.
    """

    priority = 'NORMAL'

    def after_train_epoch(self, runner) -> None:
        """Save the checkpoint and synchronize buffers after each epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        message_hub = runner.message_hub
        max_iter_num = len(runner.train_dataloader)
        speed = message_hub.get_scalar('train/time').mean(max_iter_num - 50)
        runner.logger.info(
            f'Training speed of epoch {runner.epoch + 1} is {speed} s/iter')
