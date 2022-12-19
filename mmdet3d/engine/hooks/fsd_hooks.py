from mmengine.hooks import Hook
from mmengine.hooks.hook import DATA_BATCH

from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class DisableAugmentationHook(Hook):
    """Switch the mode of YOLOX during training.
    This hook turns off the mosaic and mixup data augmentation and switches
    to use L1 loss in bbox_head.
    Args:
        num_last_epochs (int): The number of latter epochs in the end of the
            training to close the data augmentation and switch to L1 loss.
            Default: 15.
       skip_type_keys (list[str], optional): Sequence of type string to be
            skip pipeline. Default: ('Mosaic', 'RandomAffine', 'MixUp')
    """

    def __init__(self,
                 num_last_epochs=10,
                 skip_type_keys=('ObjectSample')):
        self.num_last_epochs = num_last_epochs
        self.skip_type_keys = skip_type_keys
        self._restart_dataloader = False

    def before_train_epoch(self, runner):
        epoch = runner.epoch # begin from 0
        train_loader = runner.train_dataloader
        if epoch == runner.max_epochs - self.num_last_epochs:
            runner.logger.info(f'Disable augmentations: {self.skip_type_keys}')
            # The dataset pipeline cannot be updated when persistent_workers
            # is True, so we need to force the dataloader's multi-process
            # restart. This is a very hacky approach.
            train_loader.dataset.dataset.update_skip_type_keys(self.skip_type_keys)
            if hasattr(train_loader, 'persistent_workers'
                       ) and train_loader.persistent_workers is True:

                train_loader._DataLoader__initialized = False
                train_loader._iterator = None
                self._restart_dataloader = True
                print('has persistent workers')
        else:
            # Once the restart is complete, we need to restore
            # the initialization flag.
            if self._restart_dataloader:
                train_loader._DataLoader__initialized = True

@HOOKS.register_module()
class EnableFSDDetectionHook(Hook):

    def __init__(self,
                 enable_after_epoch=1,
                 ):
        self.enable_after_epoch = enable_after_epoch

    def before_train_epoch(self, runner):
        epoch = runner.epoch # begin from 0
        if epoch == self.enable_after_epoch:
            runner.logger.info(f'Enable FSD Detection from now.')
            runner.model.module.runtime_info['enable_detection'] = True

@HOOKS.register_module()
class EnableFSDDetectionHookIter(Hook):

    def __init__(self,
                 enable_after_iter=5000,
                 threshold_buffer=0,
                 buffer_iter=2000,
                 ):
        self.enable_after_iter = enable_after_iter
        self.buffer_iter = buffer_iter
        self.delta = threshold_buffer / buffer_iter
        self.threshold_buffer = threshold_buffer

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: DATA_BATCH = None):
        cur_iter = runner.iter # begin from 0
        if cur_iter == self.enable_after_iter:
            runner.logger.info(f'Enable FSD Detection from now.')
        if cur_iter >= self.enable_after_iter: # keep the sanity when resuming model
            runner.model.module.runtime_info['enable_detection'] = True
        if self.threshold_buffer > 0 and cur_iter > self.enable_after_iter and cur_iter < self.enable_after_iter + self.buffer_iter:
            runner.model.module.runtime_info['threshold_buffer'] = (self.enable_after_iter + self.buffer_iter - cur_iter) * self.delta
        else:
            # runner.hook.runtime_info['threshold_buffer'] = 0
            self.threshold_buffer = 0