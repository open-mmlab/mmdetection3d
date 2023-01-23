from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook
from mmdet3d.datasets.pipelines import ObjectSample

@HOOKS.register_module()
class FadeOjectSampleHook(Hook):
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
                 num_last_epochs=5,
                 skip_type_keys=('ObjectSample')):
        self.num_last_epochs = num_last_epochs
        self.skip_type_keys = skip_type_keys
        self._restart_dataloader = False

    def before_train_epoch(self, runner):
        """Close ObjectSample."""

        epoch = runner.epoch
        train_loader = runner.data_loader
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        if (epoch + 1) > runner.max_epochs - self.num_last_epochs:
            runner.logger.info('No ObjectSample now!')
            # The dataset pipeline cannot be updated when persistent_workers  
            # is True, so we need to force the dataloader's multi-process
            # restart. This is a very hacky approach.
            #print(type(train_loader.dataset.pipeline.transforms))
            #print(train_loader.dataset.__dict__)
            if hasattr(train_loader.dataset, 'dataset'):
                transforms = train_loader.dataset.dataset.pipeline.transforms
            else:    
                transforms = train_loader.dataset.pipeline.transforms
            
            for transform in transforms:
                if isinstance(transform, ObjectSample):
                    transforms.remove(transform)
                    break
            
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