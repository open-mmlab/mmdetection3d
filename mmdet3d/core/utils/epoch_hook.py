from mmcv.runner import HOOKS, Hook
@HOOKS.register_module()
class EpochHook(Hook):
    def __init__(self):
        pass

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        from mmdet3d.datasets.pipelines.transforms_3d import ObjectSample
        try:
            lens = len(runner.data_loader.dataset.pipeline.transforms)
            for i in range(lens):
                if isinstance(runner.data_loader.dataset.pipeline.transforms[i], ObjectSample):
                    runner.data_loader.dataset.pipeline.transforms[i].cur_epoch = runner.epoch + 1
                    break
        except:
            import logging
            logging.error("EpochHook set ObjectSample's cur_epoch:{} failed!".format(runner.epoch + 1))

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass
