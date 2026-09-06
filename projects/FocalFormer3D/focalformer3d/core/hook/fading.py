# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates.
# All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FocalFormer3D/blob/main/LICENSE

import logging

from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.registry import HOOKS


def _find_dataset(dataloader):
    """Unwrap dataset wrappers until one with a ``pipeline`` turns up.

    The training dataset is usually a ``CBGSDataset`` around the real one, and
    may pick up a ``RepeatDataset`` or ``ConcatDataset`` on the way.
    """
    dataset = dataloader.dataset
    depth = 0
    while hasattr(dataset, 'dataset') and not hasattr(dataset, 'pipeline'):
        dataset = dataset.dataset
        depth += 1
        if depth > 10:
            break
    return dataset


@HOOKS.register_module()
class Fading(Hook):
    """Drop ``ObjectSample`` from the training pipeline at ``fade_epoch``.

    Ground-truth sampling helps early on and hurts at the end of training, so
    the upstream recipe removes it for the last epochs. mmdet3d ships
    ``DisableObjectSampleHook``, which does the same thing; this hook exists so
    the ported configs keep the upstream name and semantics.

    Args:
        fade_epoch (int): First epoch to train without ``ObjectSample``.
            Epochs are 0-indexed, so ``fade_epoch=1`` keeps it for epoch 0
            only. Defaults to 100000, i.e. never.
    """

    def __init__(self, fade_epoch=100000):
        super().__init__()
        self.fade_epoch = fade_epoch
        self._removed = False

    def before_train_epoch(self, runner):
        if self._removed or runner.epoch < self.fade_epoch:
            return

        dataset = _find_dataset(runner.train_dataloader)
        if not hasattr(dataset, 'pipeline'):
            print_log(
                f'[Fading] {type(dataset).__name__} has no .pipeline, '
                'leaving the pipeline untouched.',
                logger='current',
                level=logging.WARNING)
            self._removed = True
            return

        transforms = dataset.pipeline.transforms
        for i, transform in enumerate(transforms):
            if type(transform).__name__ == 'ObjectSample':
                transforms.pop(i)
                print_log(
                    f'[Fading] removed ObjectSample at epoch {runner.epoch}.',
                    logger='current')
                break
        else:
            print_log(
                '[Fading] no ObjectSample in the training pipeline, '
                'nothing to remove.',
                logger='current',
                level=logging.WARNING)

        # Either way, do not search again on the next epoch.
        self._removed = True
