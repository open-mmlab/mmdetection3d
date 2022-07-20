# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmdet3d.registry import TASK_UTILS

PRIOR_GENERATORS = TASK_UTILS

ANCHOR_GENERATORS = TASK_UTILS


def build_prior_generator(cfg, default_args=None):
    warnings.warn(
        '``build_prior_generator`` would be deprecated soon, please use '
        '``mmdet3d.registry.TASK_UTILS.build()`` ')
    return TASK_UTILS.build(cfg, default_args=default_args)


def build_anchor_generator(cfg, default_args=None):
    warnings.warn(
        '``build_anchor_generator`` would be deprecated soon, please use '
        '``mmdet3d.registry.TASK_UTILS.build()`` ')
    return TASK_UTILS.build(cfg, default_args=default_args)
