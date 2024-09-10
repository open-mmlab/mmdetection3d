# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Any

from mmdet3d.registry import TASK_UTILS
from mmdet3d.utils.typing_utils import ConfigType

BBOX_ASSIGNERS = TASK_UTILS
BBOX_SAMPLERS = TASK_UTILS
BBOX_CODERS = TASK_UTILS


def build_assigner(cfg: ConfigType, **default_args) -> Any:
    """Builder of box assigner."""
    warnings.warn('``build_assigner`` would be deprecated soon, please use '
                  '``mmdet3d.registry.TASK_UTILS.build()`` ')
    return TASK_UTILS.build(cfg, default_args=default_args)


def build_sampler(cfg: ConfigType, **default_args) -> Any:
    """Builder of box sampler."""
    warnings.warn('``build_sampler`` would be deprecated soon, please use '
                  '``mmdet3d.registry.TASK_UTILS.build()`` ')
    return TASK_UTILS.build(cfg, default_args=default_args)


def build_bbox_coder(cfg: ConfigType, **default_args) -> Any:
    """Builder of box coder."""
    warnings.warn('``build_bbox_coder`` would be deprecated soon, please use '
                  '``mmdet3d.registry.TASK_UTILS.build()`` ')
    return TASK_UTILS.build(cfg, default_args=default_args)
