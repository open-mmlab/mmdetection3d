# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

from mmengine.registry import Registry
from torch import nn as nn

SA_MODULES = Registry(
    name='point_sa_module',
    locations=['mmdet3d.models.layers.pointnet_modules'])


def build_sa_module(cfg: Union[dict, None], *args, **kwargs) -> nn.Module:
    """Build PointNet2 set abstraction (SA) module.

    Args:
        cfg (dict or None): The SA module config, which should contain:

            - type (str): Module type.
            - module args: Args needed to instantiate an SA module.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding module.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding SA module .

    Returns:
        nn.Module: Created SA module.
    """
    if cfg is None:
        cfg_ = dict(type='PointSAModule')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    module_type = cfg_.pop('type')
    if module_type not in SA_MODULES:
        raise KeyError(f'Unrecognized module type {module_type}')
    else:
        sa_module = SA_MODULES.get(module_type)

    module = sa_module(*args, **kwargs, **cfg_)

    return module
