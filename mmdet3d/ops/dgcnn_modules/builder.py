from mmcv.utils import Registry

GF_MODULES = Registry('dgcnn_gf_module')
FA_MODULES = Registry('dgcnn_fa_module')


def build_gf_module(cfg, *args, **kwargs):
    """Build DGCNN graph feature (GF) module.

    Args:
        cfg (None or dict): The GF module config, which should contain:
            - type (str): Module type.
            - module args: Args needed to instantiate an GF module.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding module.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding GF module .

    Returns:
        nn.Module: Created GF module.
    """
    if cfg is None:
        cfg_ = dict(type='DGCNNGFModule')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    module_type = cfg_.pop('type')
    if module_type not in GF_MODULES:
        raise KeyError(f'Unrecognized module type {module_type}')
    else:
        gf_module = GF_MODULES.get(module_type)

    module = gf_module(*args, **kwargs, **cfg_)

    return module


def build_fa_module(cfg, *args, **kwargs):
    """Build DGCNN feature aggregation (FA) module.

    Args:
        cfg (None or dict): The FA module config, which should contain:
            - type (str): Module type.
            - module args: Args needed to instantiate an FA module.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding module.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding FA module .

    Returns:
        nn.Module: Created FA module.
    """
    if cfg is None:
        cfg_ = dict(type='DGCNNFAModule')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    module_type = cfg_.pop('type')
    if module_type not in FA_MODULES:
        raise KeyError(f'Unrecognized module type {module_type}')
    else:
        fa_module = FA_MODULES.get(module_type)

    module = fa_module(*args, **kwargs, **cfg_)

    return module
