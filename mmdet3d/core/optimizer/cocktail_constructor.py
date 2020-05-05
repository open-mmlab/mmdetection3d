from mmcv.utils import build_from_cfg

from mmdet.core.optimizer import OPTIMIZER_BUILDERS, OPTIMIZERS
from mmdet.utils import get_root_logger
from .cocktail_optimizer import CocktailOptimizer


@OPTIMIZER_BUILDERS.register_module()
class CocktailOptimizerConstructor(object):
    """Special constructor for cocktail optimizers.

    This constructor constructs cocktail optimizer for multi-modality
    detectors. It builds separate optimizers for separate branchs for
    different modalities. More details can be found in the ECCV submission
    (to be release).

    Attributes:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer. The keys of
            the dict are used to search for the corresponding keys in the
            model, and the value if a dict that really defines the optimizer.
            See example below for the usage.
        paramwise_cfg (dict): The dict for paramwise options. This is not
            supported in the current version. But it should be supported in
            the future release.

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> model = nn.ModuleDict({
        >>>     'pts': nn.modules.Conv1d(1, 1, 1, bias=False),
        >>>     'img': nn.modules.Conv1d(1, 1, 1, bias=False)
        >>> })
        >>> optimizer_cfg = dict(
        >>>    pts=dict(type='AdamW', lr=0.001,
        >>>             weight_decay=0.01, step_interval=1),
        >>>    img=dict(type='SGD', lr=0.02, momentum=0.9,
        >>>             weight_decay=0.0001, step_interval=2))
        >>> optim_builder = CocktailOptimizerConstructor(optimizer_cfg)
        >>> optimizer = optim_builder(model)
        >>> print(optimizer)
        CocktailOptimizer (
        Update interval: 1
        AdamW (
          Parameter Group 0
              amsgrad: False
              betas: (0.9, 0.999)
              eps: 1e-08
              lr: 0.001
              weight_decay: 0.01
          ),
        Update interval: 2
        SGD (
          Parameter Group 0
              dampening: 0
              lr: 0.02
              momentum: 0.9
              nesterov: False
              weight_decay: 0.0001
          ),
        )
    """

    def __init__(self, optimizer_cfg, paramwise_cfg=None):
        if not isinstance(optimizer_cfg, dict):
            raise TypeError('optimizer_cfg should be a dict',
                            'but got {}'.format(type(optimizer_cfg)))
        assert paramwise_cfg is None, \
            'Parameter wise config is not supported in Cocktail Optimizer'
        self.optimizer_cfg = optimizer_cfg

    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module

        optimizer_cfg = self.optimizer_cfg.copy()
        logger = get_root_logger()
        keys_prefix = [key_prefix for key_prefix in optimizer_cfg.keys()]
        keys_params = {key: [] for key in keys_prefix}
        keys_params_name = {key: [] for key in keys_prefix}
        keys_optimizer = []
        for name, param in model.named_parameters():
            param_group = {'params': [param]}
            find_flag = False
            for key in keys_prefix:
                if key in name:
                    keys_params[key].append(param_group)
                    keys_params_name[key].append(name)
                    find_flag = True
                    break
            assert find_flag, 'key {} is not matched to any optimizer'.format(
                name)

        step_intervals = []
        for key, single_cfg in optimizer_cfg.items():
            step_intervals.append(single_cfg.pop('step_interval', 1))
            single_cfg['params'] = keys_params[key]
            single_optim = build_from_cfg(single_cfg, OPTIMIZERS)
            keys_optimizer.append(single_optim)
            logger.info('{} optimizes key:\n {}\n'.format(
                single_cfg['type'], keys_params_name[key]))

        cocktail_optimizer = CocktailOptimizer(keys_optimizer, step_intervals)
        return cocktail_optimizer
