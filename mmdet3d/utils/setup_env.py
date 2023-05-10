# Copyright (c) OpenMMLab. All rights reserved.

import datetime
import os
import platform
import warnings

import cv2
from mmengine import DefaultScope
from torch import multiprocessing as mp


def setup_multi_processes(cfg):
    """Setup multi-processing environment variables."""
    # set multi-process start method as `fork` to speed up the training
    if platform.system() != 'Windows':
        mp_start_method = cfg.get('mp_start_method', 'fork')
        current_method = mp.get_start_method(allow_none=True)
        if current_method is not None and current_method != mp_start_method:
            warnings.warn(
                f'Multi-processing start method `{mp_start_method}` is '
                f'different from the previous setting `{current_method}`.'
                f'It will be force set to `{mp_start_method}`. You can change '
                f'this behavior by changing `mp_start_method` in your config.')
        mp.set_start_method(mp_start_method, force=True)

    # disable opencv multithreading to avoid system being overloaded
    opencv_num_threads = cfg.get('opencv_num_threads', 0)
    cv2.setNumThreads(opencv_num_threads)

    # setup OMP threads
    # This code is referred from https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py  # noqa
    workers_per_gpu = cfg.data.get('workers_per_gpu', 1)
    if 'train_dataloader' in cfg.data:
        workers_per_gpu = \
            max(cfg.data.train_dataloader.get('workers_per_gpu', 1),
                workers_per_gpu)

    if 'OMP_NUM_THREADS' not in os.environ and workers_per_gpu > 1:
        omp_num_threads = 1
        warnings.warn(
            f'Setting OMP_NUM_THREADS environment variable for each process '
            f'to be {omp_num_threads} in default, to avoid your system being '
            f'overloaded, please further tune the variable for optimal '
            f'performance in your application as needed.')
        os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in os.environ and workers_per_gpu > 1:
        mkl_num_threads = 1
        warnings.warn(
            f'Setting MKL_NUM_THREADS environment variable for each process '
            f'to be {mkl_num_threads} in default, to avoid your system being '
            f'overloaded, please further tune the variable for optimal '
            f'performance in your application as needed.')
        os.environ['MKL_NUM_THREADS'] = str(mkl_num_threads)


def register_all_modules(init_default_scope: bool = True) -> None:
    """Register all modules in mmdet3d into the registries.

    Args:
        init_default_scope (bool): Whether initialize the mmdet default scope.
            When `init_default_scope=True`, the global default scope will be
            set to `mmdet3d`, and all registries will build modules from mmdet3d's
            registry node. To understand more about the registry, please refer
            to https://github.com/open-mmlab/mmengine/blob/main/docs/en/advanced_tutorials/registry.md
            Defaults to True.
    """  # noqa
    import mmdet3d.datasets  # noqa: F401,F403
    import mmdet3d.engine  # noqa: F401,F403
    import mmdet3d.evaluation.metrics  # noqa: F401,F403
    import mmdet3d.models  # noqa: F401,F403
    import mmdet3d.structures  # noqa: F401,F403
    import mmdet3d.visualization  # noqa: F401,F403
    if init_default_scope:
        never_created = DefaultScope.get_current_instance() is None \
                        or not DefaultScope.check_instance_created('mmdet3d')
        if never_created:
            DefaultScope.get_instance('mmdet3d', scope_name='mmdet3d')
            return
        current_scope = DefaultScope.get_current_instance()
        if current_scope.scope_name != 'mmdet3d':
            warnings.warn('The current default scope '
                          f'"{current_scope.scope_name}" is not "mmdet3d", '
                          '`register_all_modules` will force the current'
                          'default scope to be "mmdet3d". If this is not '
                          'expected, please set `init_default_scope=False`.')
            # avoid name conflict
            new_instance_name = f'mmdet3d-{datetime.datetime.now()}'
            DefaultScope.get_instance(new_instance_name, scope_name='mmdet3d')
