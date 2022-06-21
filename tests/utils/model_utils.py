# Copyright (c) OpenMMLab. All rights reserved.
import copy
import random
from os.path import dirname, exists, join

import numpy as np
import torch
from mmengine import InstanceData

from mmdet3d.core import Det3DDataSample, LiDARInstance3DBoxes


def _setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _get_config_directory():
    """Find the predefined detector config directory."""
    try:
        # Assume we are running in the source mmdetection3d repo
        repo_dpath = dirname(dirname(dirname(__file__)))
    except NameError:
        # For IPython development when this __file__ is not defined
        import mmdet3d
        repo_dpath = dirname(dirname(mmdet3d.__file__))
    config_dpath = join(repo_dpath, 'configs')
    if not exists(config_dpath):
        raise Exception('Cannot find config path')
    return config_dpath


def _get_config_module(fname):
    """Load a configuration as a python module."""
    from mmcv import Config
    config_dpath = _get_config_directory()
    config_fpath = join(config_dpath, fname)
    config_mod = Config.fromfile(config_fpath)
    return config_mod


def _get_model_cfg(fname):
    """Grab configs necessary to create a model.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)

    return model


def _get_detector_cfg(fname):
    """Grab configs necessary to create a detector.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    import mmcv
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    train_cfg = mmcv.Config(copy.deepcopy(config.model.train_cfg))
    test_cfg = mmcv.Config(copy.deepcopy(config.model.test_cfg))

    model.update(train_cfg=train_cfg)
    model.update(test_cfg=test_cfg)
    return model


def _create_detector_inputs(seed=0,
                            num_gt_instance=20,
                            points_feat_dim=4,
                            num_classes=3):
    _setup_seed(seed)
    inputs_dict = dict(points=torch.rand([10, points_feat_dim]))
    gt_instance_3d = InstanceData()
    gt_instance_3d.bboxes_3d = LiDARInstance3DBoxes(
        torch.rand([num_gt_instance, 7]))
    gt_instance_3d.labels_3d = torch.randint(0, num_classes, [num_gt_instance])
    data_sample = Det3DDataSample(
        metainfo=dict(box_type_3d=LiDARInstance3DBoxes))
    data_sample.gt_instances_3d = gt_instance_3d
    data_sample.seg_data = dict()
    return dict(inputs=inputs_dict, data_sample=data_sample)
