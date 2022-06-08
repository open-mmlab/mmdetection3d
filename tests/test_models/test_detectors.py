# Copyright (c) OpenMMLab. All rights reserved.
import copy
import random
from os.path import dirname, exists, join

import numpy as np
import pytest
import torch
from mmengine.data import InstanceData

from mmdet3d.core import Det3DDataSample
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.registry import MODELS


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


def test_voxel_net():
    import mmdet3d.models
    assert hasattr(mmdet3d.models, 'VoxelNet')
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    _setup_seed(0)
    voxel_net_cfg = _get_detector_cfg(
        'pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py')
    model = MODELS.build(voxel_net_cfg).cuda()
    input_dict0 = dict(points=torch.rand([2010, 4], device='cuda'))
    input_dict1 = dict(points=torch.rand([2020, 4], device='cuda'))
    gt_instance_3d_0 = InstanceData()
    gt_instance_3d_0.bboxes_3d = LiDARInstance3DBoxes(
        torch.rand([20, 7], device='cuda'))
    gt_instance_3d_0.labels_3d = torch.randint(0, 3, [20], device='cuda')
    data_sample_0 = Det3DDataSample(
        metainfo=dict(box_type_3d=LiDARInstance3DBoxes))
    data_sample_0.gt_instances_3d = gt_instance_3d_0

    gt_instance_3d_1 = InstanceData()
    gt_instance_3d_1.bboxes_3d = LiDARInstance3DBoxes(
        torch.rand([50, 7], device='cuda'))
    gt_instance_3d_1.labels_3d = torch.randint(0, 3, [50], device='cuda')
    data_sample_1 = Det3DDataSample(
        metainfo=dict(box_type_3d=LiDARInstance3DBoxes))
    data_sample_1.gt_instances_3d = gt_instance_3d_1
    data = [dict(inputs=input_dict0, data_sample=data_sample_0)]

    # test simple_test
    with torch.no_grad():
        results = model.forward(data, return_loss=False)
    bboxes_3d = results[0].pred_instances_3d['bboxes_3d']
    scores_3d = results[0].pred_instances_3d['scores_3d']
    labels_3d = results[0].pred_instances_3d['labels_3d']
    assert bboxes_3d.tensor.shape == (50, 7)
    assert scores_3d.shape == torch.Size([50])
    assert labels_3d.shape == torch.Size([50])

    # test forward_train
    data = [
        dict(inputs=input_dict0, data_sample=data_sample_0),
        dict(inputs=input_dict1, data_sample=data_sample_1)
    ]
    losses = model.forward(data, return_loss=True)
    assert losses['log_vars']['loss_cls'] >= 0
    assert losses['log_vars']['loss_bbox'] >= 0
    assert losses['log_vars']['loss_dir'] >= 0
    assert losses['log_vars']['loss'] >= 0

    # test_aug_test
    metainfo = {
        'pcd_scale_factor': 1,
        'pcd_horizontal_flip': 1,
        'pcd_vertical_flip': 1,
        'box_type_3d': LiDARInstance3DBoxes
    }
    data_sample_0.set_metainfo(metainfo)
    data_sample_1.set_metainfo(metainfo)
    data = [
        dict(inputs=input_dict0, data_sample=data_sample_0),
        dict(inputs=input_dict1, data_sample=data_sample_1)
    ]
    results = model.forward(data, return_loss=False)
