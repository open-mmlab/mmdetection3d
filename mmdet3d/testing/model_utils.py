# Copyright (c) OpenMMLab. All rights reserved.
import copy
import random
from os.path import dirname, exists, join

import numpy as np
import torch
from mmengine.structures import InstanceData

from mmdet3d.structures import (CameraInstance3DBoxes, DepthInstance3DBoxes,
                                Det3DDataSample, LiDARInstance3DBoxes,
                                PointData)


def setup_seed(seed):
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
    from mmengine import Config
    config_dpath = _get_config_directory()
    config_fpath = join(config_dpath, fname)
    config_mod = Config.fromfile(config_fpath)
    return config_mod


def get_model_cfg(fname):
    """Grab configs necessary to create a model.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)

    return model


def get_detector_cfg(fname):
    """Grab configs necessary to create a detector.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    import mmengine
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    train_cfg = mmengine.Config(copy.deepcopy(config.model.train_cfg))
    test_cfg = mmengine.Config(copy.deepcopy(config.model.test_cfg))

    model.update(train_cfg=train_cfg)
    model.update(test_cfg=test_cfg)
    return model


def create_detector_inputs(seed=0,
                           with_points=True,
                           with_img=False,
                           img_size=10,
                           num_gt_instance=20,
                           num_points=10,
                           points_feat_dim=4,
                           num_classes=3,
                           gt_bboxes_dim=7,
                           with_pts_semantic_mask=False,
                           with_pts_instance_mask=False,
                           bboxes_3d_type='lidar'):
    setup_seed(seed)
    assert bboxes_3d_type in ('lidar', 'depth', 'cam')
    bbox_3d_class = {
        'lidar': LiDARInstance3DBoxes,
        'depth': DepthInstance3DBoxes,
        'cam': CameraInstance3DBoxes
    }
    meta_info = dict()
    meta_info['depth2img'] = np.array(
        [[5.23289349e+02, 3.68831943e+02, 6.10469439e+01],
         [1.09560138e+02, 1.97404735e+02, -5.47377738e+02],
         [1.25930002e-02, 9.92229998e-01, -1.23769999e-01]])
    meta_info['lidar2img'] = np.array(
        [[5.23289349e+02, 3.68831943e+02, 6.10469439e+01],
         [1.09560138e+02, 1.97404735e+02, -5.47377738e+02],
         [1.25930002e-02, 9.92229998e-01, -1.23769999e-01]])

    inputs_dict = dict()

    if with_points:
        points = torch.rand([num_points, points_feat_dim])
        inputs_dict['points'] = [points]

    if with_img:
        if isinstance(img_size, tuple):
            img = torch.rand(3, img_size[0], img_size[1])
            meta_info['img_shape'] = img_size
            meta_info['ori_shape'] = img_size
        else:
            img = torch.rand(3, img_size, img_size)
            meta_info['img_shape'] = (img_size, img_size)
            meta_info['ori_shape'] = (img_size, img_size)
        meta_info['scale_factor'] = np.array([1., 1.])
        inputs_dict['img'] = [img]

    gt_instance_3d = InstanceData()

    gt_instance_3d.bboxes_3d = bbox_3d_class[bboxes_3d_type](
        torch.rand([num_gt_instance, gt_bboxes_dim]), box_dim=gt_bboxes_dim)
    gt_instance_3d.labels_3d = torch.randint(0, num_classes, [num_gt_instance])
    data_sample = Det3DDataSample(
        metainfo=dict(box_type_3d=bbox_3d_class[bboxes_3d_type]))
    data_sample.set_metainfo(meta_info)
    data_sample.gt_instances_3d = gt_instance_3d

    gt_instance = InstanceData()
    gt_instance.labels = torch.randint(0, num_classes, [num_gt_instance])
    gt_instance.bboxes = torch.rand(num_gt_instance, 4)
    gt_instance.bboxes[:,
                       2:] = gt_instance.bboxes[:, :2] + gt_instance.bboxes[:,
                                                                            2:]

    data_sample.gt_instances = gt_instance
    data_sample.gt_pts_seg = PointData()
    if with_pts_instance_mask:
        pts_instance_mask = torch.randint(0, num_gt_instance, [num_points])
        data_sample.gt_pts_seg['pts_instance_mask'] = pts_instance_mask
    if with_pts_semantic_mask:
        pts_semantic_mask = torch.randint(0, num_classes, [num_points])
        data_sample.gt_pts_seg['pts_semantic_mask'] = pts_semantic_mask

    return dict(inputs=inputs_dict, data_samples=[data_sample])
