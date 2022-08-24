# Copyright (c) OpenMMLab. All rights reserved.
import copy
import random
from os.path import dirname, exists, join

import numpy as np
import torch
from mmengine import InstanceData

from ..datasets import LoadAnnotations3D, LoadPointsFromFile
from ..structures import (CameraInstance3DBoxes, DepthInstance3DBoxes,
                          Det3DDataSample, LiDARInstance3DBoxes, PointData)


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


def get_detector_cfg(fname):
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
    _setup_seed(seed)
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
    if with_points:
        points = torch.rand([num_points, points_feat_dim])
    else:
        points = None
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

    else:
        img = None
    inputs_dict = dict(img=img, points=points)
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

    return dict(inputs=inputs_dict, data_sample=data_sample)


# create a dummy `results` to test the pipeline
def create_dummy_data_info(with_ann=True):

    ann_info = {
        'gt_bboxes':
        np.array([[712.4, 143., 810.73, 307.92]]),
        'gt_labels':
        np.array([1]),
        'gt_bboxes_3d':
        LiDARInstance3DBoxes(
            np.array(
                [[8.7314, -1.8559, -1.5997, 1.2000, 0.4800, 1.8900,
                  -1.5808]])),
        'gt_labels_3d':
        np.array([1]),
        'num_lidar_pts':
        np.array([377]),
        'difficulty':
        np.array([0]),
        'truncated':
        np.array([0]),
        'occluded':
        np.array([0]),
        'alpha':
        np.array([-0.2]),
        'score':
        np.array([0.]),
        'index':
        np.array([0]),
        'group_id':
        np.array([0])
    }
    data_info = {
        'sample_id':
        0,
        'images': {
            'CAM0': {
                'cam2img': [[707.0493, 0.0, 604.0814, 0.0],
                            [0.0, 707.0493, 180.5066, 0.0],
                            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
            },
            'CAM1': {
                'cam2img': [[707.0493, 0.0, 604.0814, -379.7842],
                            [0.0, 707.0493, 180.5066, 0.0],
                            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
            },
            'CAM2': {
                'img_path':
                'tests/data/kitti/training/image_2/000000.png',
                'height':
                370,
                'width':
                1224,
                'cam2img': [[707.0493, 0.0, 604.0814, 45.75831],
                            [0.0, 707.0493, 180.5066, -0.3454157],
                            [0.0, 0.0, 1.0, 0.004981016], [0.0, 0.0, 0.0, 1.0]]
            },
            'CAM3': {
                'cam2img': [[707.0493, 0.0, 604.0814, -334.1081],
                            [0.0, 707.0493, 180.5066, 2.33066],
                            [0.0, 0.0, 1.0, 0.003201153], [0.0, 0.0, 0.0, 1.0]]
            },
            'R0_rect': [[
                0.9999127984046936, 0.010092630051076412,
                -0.008511931635439396, 0.0
            ],
                        [
                            -0.010127290152013302, 0.9999405741691589,
                            -0.004037670791149139, 0.0
                        ],
                        [
                            0.008470674976706505, 0.0041235219687223434,
                            0.9999555945396423, 0.0
                        ], [0.0, 0.0, 0.0, 1.0]]
        },
        'lidar_points': {
            'num_pts_feats':
            4,
            'lidar_path':
            'tests/data/kitti/training/velodyne_reduced/000000.bin',
            'lidar2cam': [[
                -0.0015960992313921452, -0.9999162554740906,
                -0.012840436771512032, -0.022366708144545555
            ],
                          [
                              -0.00527064548805356, 0.012848696671426296,
                              -0.9999035596847534, -0.05967890843749046
                          ],
                          [
                              0.9999848008155823, -0.0015282672829926014,
                              -0.005290712229907513, -0.33254900574684143
                          ], [0.0, 0.0, 0.0, 1.0]],
            'Tr_velo_to_cam': [[
                0.006927963811904192, -0.9999722242355347, -0.0027578289154917,
                -0.024577289819717407
            ],
                               [
                                   -0.0011629819637164474,
                                   0.0027498360723257065, -0.9999955296516418,
                                   -0.06127237156033516
                               ],
                               [
                                   0.999975323677063, 0.006931141018867493,
                                   -0.0011438990477472544, -0.33210289478302
                               ], [0.0, 0.0, 0.0, 1.0]],
            'Tr_imu_to_velo': [[
                0.999997615814209, 0.0007553070900030434,
                -0.002035825978964567, -0.8086758852005005
            ],
                               [
                                   -0.0007854027207940817, 0.9998897910118103,
                                   -0.014822980388998985, 0.3195559084415436
                               ],
                               [
                                   0.002024406101554632, 0.014824540354311466,
                                   0.9998881220817566, -0.7997230887413025
                               ], [0.0, 0.0, 0.0, 1.0]]
        },
        'instances': [{
            'bbox': [712.4, 143.0, 810.73, 307.92],
            'bbox_label':
            -1,
            'bbox_3d': [
                1.840000033378601, 1.4700000286102295, 8.40999984741211,
                1.2000000476837158, 1.8899999856948853, 0.47999998927116394,
                0.009999999776482582
            ],
            'bbox_label_3d':
            -1,
            'num_lidar_pts':
            377,
            'difficulty':
            0,
            'truncated':
            0,
            'occluded':
            0,
            'alpha':
            -0.2,
            'score':
            0.0,
            'index':
            0,
            'group_id':
            0
        }],
        'plane':
        None
    }
    if with_ann:
        data_info['ann_info'] = ann_info
    return data_info


def create_data_info_after_loading():
    load_anns_transform = LoadAnnotations3D(
        with_bbox_3d=True, with_label_3d=True)
    load_points_transform = LoadPointsFromFile(
        coord_type='LIDAR', load_dim=4, use_dim=3)
    data_info = create_dummy_data_info()
    data_info = load_points_transform(data_info)
    data_info_after_loading = load_anns_transform(data_info)
    return data_info_after_loading


# TODO: refactor the ceph replace code
def replace_to_ceph(cfg):
    cfg_pretty_text = cfg.pretty_text

    replace_strs = \
        r'''file_client_args = dict(
            backend='petrel',
            path_mapping=dict({
                './data/DATA/': 's3://openmmlab/datasets/detection3d/CEPH/',
                'data/DATA/': 's3://openmmlab/datasets/detection3d/CEPH/'
            }))
        '''

    if 'nuscenes' in cfg_pretty_text:
        replace_strs = replace_strs.replace('DATA', 'nuscenes')
        replace_strs = replace_strs.replace('CEPH', 'nuscenes')
    elif 'lyft' in cfg_pretty_text:
        replace_strs = replace_strs.replace('DATA', 'lyft')
        replace_strs = replace_strs.replace('CEPH', 'lyft')
    elif 'waymo' in cfg_pretty_text:
        replace_strs = replace_strs.replace('DATA', 'waymo')
        replace_strs = replace_strs.replace('CEPH', 'waymo')
    elif 'kitti' in cfg_pretty_text:
        replace_strs = replace_strs.replace('DATA', 'kitti')
        replace_strs = replace_strs.replace('CEPH', 'kitti')
    elif 'scannet' in cfg_pretty_text:
        replace_strs = replace_strs.replace('DATA', 'scannet')
        replace_strs = replace_strs.replace('CEPH', 'scannet_processed')
    elif 's3dis' in cfg_pretty_text:
        replace_strs = replace_strs.replace('DATA', 's3dis')
        replace_strs = replace_strs.replace('CEPH', 's3dis_processed')
    elif 'sunrgbd' in cfg_pretty_text:
        replace_strs = replace_strs.replace('DATA', 'sunrgbd')
        replace_strs = replace_strs.replace('CEPH', 'sunrgbd_processed')
    elif 'semantickitti' in cfg_pretty_text:
        replace_strs = replace_strs.replace('DATA', 'semantickitti')
        replace_strs = replace_strs.replace('CEPH', 'semantickitti')
    elif 'nuimages' in cfg_pretty_text:
        replace_strs = replace_strs.replace('DATA', 'nuimages')
        replace_strs = replace_strs.replace('CEPH', 'nuimages')
    else:
        NotImplemented('Does not support global replacement')

    replace_strs = replace_strs.replace(' ', '').replace('\n', '')

    # use data info file from ceph
    # cfg_pretty_text = cfg_pretty_text.replace(
    #   'ann_file', replace_strs + ', ann_file')

    # replace LoadImageFromFile
    cfg_pretty_text = cfg_pretty_text.replace(
        'LoadImageFromFile\'', 'LoadImageFromFile\',' + replace_strs)

    # replace LoadImageFromFileMono3D
    cfg_pretty_text = cfg_pretty_text.replace(
        'LoadImageFromFileMono3D\'',
        'LoadImageFromFileMono3D\',' + replace_strs)

    # replace LoadPointsFromFile
    cfg_pretty_text = cfg_pretty_text.replace(
        'LoadPointsFromFile\'', 'LoadPointsFromFile\',' + replace_strs)

    # replace LoadPointsFromMultiSweeps
    cfg_pretty_text = cfg_pretty_text.replace(
        'LoadPointsFromMultiSweeps\'',
        'LoadPointsFromMultiSweeps\',' + replace_strs)

    # replace LoadAnnotations
    cfg_pretty_text = cfg_pretty_text.replace(
        'LoadAnnotations\'', 'LoadAnnotations\',' + replace_strs)

    # replace LoadAnnotations3D
    cfg_pretty_text = cfg_pretty_text.replace(
        'LoadAnnotations3D\'', 'LoadAnnotations3D\',' + replace_strs)

    # replace dbsampler
    cfg_pretty_text = cfg_pretty_text.replace('info_path',
                                              replace_strs + ', info_path')

    cfg = cfg.fromstring(cfg_pretty_text, file_format='.py')
    return cfg
