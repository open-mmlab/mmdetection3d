# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from copy import deepcopy
from os import path as osp
from typing import Sequence, Union

import mmcv
import mmengine
import numpy as np
import torch
import torch.nn as nn
from mmengine.dataset import Compose
from mmengine.runner import load_checkpoint

from mmdet3d.core import Box3DMode, Det3DDataSample, SampleList
from mmdet3d.core.bbox import get_box_type
from mmdet3d.models import build_model


def convert_SyncBN(config):
    """Convert config's naiveSyncBN to BN.

    Args:
         config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
    """
    if isinstance(config, dict):
        for item in config:
            if item == 'norm_cfg':
                config[item]['type'] = config[item]['type']. \
                                    replace('naiveSyncBN', 'BN')
            else:
                convert_SyncBN(config[item])


def init_model(config, checkpoint=None, device='cuda:0'):
    """Initialize a model from config file, which could be a 3D detector or a
    3D segmentor.

    Args:
        config (str or :obj:`mmengine.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): Device to use.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmengine.Config.fromfile(config)
    elif not isinstance(config, mmengine.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    convert_SyncBN(config.model)
    config.model.train_cfg = None
    model = build_model(config.model)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = config.class_names
        if 'PALETTE' in checkpoint['meta']:  # 3D Segmentor
            model.PALETTE = checkpoint['meta']['PALETTE']
    model.cfg = config  # save the config in the model for convenience
    if device != 'cpu':
        torch.cuda.set_device(device)
    else:
        warnings.warn('Don\'t suggest using CPU device. '
                      'Some functions are not supported for now.')

    model.to(device)
    model.eval()
    return model


PointsType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]
ImagesType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]


def inference_detector(model: nn.Module,
                       pcds: PointsType) -> Union[Det3DDataSample, SampleList]:
    """Inference point cloud with the detector.

    Args:
        model (nn.Module): The loaded detector.
        pcds (str, ndarray, Sequence[str/ndarray]):
            Either point cloud files or loaded point cloud.

    Returns:
        :obj:`Det3DDataSample` or list[:obj:`Det3DDataSample`]:
        If pcds is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """
    if isinstance(pcds, (list, tuple)):
        is_batch = True
    else:
        pcds = [pcds]
        is_batch = False

    cfg = model.cfg

    if not isinstance(pcds[0], str):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.test_dataloader.dataset.pipeline[0].type = 'LoadPointsFromDict'

    # build the data pipeline
    test_pipeline = deepcopy(cfg.test_dataloader.dataset.pipeline)
    test_pipeline = Compose(test_pipeline)
    # box_type_3d, box_mode_3d = get_box_type(
    # cfg.test_dataloader.dataset.box_type_3d)

    data = []
    for pcd in pcds:
        # prepare data
        if isinstance(pcd, str):
            # load from point cloud file
            data_ = dict(
                lidar_points=dict(lidar_path=pcd),
                # for ScanNet demo we need axis_align_matrix
                ann_info=dict(axis_align_matrix=np.eye(4)),
                sweeps=[],
                # set timestamp = 0
                timestamp=[0])
        else:
            # directly use loaded point cloud
            data_ = dict(
                points=pcd,
                # for ScanNet demo we need axis_align_matrix
                ann_info=dict(axis_align_matrix=np.eye(4)),
                sweeps=[],
                # set timestamp = 0
                timestamp=[0])
        data_ = test_pipeline(data_)
        data.append(data_)

    # forward the model
    with torch.no_grad():
        results = model.test_step(data)

    if not is_batch:
        return results[0]
    else:
        return results


def inference_multi_modality_detector(model: nn.Module,
                                      pcds: Union[str, Sequence[str]],
                                      imgs: Union[str, Sequence[str]],
                                      ann_files: Union[str, Sequence[str]]):
    """Inference point cloud with the multi-modality detector.

    Args:
        model (nn.Module): The loaded detector.
        pcds (str, Sequence[str]):
            Either point cloud files or loaded point cloud.
        imgs (str, Sequence[str]):
           Either image files or loaded images.
        ann_files (str, Sequence[str]): Annotation files.

    Returns:
        :obj:`Det3DDataSample` or list[:obj:`Det3DDataSample`]:
        If pcds is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    # TODO: We will support
    if isinstance(pcds, (list, tuple)):
        is_batch = True
        assert isinstance(imgs, (list, tuple))
        assert isinstance(ann_files, (list, tuple))
        assert len(pcds) == len(imgs) == len(ann_files)
    else:
        pcds = [pcds]
        imgs = [imgs]
        ann_files = [ann_files]
        is_batch = False

    cfg = model.cfg

    # build the data pipeline
    test_pipeline = deepcopy(cfg.test_dataloader.dataset.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = \
        get_box_type(cfg.test_dataloader.dataset.box_type_3d)

    data = []
    for index, pcd in enumerate(pcds):
        # get data info containing calib
        img = imgs[index]
        ann_file = ann_files[index]
        data_info = mmcv.load(ann_file)[0]
        # TODO: check the name consistency of
        # image file and point cloud file
        data_ = dict(
            lidar_points=dict(lidar_path=pcd),
            img_path=imgs[index],
            img_prefix=osp.dirname(img),
            img_info=dict(filename=osp.basename(img)),
            box_type_3d=box_type_3d,
            box_mode_3d=box_mode_3d)
        data_ = test_pipeline(data_)

        # LiDAR to image conversion for KITTI dataset
        if box_mode_3d == Box3DMode.LIDAR:
            data_['lidar2img'] = data_info['images']['CAM2']['lidar2img']
        # Depth to image conversion for SUNRGBD dataset
        elif box_mode_3d == Box3DMode.DEPTH:
            data_['depth2img'] = data_info['images']['CAM0']['depth2img']

        data.append(data_)

    # forward the model
    with torch.no_grad():
        results = model.test_step(data)

    if not is_batch:
        return results[0]
    else:
        return results


def inference_mono_3d_detector(model: nn.Module, imgs: ImagesType,
                               ann_files: Union[str, Sequence[str]]):
    """Inference image with the monocular 3D detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str, Sequence[str]):
           Either image files or loaded images.
        ann_files (str, Sequence[str]): Annotation files.

    Returns:
        :obj:`Det3DDataSample` or list[:obj:`Det3DDataSample`]:
        If pcds is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """
    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg

    # build the data pipeline
    test_pipeline = deepcopy(cfg.test_dataloader.dataset.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = \
        get_box_type(cfg.test_dataloader.dataset.box_type_3d)

    data = []
    for index, img in enumerate(imgs):
        ann_file = ann_files[index]
        # get data info containing calib
        data_info = mmcv.load(ann_file)[0]
        data_ = dict(
            img_path=img,
            images=data_info['images'],
            box_type_3d=box_type_3d,
            box_mode_3d=box_mode_3d)

        data_ = test_pipeline(data_)

    # forward the model
    with torch.no_grad():
        results = model.test_step(data)

    if not is_batch:
        return results[0]
    else:
        return results


def inference_segmentor(model: nn.Module, pcds: PointsType):
    """Inference point cloud with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        pcds (str, Sequence[str]):
            Either point cloud files or loaded point cloud.

    Returns:
        :obj:`Det3DDataSample` or list[:obj:`Det3DDataSample`]:
        If pcds is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """
    if isinstance(pcds, (list, tuple)):
        is_batch = True
    else:
        pcds = [pcds]
        is_batch = False

    cfg = model.cfg

    # build the data pipeline
    test_pipeline = deepcopy(cfg.test_dataloader.dataset.pipeline)
    test_pipeline = Compose(test_pipeline)

    data = []
    for pcd in pcds:
        data_ = dict(lidar_points=dict(lidar_path=pcd))
        data_ = test_pipeline(data_)
        data.append(data_)

    # forward the model
    with torch.no_grad():
        results = model.test_step(data)

    if not is_batch:
        return results[0]
    else:
        return results
