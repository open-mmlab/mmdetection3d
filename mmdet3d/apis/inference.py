# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from copy import deepcopy
from os import path as osp
from pathlib import Path
from typing import Optional, Sequence, Union

import mmengine
import numpy as np
import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint

from mmdet3d.registry import DATASETS, MODELS
from mmdet3d.structures import Box3DMode, Det3DDataSample, get_box_type
from mmdet3d.structures.det3d_data_sample import SampleList


def convert_SyncBN(config):
    """Convert config's naiveSyncBN to BN.

    Args:
         config (str or :obj:`mmengine.Config`): Config file path or the config
            object.
    """
    if isinstance(config, dict):
        for item in config:
            if item == 'norm_cfg':
                config[item]['type'] = config[item]['type']. \
                                    replace('naiveSyncBN', 'BN')
            else:
                convert_SyncBN(config[item])


def init_model(config: Union[str, Path, Config],
               checkpoint: Optional[str] = None,
               device: str = 'cuda:0',
               palette: str = 'none',
               cfg_options: Optional[dict] = None):
    """Initialize a model from config file, which could be a 3D detector or a
    3D segmentor.

    Args:
        config (str, :obj:`Path`, or :obj:`mmengine.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): Device to use.
        cfg_options (dict, optional): Options to override some settings in
            the used config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)

    convert_SyncBN(config.model)
    config.model.train_cfg = None
    init_default_scope(config.get('default_scope', 'mmdet3d'))
    model = MODELS.build(config.model)

    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        # save the dataset_meta in the model for convenience
        if 'dataset_meta' in checkpoint.get('meta', {}):
            # mmdet3d 1.x
            model.dataset_meta = checkpoint['meta']['dataset_meta']
        elif 'CLASSES' in checkpoint.get('meta', {}):
            # < mmdet3d 1.x
            classes = checkpoint['meta']['CLASSES']
            model.dataset_meta = {'classes': classes}

            if 'PALETTE' in checkpoint.get('meta', {}):  # 3D Segmentor
                model.dataset_meta['palette'] = checkpoint['meta']['PALETTE']
        else:
            # < mmdet3d 1.x
            model.dataset_meta = {'classes': config.class_names}

            if 'PALETTE' in checkpoint.get('meta', {}):  # 3D Segmentor
                model.dataset_meta['palette'] = checkpoint['meta']['PALETTE']

        test_dataset_cfg = deepcopy(config.test_dataloader.dataset)
        # lazy init. We only need the metainfo.
        test_dataset_cfg['lazy_init'] = True
        metainfo = DATASETS.build(test_dataset_cfg).metainfo
        cfg_palette = metainfo.get('palette', None)
        if cfg_palette is not None:
            model.dataset_meta['palette'] = cfg_palette
        else:
            if 'palette' not in model.dataset_meta:
                warnings.warn(
                    'palette does not exist, random is used by default. '
                    'You can also set the palette to customize.')
                model.dataset_meta['palette'] = 'random'

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
    box_type_3d, box_mode_3d = \
        get_box_type(cfg.test_dataloader.dataset.box_type_3d)

    data = []
    for pcd in pcds:
        # prepare data
        if isinstance(pcd, str):
            # load from point cloud file
            data_ = dict(
                lidar_points=dict(lidar_path=pcd),
                timestamp=1,
                # for ScanNet demo we need axis_align_matrix
                axis_align_matrix=np.eye(4),
                box_type_3d=box_type_3d,
                box_mode_3d=box_mode_3d)
        else:
            # directly use loaded point cloud
            data_ = dict(
                points=pcd,
                timestamp=1,
                # for ScanNet demo we need axis_align_matrix
                axis_align_matrix=np.eye(4),
                box_type_3d=box_type_3d,
                box_mode_3d=box_mode_3d)
        data_ = test_pipeline(data_)
        data.append(data_)

    collate_data = pseudo_collate(data)

    # forward the model
    with torch.no_grad():
        results = model.test_step(collate_data)

    if not is_batch:
        return results[0], data[0]
    else:
        return results, data


def inference_multi_modality_detector(model: nn.Module,
                                      pcds: Union[str, Sequence[str]],
                                      imgs: Union[str, Sequence[str]],
                                      ann_file: Union[str, Sequence[str]],
                                      cam_type: str = 'CAM2'):
    """Inference point cloud with the multi-modality detector. Now we only
    support multi-modality detector for KITTI and SUNRGBD datasets since the
    multi-view image loading is not supported yet in this inference function.

    Args:
        model (nn.Module): The loaded detector.
        pcds (str, Sequence[str]):
            Either point cloud files or loaded point cloud.
        imgs (str, Sequence[str]):
           Either image files or loaded images.
        ann_file (str, Sequence[str]): Annotation files.
        cam_type (str): Image of Camera chose to infer. When detector only uses
            single-view image, we need to specify a camera view. For kitti
            dataset, it should be 'CAM2'. For sunrgbd, it should be 'CAM0'.
            When detector uses multi-view images, we should set it to 'all'.

    Returns:
        :obj:`Det3DDataSample` or list[:obj:`Det3DDataSample`]:
        If pcds is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """
    if isinstance(pcds, (list, tuple)):
        is_batch = True
        assert isinstance(imgs, (list, tuple))
        assert len(pcds) == len(imgs)
    else:
        pcds = [pcds]
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg

    # build the data pipeline
    test_pipeline = deepcopy(cfg.test_dataloader.dataset.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = \
        get_box_type(cfg.test_dataloader.dataset.box_type_3d)

    data_list = mmengine.load(ann_file)['data_list']

    data = []
    for index, pcd in enumerate(pcds):
        # get data info containing calib
        data_info = data_list[index]
        img = imgs[index]

        if cam_type != 'all':
            assert osp.isfile(img), f'{img} must be a file.'
            img_path = data_info['images'][cam_type]['img_path']
            if osp.basename(img_path) != osp.basename(img):
                raise ValueError(
                    f'the info file of {img_path} is not provided.')
            data_ = dict(
                lidar_points=dict(lidar_path=pcd),
                img_path=img,
                box_type_3d=box_type_3d,
                box_mode_3d=box_mode_3d)
            data_info['images'][cam_type]['img_path'] = img
            if 'cam2img' in data_info['images'][cam_type]:
                # The data annotation in SRUNRGBD dataset does not contain
                # `cam2img`
                data_['cam2img'] = np.array(
                    data_info['images'][cam_type]['cam2img'])

            # LiDAR to image conversion for KITTI dataset
            if box_mode_3d == Box3DMode.LIDAR:
                if 'lidar2img' in data_info['images'][cam_type]:
                    data_['lidar2img'] = np.array(
                        data_info['images'][cam_type]['lidar2img'])
            # Depth to image conversion for SUNRGBD dataset
            elif box_mode_3d == Box3DMode.DEPTH:
                data_['depth2img'] = np.array(
                    data_info['images'][cam_type]['depth2img'])
        else:
            assert osp.isdir(img), f'{img} must be a file directory'
            for _, img_info in data_info['images'].items():
                img_info['img_path'] = osp.join(img, img_info['img_path'])
                assert osp.isfile(img_info['img_path']
                                  ), f'{img_info["img_path"]} does not exist.'
            data_ = dict(
                lidar_points=dict(lidar_path=pcd),
                images=data_info['images'],
                box_type_3d=box_type_3d,
                box_mode_3d=box_mode_3d)

        if 'timestamp' in data_info:
            # Using multi-sweeps need `timestamp`
            data_['timestamp'] = data_info['timestamp']

        data_ = test_pipeline(data_)
        data.append(data_)

    collate_data = pseudo_collate(data)

    # forward the model
    with torch.no_grad():
        results = model.test_step(collate_data)

    if not is_batch:
        return results[0], data[0]
    else:
        return results, data


def inference_mono_3d_detector(model: nn.Module,
                               imgs: ImagesType,
                               ann_file: Union[str, Sequence[str]],
                               cam_type: str = 'CAM_FRONT'):
    """Inference image with the monocular 3D detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str, Sequence[str]):
           Either image files or loaded images.
        ann_files (str, Sequence[str]): Annotation files.
        cam_type (str): Image of Camera chose to infer.
            For kitti dataset, it should be 'CAM_2',
            and for nuscenes dataset, it should be
            'CAM_FRONT'. Defaults to 'CAM_FRONT'.

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

    data_list = mmengine.load(ann_file)['data_list']
    assert len(imgs) == len(data_list)

    data = []
    for index, img in enumerate(imgs):
        # get data info containing calib
        data_info = data_list[index]
        img_path = data_info['images'][cam_type]['img_path']
        if osp.basename(img_path) != osp.basename(img):
            raise ValueError(f'the info file of {img_path} is not provided.')

        # replace the img_path in data_info with img
        data_info['images'][cam_type]['img_path'] = img
        # avoid data_info['images'] has multiple keys anout camera views.
        mono_img_info = {f'{cam_type}': data_info['images'][cam_type]}
        data_ = dict(
            images=mono_img_info,
            box_type_3d=box_type_3d,
            box_mode_3d=box_mode_3d)

        data_ = test_pipeline(data_)
        data.append(data_)

    collate_data = pseudo_collate(data)

    # forward the model
    with torch.no_grad():
        results = model.test_step(collate_data)

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

    new_test_pipeline = []
    for pipeline in test_pipeline:
        if pipeline['type'] != 'LoadAnnotations3D':
            new_test_pipeline.append(pipeline)
    test_pipeline = Compose(new_test_pipeline)

    data = []
    # TODO: support load points array
    for pcd in pcds:
        data_ = dict(lidar_points=dict(lidar_path=pcd))
        data_ = test_pipeline(data_)
        data.append(data_)

    collate_data = pseudo_collate(data)

    # forward the model
    with torch.no_grad():
        results = model.test_step(collate_data)

    if not is_batch:
        return results[0], data[0]
    else:
        return results, data
