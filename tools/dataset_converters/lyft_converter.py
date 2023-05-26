# Copyright (c) OpenMMLab. All rights reserved.
import os
from logging import warning
from os import path as osp

import mmcv
import mmengine
import numpy as np
from lyft_dataset_sdk.lyftdataset import LyftDataset as Lyft
from pyquaternion import Quaternion

from mmdet3d.datasets.convert_utils import LyftNameMapping
from .nuscenes_converter import (get_2d_boxes, get_available_scenes,
                                 obtain_sensor2top)

lyft_categories = ('car', 'truck', 'bus', 'emergency_vehicle', 'other_vehicle',
                   'motorcycle', 'bicycle', 'pedestrian', 'animal')


def create_lyft_infos(root_path,
                      info_prefix,
                      version='v1.01-train',
                      max_sweeps=10):
    """Create info file of lyft dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str, optional): Version of the data.
            Default: 'v1.01-train'.
        max_sweeps (int, optional): Max number of sweeps.
            Default: 10.
    """
    lyft = Lyft(
        data_path=osp.join(root_path, version),
        json_path=osp.join(root_path, version, version),
        verbose=True)
    available_vers = ['v1.01-train', 'v1.01-test']
    assert version in available_vers
    if version == 'v1.01-train':
        train_scenes = mmengine.list_from_file('data/lyft/train.txt')
        val_scenes = mmengine.list_from_file('data/lyft/val.txt')
    elif version == 'v1.01-test':
        train_scenes = mmengine.list_from_file('data/lyft/test.txt')
        val_scenes = []
    else:
        raise ValueError('unknown')

    # filter existing scenes.
    available_scenes = get_available_scenes(lyft)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    test = 'test' in version
    if test:
        print(f'test scene: {len(train_scenes)}')
    else:
        print(f'train scene: {len(train_scenes)}, \
                val scene: {len(val_scenes)}')
    train_lyft_infos, val_lyft_infos = _fill_trainval_infos(
        lyft, train_scenes, val_scenes, test, max_sweeps=max_sweeps)

    metadata = dict(version=version)
    if test:
        print(f'test sample: {len(train_lyft_infos)}')
        data = dict(infos=train_lyft_infos, metadata=metadata)
        info_name = f'{info_prefix}_infos_test'
        info_path = osp.join(root_path, f'{info_name}.pkl')
        mmengine.dump(data, info_path)
    else:
        print(f'train sample: {len(train_lyft_infos)}, \
                val sample: {len(val_lyft_infos)}')
        data = dict(infos=train_lyft_infos, metadata=metadata)
        train_info_name = f'{info_prefix}_infos_train'
        info_path = osp.join(root_path, f'{train_info_name}.pkl')
        mmengine.dump(data, info_path)
        data['infos'] = val_lyft_infos
        val_info_name = f'{info_prefix}_infos_val'
        info_val_path = osp.join(root_path, f'{val_info_name}.pkl')
        mmengine.dump(data, info_val_path)


def _fill_trainval_infos(lyft,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10):
    """Generate the train/val infos from the raw data.

    Args:
        lyft (:obj:`LyftDataset`): Dataset class in the Lyft dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool, optional): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int, optional): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and
            validation set that will be saved to the info file.
    """
    train_lyft_infos = []
    val_lyft_infos = []

    for sample in mmengine.track_iter_progress(lyft.sample):
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = lyft.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = lyft.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = lyft.get('ego_pose', sd_rec['ego_pose_token'])
        abs_lidar_path, boxes, _ = lyft.get_sample_data(lidar_token)
        # nuScenes devkit returns more convenient relative paths while
        # lyft devkit returns absolute paths
        abs_lidar_path = str(abs_lidar_path)  # absolute path
        lidar_path = abs_lidar_path.split(f'{os.getcwd()}/')[-1]
        # relative path

        mmengine.check_file_exist(lidar_path)

        info = {
            'lidar_path': lidar_path,
            'num_features': 5,
            'token': sample['token'],
            'sweeps': [],
            'cams': dict(),
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
        }

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # obtain 6 image's information per frame
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = lyft.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(lyft, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        sd_rec = lyft.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = obtain_sensor2top(lyft, sd_rec['prev'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                sweeps.append(sweep)
                sd_rec = lyft.get('sample_data', sd_rec['prev'])
            else:
                break
        info['sweeps'] = sweeps
        # obtain annotation
        if not test:
            annotations = [
                lyft.get('sample_annotation', token)
                for token in sample['anns']
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                             for b in boxes]).reshape(-1, 1)

            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in LyftNameMapping:
                    names[i] = LyftNameMapping[names[i]]
            names = np.array(names)

            # we need to convert box size to
            # the format of our lidar coordinate system
            # which is x_size, y_size, z_size (corresponding to l, w, h)
            gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
            assert len(gt_boxes) == len(
                annotations), f'{len(gt_boxes)}, {len(annotations)}'
            info['gt_boxes'] = gt_boxes
            info['gt_names'] = names
            info['num_lidar_pts'] = np.array(
                [a['num_lidar_pts'] for a in annotations])
            info['num_radar_pts'] = np.array(
                [a['num_radar_pts'] for a in annotations])

        if sample['scene_token'] in train_scenes:
            train_lyft_infos.append(info)
        else:
            val_lyft_infos.append(info)

    return train_lyft_infos, val_lyft_infos


def export_2d_annotation(root_path, info_path, version):
    """Export 2d annotation from the info file and raw data.

    Args:
        root_path (str): Root path of the raw data.
        info_path (str): Path of the info file.
        version (str): Dataset version.
    """
    warning.warn('DeprecationWarning: 2D annotations are not used on the '
                 'Lyft dataset. The function export_2d_annotation will be '
                 'deprecated.')
    # get bbox annotations for camera
    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]
    lyft_infos = mmengine.load(info_path)['infos']
    lyft = Lyft(
        data_path=osp.join(root_path, version),
        json_path=osp.join(root_path, version, version),
        verbose=True)
    # info_2d_list = []
    cat2Ids = [
        dict(id=lyft_categories.index(cat_name), name=cat_name)
        for cat_name in lyft_categories
    ]
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    for info in mmengine.track_iter_progress(lyft_infos):
        for cam in camera_types:
            cam_info = info['cams'][cam]
            coco_infos = get_2d_boxes(
                lyft,
                cam_info['sample_data_token'],
                visibilities=['', '1', '2', '3', '4'])
            (height, width, _) = mmcv.imread(cam_info['data_path']).shape
            coco_2d_dict['images'].append(
                dict(
                    file_name=cam_info['data_path'],
                    id=cam_info['sample_data_token'],
                    width=width,
                    height=height))
            for coco_info in coco_infos:
                if coco_info is None:
                    continue
                # add an empty key for coco format
                coco_info['segmentation'] = []
                coco_info['id'] = coco_ann_id
                coco_2d_dict['annotations'].append(coco_info)
                coco_ann_id += 1
    mmengine.dump(coco_2d_dict, f'{info_path[:-4]}.coco.json')
