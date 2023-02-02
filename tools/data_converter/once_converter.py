# Copyright (c) OpenMMLab. All rights reserved.
r"""Adapted from `openpcdet/once_dataset
    <https://github.com/open-mmlab/openpcdet>`_.
"""

import json
import os.path as osp
from typing import List

import mmcv
import numpy as np


camera_list = ['cam01', 'cam03', 'cam05', 'cam06', 'cam07', 'cam08', 'cam09']


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [str(line).strip('\n') for line in lines]


def create_once_infos(root_path,
                      info_prefix,
                      split):
    """Create info file of once dataset.
    
    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        split (str, optional): Version of the data.
        max_sweeps (int, optional): Max number of sweeps.
            Default: 10.
    """
    assert split in ['train', 'val', 'trainval', 'test']

    imageset_path = osp.join(root_path, 'ImageSets')
    train_seqs = _read_imageset_file(osp.join(imageset_path, 'train.txt'))
    val_seqs = _read_imageset_file(osp.join(imageset_path, 'val.txt'))
    test_seqs = _read_imageset_file(osp.join(imageset_path, 'test.txt'))

    test = split == 'test'
    if test:
        print('test sequences: {}'.format(len(train_seqs)))
    else:
        print('train sequences: {}, val sequences {}'.format(
            len(train_seqs), len(val_seqs)))
    train_once_infos, val_once_infos = _fill_trainval_infos(
        root_path, train_seqs, val_seqs, test
    )

    if test:
        print(f'test frames: {len(train_once_infos)}')
        info_path = osp.join(root_path, f'{info_prefix}_infos_test.pkl')
        mmcv.dump(train_once_infos, info_path)
    else:
        print(f'train frames: {len(train_once_infos)}, \
                val frames: {len(val_once_infos)}')
        info_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
        mmcv.dump(train_once_infos, info_path)
        info_val_path = osp.join(root_path, f'{info_prefix}_infos_val.pkl')
        mmcv.dump(val_once_infos, info_val_path)


def _fill_trainval_infos(
        root_path: str,
        train_seqs: List[str],
        val_seqs: List[str],
        test = False,
):
    """Generate the train/val infos from the raw data.

    Args:
        root_path (str): path to ImageSets and data
        train_seqs (list[str]): Basic information of training sequences.
        val_seqs (list[str]): Basic information of validation sequences.
        test (bool, optional): Whether use the test mode. In test mode, no
            annotations can be accessed. Default: False.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    train_once_infos = []
    val_once_infos = []

    for seq_id in (train_seqs + val_seqs):
        # last line of txt file '\n' introduce '' to seqs list sometimes
        if seq_id == '':
            continue
        seq_path = osp.join(root_path, 'data', seq_id)
        json_path = osp.join(seq_path, '{}.json'.format(seq_id))
        with open(json_path, 'r') as f:
            json_seq = json.load(f)
        meta_info = json_seq['meta_info']
        calib = json_seq['calib']
        for f_idx, frame in enumerate(json_seq['frames']):
            frame_id = frame['frame_id']
            print(f'Process seq_id: {seq_id}, frame_id: {frame_id}')
            if f_idx == 0:
                prev_id = None
            else:
                prev_id = json_seq['frames'][f_idx-1]['frame_id']
            if f_idx == len(json_seq['frames'])-1:
                next_id = None
            else:
                next_id = json_seq['frames'][f_idx+1]['frame_id']
            lidar_path = osp.join(seq_path, 'lidar_roof', '{}.bin'.format(frame_id))
            mmcv.check_file_exist(lidar_path)
            pose = np.array(frame['pose'])
            frame_dict = {
                'sequence_id': seq_id,
                'frame_id': frame_id,
                'timestamp': int(frame_id),
                'prev_id': prev_id,
                'next_id': next_id,
                'meta_info': meta_info,
                'lidar_path': lidar_path,
                'pose': pose
            }
            calib_dict = {}
            for camera in camera_list:
                img_path = osp.join(seq_path, camera, '{}.jpg'.format(frame_id))
                frame_dict.update({camera: img_path})
                calib_dict[camera] = {}
                calib_dict[camera]['cam_to_velo'] = np.array(calib[camera]['cam_to_velo'])
                calib_dict[camera]['cam_intrinsic'] = np.array(calib[camera]['cam_intrinsic'])
                calib_dict[camera]['distortion'] = np.array(calib[camera]['distortion'])
            frame_dict.update({'calib': calib_dict})

            if 'annos' in frame:
                annos = frame['annos']
                names = annos['names']
                if len(names) == 0:
                    print(f'Skipping {frame_id} since no objects.')
                    continue
                boxes_3d = np.array(annos['boxes_3d'])
                boxes_2d_dict = {}
                for camera in camera_list:
                    boxes_2d_dict[camera] = np.array(annos['boxes_2d'][camera])
                annos_dict = {
                    'name': np.array(annos['names']),
                    'boxes_3d': boxes_3d,
                    'boxes_2d': boxes_2d_dict
                }

                frame_dict.update({'annos': annos_dict})

            if seq_id in train_seqs:
                train_once_infos.append(frame_dict)
            if seq_id in val_seqs:
                val_once_infos.append(frame_dict)

    return train_once_infos, val_once_infos