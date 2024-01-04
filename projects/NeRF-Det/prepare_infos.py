# Copyright (c) OpenMMLab. All rights reserved.
"""Prepare the dataset for NeRF-Det.

Example:
    python projects/NeRF-Det/prepare_infos.py
        --root-path ./data/scannet
        --out-dir ./data/scannet
"""
import argparse
import time
from os import path as osp
from pathlib import Path

import mmengine

from ...tools.dataset_converters import indoor_converter as indoor
from ...tools.dataset_converters.update_infos_to_v2 import (
    clear_data_info_unused_keys, clear_instance_unused_keys,
    get_empty_instance, get_empty_standard_data_info)


def update_scannet_infos_nerfdet(pkl_path, out_dir):
    """Update the origin pkl to the new format which will be used in nerf-det.

    Args:
        pkl_path (str): Path of the origin pkl.
        out_dir (str): Output directory of the generated info file.

    Returns:
        The pkl will be overwritTen.
        The new pkl is a dict containing two keys:
        metainfo: Some base information of the pkl
        data_list (list): A list containing all the information of the scenes.
    """
    print('The new refactored process is running.')
    print(f'{pkl_path} will be modified.')
    if out_dir in pkl_path:
        print(f'Warning, you may overwriting '
              f'the original data {pkl_path}.')
        time.sleep(5)
    METAINFO = {
        'classes':
        ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
         'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
         'showercurtrain', 'toilet', 'sink', 'bathtub', 'garbagebin')
    }
    print(f'Reading from input file: {pkl_path}.')
    data_list = mmengine.load(pkl_path)
    print('Start updating:')
    converted_list = []
    for ori_info_dict in mmengine.track_iter_progress(data_list):
        temp_data_info = get_empty_standard_data_info()

        # intrinsics, extrinsics and imgs
        temp_data_info['cam2img'] = ori_info_dict['intrinsics']
        temp_data_info['lidar2cam'] = ori_info_dict['extrinsics']
        temp_data_info['img_paths'] = ori_info_dict['img_paths']

        # annotation information
        anns = ori_info_dict.get('annos', None)
        ignore_class_name = set()
        if anns is not None:
            temp_data_info['axis_align_matrix'] = anns[
                'axis_align_matrix'].tolist()
            if anns['gt_num'] == 0:
                instance_list = []
            else:
                num_instances = len(anns['name'])
                instance_list = []
                for instance_id in range(num_instances):
                    empty_instance = get_empty_instance()
                    empty_instance['bbox_3d'] = anns['gt_boxes_upright_depth'][
                        instance_id].tolist()

                    if anns['name'][instance_id] in METAINFO['classes']:
                        empty_instance['bbox_label_3d'] = METAINFO[
                            'classes'].index(anns['name'][instance_id])
                    else:
                        ignore_class_name.add(anns['name'][instance_id])
                        empty_instance['bbox_label_3d'] = -1

                    empty_instance = clear_instance_unused_keys(empty_instance)
                    instance_list.append(empty_instance)
            temp_data_info['instances'] = instance_list
        temp_data_info, _ = clear_data_info_unused_keys(temp_data_info)
        converted_list.append(temp_data_info)
    pkl_name = Path(pkl_path).name
    out_path = osp.join(out_dir, pkl_name)
    print(f'Writing to output file: {out_path}.')
    print(f'ignore classes: {ignore_class_name}')

    # dataset metainfo
    metainfo = dict()
    metainfo['categories'] = {k: i for i, k in enumerate(METAINFO['classes'])}
    if ignore_class_name:
        for ignore_class in ignore_class_name:
            metainfo['categories'][ignore_class] = -1
    metainfo['dataset'] = 'scannet'
    metainfo['info_version'] = '1.1'

    converted_data_info = dict(metainfo=metainfo, data_list=converted_list)

    mmengine.dump(converted_data_info, out_path, 'pkl')


def scannet_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for scannet dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
        version (str): Only used to generate the dataset of nerfdet now.
    """
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)
    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    info_test_path = osp.join(out_dir, f'{info_prefix}_infos_test.pkl')
    update_scannet_infos_nerfdet(out_dir=out_dir, pkl_path=info_train_path)
    update_scannet_infos_nerfdet(out_dir=out_dir, pkl_path=info_val_path)
    update_scannet_infos_nerfdet(out_dir=out_dir, pkl_path=info_test_path)


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/scannet',
    help='specify the root path of dataset')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/scannet',
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='scannet')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    from mmdet3d.utils import register_all_modules
    register_all_modules()

    scannet_data_prep(
        root_path=args.root_path,
        info_prefix=args.extra_tag,
        out_dir=args.out_dir,
        workers=args.workers)
