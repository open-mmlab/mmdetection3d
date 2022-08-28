# Copyright (c) OpenMMLab. All rights reserved.
import os

import mmengine
import numpy as np

from tools.dataset_converters.s3dis_data_utils import S3DISData, S3DISSegData
from tools.dataset_converters.scannet_data_utils import (ScanNetData,
                                                         ScanNetSegData)
from tools.dataset_converters.sunrgbd_data_utils import SUNRGBDData


def create_indoor_info_file(data_path,
                            pkl_prefix='sunrgbd',
                            save_path=None,
                            use_v1=False,
                            workers=4):
    """Create indoor information file.

    Get information of the raw data and save it to the pkl file.

    Args:
        data_path (str): Path of the data.
        pkl_prefix (str, optional): Prefix of the pkl to be saved.
            Default: 'sunrgbd'.
        save_path (str, optional): Path of the pkl to be saved. Default: None.
        use_v1 (bool, optional): Whether to use v1. Default: False.
        workers (int, optional): Number of threads to be used. Default: 4.
    """
    assert os.path.exists(data_path)
    assert pkl_prefix in ['sunrgbd', 'scannet', 's3dis'], \
        f'unsupported indoor dataset {pkl_prefix}'
    save_path = data_path if save_path is None else save_path
    assert os.path.exists(save_path)

    # generate infos for both detection and segmentation task
    if pkl_prefix in ['sunrgbd', 'scannet']:
        train_filename = os.path.join(save_path,
                                      f'{pkl_prefix}_infos_train.pkl')
        val_filename = os.path.join(save_path, f'{pkl_prefix}_infos_val.pkl')
        if pkl_prefix == 'sunrgbd':
            # SUN RGB-D has a train-val split
            train_dataset = SUNRGBDData(
                root_path=data_path, split='train', use_v1=use_v1)
            val_dataset = SUNRGBDData(
                root_path=data_path, split='val', use_v1=use_v1)
        else:
            # ScanNet has a train-val-test split
            train_dataset = ScanNetData(root_path=data_path, split='train')
            val_dataset = ScanNetData(root_path=data_path, split='val')
            test_dataset = ScanNetData(root_path=data_path, split='test')
            test_filename = os.path.join(save_path,
                                         f'{pkl_prefix}_infos_test.pkl')

        infos_train = train_dataset.get_infos(
            num_workers=workers, has_label=True)
        mmengine.dump(infos_train, train_filename, 'pkl')
        print(f'{pkl_prefix} info train file is saved to {train_filename}')

        infos_val = val_dataset.get_infos(num_workers=workers, has_label=True)
        mmengine.dump(infos_val, val_filename, 'pkl')
        print(f'{pkl_prefix} info val file is saved to {val_filename}')

    if pkl_prefix == 'scannet':
        infos_test = test_dataset.get_infos(
            num_workers=workers, has_label=False)
        mmengine.dump(infos_test, test_filename, 'pkl')
        print(f'{pkl_prefix} info test file is saved to {test_filename}')

    # generate infos for the semantic segmentation task
    # e.g. re-sampled scene indexes and label weights
    # scene indexes are used to re-sample rooms with different number of points
    # label weights are used to balance classes with different number of points
    if pkl_prefix == 'scannet':
        # label weight computation function is adopted from
        # https://github.com/charlesq34/pointnet2/blob/master/scannet/scannet_dataset.py#L24
        train_dataset = ScanNetSegData(
            data_root=data_path,
            ann_file=train_filename,
            split='train',
            num_points=8192,
            label_weight_func=lambda x: 1.0 / np.log(1.2 + x))
        # TODO: do we need to generate on val set?
        val_dataset = ScanNetSegData(
            data_root=data_path,
            ann_file=val_filename,
            split='val',
            num_points=8192,
            label_weight_func=lambda x: 1.0 / np.log(1.2 + x))
        # no need to generate for test set
        train_dataset.get_seg_infos()
        val_dataset.get_seg_infos()
    elif pkl_prefix == 's3dis':
        # S3DIS doesn't have a fixed train-val split
        # it has 6 areas instead, so we generate info file for each of them
        # in training, we will use dataset to wrap different areas
        splits = [f'Area_{i}' for i in [1, 2, 3, 4, 5, 6]]
        for split in splits:
            dataset = S3DISData(root_path=data_path, split=split)
            info = dataset.get_infos(num_workers=workers, has_label=True)
            filename = os.path.join(save_path,
                                    f'{pkl_prefix}_infos_{split}.pkl')
            mmengine.dump(info, filename, 'pkl')
            print(f'{pkl_prefix} info {split} file is saved to {filename}')
            seg_dataset = S3DISSegData(
                data_root=data_path,
                ann_file=filename,
                split=split,
                num_points=4096,
                label_weight_func=lambda x: 1.0 / np.log(1.2 + x))
            seg_dataset.get_seg_infos()
