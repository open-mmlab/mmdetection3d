# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from os import path as osp

from tools.dataset_converters import indoor_converter as indoor
from tools.dataset_converters import kitti_converter as kitti
from tools.dataset_converters import lyft_converter as lyft_converter
from tools.dataset_converters import nuscenes_converter as nuscenes_converter
from tools.dataset_converters.create_gt_database import (
    GTDatabaseCreater, create_groundtruth_database)
from tools.dataset_converters.update_infos_to_v2 import update_pkl_infos


def kitti_data_prep(root_path,
                    info_prefix,
                    version,
                    out_dir,
                    with_plane=False):
    """Prepare data related to Kitti dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        out_dir (str): Output directory of the groundtruth database info.
        with_plane (bool, optional): Whether to use plane information.
            Default: False.
    """
    kitti.create_kitti_info_file(root_path, info_prefix, with_plane)
    kitti.create_reduced_point_cloud(root_path, info_prefix)

    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    info_trainval_path = osp.join(out_dir, f'{info_prefix}_infos_trainval.pkl')
    update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_train_path)
    update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_val_path)
    update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_trainval_path)
    create_groundtruth_database(
        'KittiDataset',
        root_path,
        info_prefix,
        f'{info_prefix}_infos_train.pkl',
        relative_path=False,
        mask_anno_path='instances_train.json',
        with_mask=(version == 'mask'))


def nuscenes_data_prep(root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 10
    """
    nuscenes_converter.create_nuscenes_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)

    if version == 'v1.0-test':
        info_test_path = osp.join(out_dir, f'{info_prefix}_infos_test.pkl')
        update_pkl_infos('nuscenes', out_dir=out_dir, pkl_path=info_test_path)
        return

    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    update_pkl_infos('nuscenes', out_dir=out_dir, pkl_path=info_train_path)
    update_pkl_infos('nuscenes', out_dir=out_dir, pkl_path=info_val_path)
    create_groundtruth_database(dataset_name, root_path, info_prefix,
                                f'{info_prefix}_infos_train.pkl')


def lyft_data_prep(root_path, info_prefix, version, max_sweeps=10):
    """Prepare data related to Lyft dataset.

    Related data consists of '.pkl' files recording basic infos.
    Although the ground truth database and 2D annotations are not used in
    Lyft, it can also be generated like nuScenes.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        max_sweeps (int, optional): Number of input consecutive frames.
            Defaults to 10.
    """
    lyft_converter.create_lyft_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)
    if version == 'v1.01-test':
        info_test_path = osp.join(root_path, f'{info_prefix}_infos_test.pkl')
        update_pkl_infos('lyft', out_dir=root_path, pkl_path=info_test_path)
    elif version == 'v1.01-train':
        info_train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
        info_val_path = osp.join(root_path, f'{info_prefix}_infos_val.pkl')
        update_pkl_infos('lyft', out_dir=root_path, pkl_path=info_train_path)
        update_pkl_infos('lyft', out_dir=root_path, pkl_path=info_val_path)


def scannet_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for scannet dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)
    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_test_path = osp.join(out_dir, f'{info_prefix}_infos_test.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    update_pkl_infos('scannet', out_dir=out_dir, pkl_path=info_train_path)
    update_pkl_infos('scannet', out_dir=out_dir, pkl_path=info_test_path)
    update_pkl_infos('scannet', out_dir=out_dir, pkl_path=info_val_path)


def s3dis_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for s3dis dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)
    splits = [f'Area_{i}' for i in [1, 2, 3, 4, 5, 6]]
    for split in splits:
        filename = osp.join(out_dir, f'{info_prefix}_infos_{split}.pkl')
        update_pkl_infos('s3dis', out_dir=out_dir, pkl_path=filename)


def sunrgbd_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for sunrgbd dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)
    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    update_pkl_infos('scannet', out_dir=out_dir, pkl_path=info_train_path)
    update_pkl_infos('scannet', out_dir=out_dir, pkl_path=info_val_path)


def waymo_data_prep(root_path,
                    info_prefix,
                    version,
                    out_dir,
                    workers,
                    max_sweeps=5):
    """Prepare the info file for waymo dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 5. Here we store pose information of these frames
            for later use.
    """
    from tools.dataset_converters import waymo_converter as waymo

    splits = [
        'training', 'validation', 'testing', 'testing_3d_camera_only_detection'
    ]
    for i, split in enumerate(splits):
        load_dir = osp.join(root_path, 'waymo_format', split)
        if split == 'validation':
            save_dir = osp.join(out_dir, 'kitti_format', 'training')
        else:
            save_dir = osp.join(out_dir, 'kitti_format', split)
        converter = waymo.Waymo2KITTI(
            load_dir,
            save_dir,
            prefix=str(i),
            workers=workers,
            test_mode=(split
                       in ['testing', 'testing_3d_camera_only_detection']))
        converter.convert()
    # Generate waymo infos
    out_dir = osp.join(out_dir, 'kitti_format')
    kitti.create_waymo_info_file(
        out_dir, info_prefix, max_sweeps=max_sweeps, workers=workers)
    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    info_trainval_path = osp.join(out_dir, f'{info_prefix}_infos_trainval.pkl')
    update_pkl_infos('waymo', out_dir=out_dir, pkl_path=info_train_path)
    update_pkl_infos('waymo', out_dir=out_dir, pkl_path=info_val_path)
    update_pkl_infos('waymo', out_dir=out_dir, pkl_path=info_trainval_path)
    GTDatabaseCreater(
        'WaymoDataset',
        out_dir,
        info_prefix,
        f'{info_prefix}_infos_train.pkl',
        relative_path=False,
        with_mask=False,
        num_worker=workers).create()


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--with-plane',
    action='store_true',
    help='Whether to use plane information for kitti.')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    from mmdet3d.utils import register_all_modules
    register_all_modules()

    # Set to spawn mode to avoid stuck when process dataset creating
    import multiprocessing
    multiprocessing.set_start_method('spawn')

    if args.dataset == 'kitti':
        kitti_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir,
            with_plane=args.with_plane)
    elif args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        train_version = f'{args.version}-trainval'
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
        test_version = f'{args.version}-test'
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=test_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'nuscenes' and args.version == 'v1.0-mini':
        train_version = f'{args.version}'
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'lyft':
        train_version = f'{args.version}-train'
        lyft_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            max_sweeps=args.max_sweeps)
        test_version = f'{args.version}-test'
        lyft_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=test_version,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'waymo':
        waymo_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir,
            workers=args.workers,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'scannet':
        scannet_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 's3dis':
        s3dis_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 'sunrgbd':
        sunrgbd_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    else:
        raise NotImplementedError(f'Don\'t support {args.dataset} dataset.')
