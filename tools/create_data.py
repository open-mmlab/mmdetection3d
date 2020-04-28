import argparse
import os.path as osp

import tools.data_converter.indoor_converter as indoor
import tools.data_converter.kitti_converter as kitti
import tools.data_converter.nuscenes_converter as nuscenes_converter
from tools.data_converter.create_gt_database import create_groundtruth_database


def kitti_data_prep(root_path, info_prefix, version, out_dir):
    kitti.create_kitti_info_file(root_path, info_prefix)
    kitti.create_reduced_point_cloud(root_path, info_prefix)
    create_groundtruth_database(
        'KittiDataset',
        root_path,
        info_prefix,
        '{}/{}_infos_train.pkl'.format(out_dir, info_prefix),
        relative_path=False,
        mask_anno_path='instances_train.json',
        with_mask=(version == 'mask'))


def nuscenes_data_prep(root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10):
    nuscenes_converter.create_nuscenes_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)

    if version == 'v1.0-test':
        return

    info_train_path = osp.join(root_path,
                               '{}_infos_train.pkl'.format(info_prefix))
    info_val_path = osp.join(root_path, '{}_infos_val.pkl'.format(info_prefix))
    nuscenes_converter.export_2d_annotation(
        root_path, info_train_path, version=version)
    nuscenes_converter.export_2d_annotation(
        root_path, info_val_path, version=version)
    create_groundtruth_database(
        dataset_name, root_path, info_prefix,
        '{}/{}_infos_train.pkl'.format(out_dir, info_prefix))


def scannet_data_prep(root_path, info_prefix, out_dir):
    indoor.create_indoor_info_file(root_path, info_prefix, out_dir)


def sunrgbd_data_prep(root_path, info_prefix, out_dir):
    indoor.create_indoor_info_file(root_path, info_prefix, out_dir)


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
    '--out-dir',
    type=str,
    default='./data/kitti',
    required='False',
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
args = parser.parse_args()

if __name__ == '__main__':
    if args.dataset == 'kitti':
        kitti_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir)
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
    elif args.dataset == 'scannet':
        scannet_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir)
    elif args.dataset == 'sunrgbd':
        sunrgbd_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir)
