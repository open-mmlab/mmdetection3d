import argparse
import time
from os import path as osp

import mmengine
import numpy as np

from mmdet3d.structures import limit_period


def update_sunrgbd_infos(root_dir, out_dir, pkl_files):
    print(f'{pkl_files} will be modified because '
          f'of the refactor of the Depth coordinate system.')
    if root_dir == out_dir:
        print(f'Warning, you are overwriting '
              f'the original data under {root_dir}.')
        time.sleep(3)
    for pkl_file in pkl_files:
        in_path = osp.join(root_dir, pkl_file)
        print(f'Reading from input file: {in_path}.')
        a = mmengine.load(in_path)
        print('Start updating:')
        for item in mmengine.track_iter_progress(a):
            if 'rotation_y' in item['annos']:
                item['annos']['rotation_y'] = -item['annos']['rotation_y']
                item['annos']['gt_boxes_upright_depth'][:, -1:] = \
                    -item['annos']['gt_boxes_upright_depth'][:, -1:]

        out_path = osp.join(out_dir, pkl_file)
        print(f'Writing to output file: {out_path}.')
        mmengine.dump(a, out_path, 'pkl')


def update_outdoor_dbinfos(root_dir, out_dir, pkl_files):
    print(f'{pkl_files} will be modified because '
          f'of the refactor of the LIDAR coordinate system.')
    if root_dir == out_dir:
        print(f'Warning, you are overwriting '
              f'the original data under {root_dir}.')
        time.sleep(3)
    for pkl_file in pkl_files:
        in_path = osp.join(root_dir, pkl_file)
        print(f'Reading from input file: {in_path}.')
        a = mmengine.load(in_path)
        print('Start updating:')
        for k in a.keys():
            print(f'Updating samples of class {k}:')
            for item in mmengine.track_iter_progress(a[k]):
                boxes = item['box3d_lidar'].copy()
                # swap l, w (or dx, dy)
                item['box3d_lidar'][3] = boxes[4]
                item['box3d_lidar'][4] = boxes[3]
                # change yaw
                item['box3d_lidar'][6] = -boxes[6] - np.pi / 2
                item['box3d_lidar'][6] = limit_period(
                    item['box3d_lidar'][6], period=np.pi * 2)

        out_path = osp.join(out_dir, pkl_file)
        print(f'Writing to output file: {out_path}.')
        mmengine.dump(a, out_path, 'pkl')


def update_nuscenes_or_lyft_infos(root_dir, out_dir, pkl_files):

    print(f'{pkl_files} will be modified because '
          f'of the refactor of the LIDAR coordinate system.')
    if root_dir == out_dir:
        print(f'Warning, you are overwriting '
              f'the original data under {root_dir}.')
        time.sleep(3)
    for pkl_file in pkl_files:
        in_path = osp.join(root_dir, pkl_file)
        print(f'Reading from input file: {in_path}.')
        a = mmengine.load(in_path)
        print('Start updating:')
        for item in mmengine.track_iter_progress(a['infos']):
            boxes = item['gt_boxes'].copy()
            # swap l, w (or dx, dy)
            item['gt_boxes'][:, 3] = boxes[:, 4]
            item['gt_boxes'][:, 4] = boxes[:, 3]
            # change yaw
            item['gt_boxes'][:, 6] = -boxes[:, 6] - np.pi / 2
            item['gt_boxes'][:, 6] = limit_period(
                item['gt_boxes'][:, 6], period=np.pi * 2)

        out_path = osp.join(out_dir, pkl_file)
        print(f'Writing to output file: {out_path}.')
        mmengine.dump(a, out_path, 'pkl')


parser = argparse.ArgumentParser(description='Arg parser for data coords '
                                 'update due to coords sys refactor.')
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
parser.add_argument(
    '--root-dir',
    type=str,
    default='./data/kitti',
    help='specify the root dir of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--out-dir',
    type=str,
    default=None,
    required=False,
    help='name of info pkl')
args = parser.parse_args()

if __name__ == '__main__':
    if args.out_dir is None:
        args.out_dir = args.root_dir
    if args.dataset == 'kitti':
        # KITTI infos is in CAM coord sys (unchanged)
        # KITTI dbinfos is in LIDAR coord sys (changed)
        # so we only update dbinfos
        pkl_files = ['kitti_dbinfos_train.pkl']
        update_outdoor_dbinfos(
            root_dir=args.root_dir, out_dir=args.out_dir, pkl_files=pkl_files)
    elif args.dataset == 'nuscenes':
        # nuScenes infos is in LIDAR coord sys (changed)
        # nuScenes dbinfos is in LIDAR coord sys (changed)
        # so we update both infos and dbinfos
        pkl_files = ['nuscenes_infos_val.pkl']
        if args.version != 'v1.0-mini':
            pkl_files.append('nuscenes_infos_train.pkl')
        else:
            pkl_files.append('nuscenes_infos_train_tiny.pkl')
        update_nuscenes_or_lyft_infos(
            root_dir=args.root_dir, out_dir=args.out_dir, pkl_files=pkl_files)
        if args.version != 'v1.0-mini':
            pkl_files = ['nuscenes_dbinfos_train.pkl']
            update_outdoor_dbinfos(
                root_dir=args.root_dir,
                out_dir=args.out_dir,
                pkl_files=pkl_files)
    elif args.dataset == 'lyft':
        # Lyft infos is in LIDAR coord sys (changed)
        # Lyft has no dbinfos
        # so we update infos
        pkl_files = ['lyft_infos_train.pkl', 'lyft_infos_val.pkl']
        update_nuscenes_or_lyft_infos(
            root_dir=args.root_dir, out_dir=args.out_dir, pkl_files=pkl_files)
    elif args.dataset == 'waymo':
        # Waymo infos is in CAM coord sys (unchanged)
        # Waymo dbinfos is in LIDAR coord sys (changed)
        # so we only update dbinfos
        pkl_files = ['waymo_dbinfos_train.pkl']
        update_outdoor_dbinfos(
            root_dir=args.root_dir, out_dir=args.out_dir, pkl_files=pkl_files)
    elif args.dataset == 'scannet':
        # ScanNet infos is in DEPTH coord sys (changed)
        # but bbox is without yaw
        # so ScanNet is unaffected
        pass
    elif args.dataset == 's3dis':
        # Segmentation datasets are not affected
        pass
    elif args.dataset == 'sunrgbd':
        # SUNRGBD infos is in DEPTH coord sys (changed)
        # and bbox is with yaw
        # so we update infos
        pkl_files = ['sunrgbd_infos_train.pkl', 'sunrgbd_infos_val.pkl']
        update_sunrgbd_infos(
            root_dir=args.root_dir, out_dir=args.out_dir, pkl_files=pkl_files)
