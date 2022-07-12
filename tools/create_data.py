# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from os import path as osp


from tools.data_converter import indoor_converter as indoor
from tools.data_converter import kitti_converter as kitti
from tools.data_converter import lyft_converter as lyft_converter
from tools.data_converter import nuscenes_converter as nuscenes_converter
from tools.data_converter.create_gt_database import (
    GTDatabaseCreater, create_groundtruth_database)

from pypcd import pypcd
import numpy as np

def read_PC(pcd_file):
    """
        parameter
            .pcd format file.

        return
            np.array, nx3, xyz coordinates of given pointcloud points.
    """
    points_pcd = pypcd.PointCloud.from_path(pcd_file)
    x = points_pcd.pc_data["x"].copy()
    y = points_pcd.pc_data["y"].copy()
    z = points_pcd.pc_data["z"].copy()
    return np.array([x,y,z]).T

def cross(a, b):
    """
        Cross product of 2d points.

        parameter
            a: np.array, size nx2, point vectors to check 
            b: np.array, size 1x2, line vectors  
        
        return
            c: np.array, size nx2, 
            if c[i] is positive, a[i] is to the right of b, 
            elif c[i] is negative, a[i] is to the left of b. 
    """
    c = a[:,0]*b[1] - a[:,1]*b[0]
    return c 

def check_inside_convex_polygon(points, vertices):
    """
        The function to check indices of points which are inside of a given convex polygon.
        
        parameter
            points: np.array, size nx2, 2d points vectors.
            vertices: np.array, size mx2, vertices of convex polygon, assume that the order of vertices is unknown. 
        
        return 
            indices of the points that are inside of the given convex polygon.
    """
    roi1, roi2 = True, True
    vertex_num = len(vertices)
    for i, v in enumerate(vertices):
        roi1 &= (cross(points - v, vertices[(i + 1)%vertex_num] - v) > 0)
        roi2 &= (cross(points - v, vertices[(i + 1)%vertex_num] - v) < 0)
    return roi1 | roi2

def get_original_box_xy_coords(box_info):
    cx,cy,cz,w,l,h,theta = box_info
    rotation_mtx_T = np.array([[np.cos(theta),np.sin(theta)],
                                [-np.sin(theta),np.cos(theta)]])
    xys = np.array([[-w/2, -l/2],[w/2, -l/2],[w/2, l/2],[-w/2, l/2]])
    rotated_xys = xys @ rotation_mtx_T
    rotated_xys[:,0] += cx
    rotated_xys[:,1] += cy
    return rotated_xys

def get_estimated_z_h(roi_path, box_annot):
    points_xyz = read_PC(roi_path)
    cz_list,h_list = [], []
    Z_MIN_DEFALUT = -1
    H_DEFAULT = 2.5
    PED_H_DEFAULT = 1.7
    for single_annot in box_annot:
        label_cls = single_annot[0]
        box_coor = single_annot[1:].astype(np.float32)
        rotated_xys = get_original_box_xy_coords(box_coor)
        roi_points = points_xyz[check_inside_convex_polygon(points_xyz[:, :2], rotated_xys)]
        try:
            q75, q25 = np.percentile(roi_points[:, 2], [75,25])
            iqr = q75 - q25
            upper_valid = roi_points[:, 2] <= q75 + 1.5 * iqr
            lower_valid = roi_points[:, 2] >= q25 - 1.5 * iqr
            roi_points = roi_points[upper_valid & lower_valid]
            q95, q5 = np.percentile(roi_points[:, 2], [95,5])
            z1, z2 = q5-0.3, q95+0.3
            h = H_DEFAULT if label_cls != 'Pedestrian' else PED_H_DEFAULT
            # 이 부분을 object class에 따라 구분
            if abs(z2 - z1) < 1.5:
                z1 = z2 - h
            elif abs(z2 - z1) > 4.2:
                z2 = z1 + h
            # z1, z2 = min(roi_points[:,2]), max(roi_points[:,2])
            cz, h = int((z1+z2)/2 * 100)/100, int(abs(z2-z1)*100)/100
        except:
            # edge case, if xy-plane labeling is invalid, roi_buf could be empty.
            # then just set z, h as default value
            cz, h = Z_MIN_DEFALUT, H_DEFAULT
            if label_cls == "Pedestrian":
                h = PED_H_DEFAULT
        cz_list.append(cz)
        h_list.append(h)
    return np.array(cz_list), np.array(h_list)

def rf2021_data_prep(root_path,
                     info_prefix):
    """ Prepare data related to RF2021 dataset.

    1. loop pcd file directory.
    2. vehicle (car) 과 Pedestrian의 label의 형태를 (cls, x, y, z, dx, dy, dz, theta)로 통일
        * 각도를 바꾸는 이유는 우리 데이터셋의 좌표계와 mmdetection3d의 좌표계가 갖는 각도에 대한 기준이 달라서 이를 맞춰줌
    3. z값이 제대로 안만들어져 있는 경우에는 해당 box내에 포인트들을 적절히 봐서 임의로 z 값 채움
    4. custom3DDataset의 형태로 저장. train,val,test 나눠서 pickle 파일로 저장. 

    """
    import os
    from pathlib import Path
    import numpy as np
    import mmcv
    from collections import deque
    from tqdm import tqdm
    import math

    root_path = Path(root_path)

    pcd_dir = osp.join(root_path, "NIA_tracking_data", "data")
    label_dir = osp.join(root_path, "NIA_2021_label", "label")
    sample_idx = 0
    annot_deque = deque([])
    if osp.isdir(pcd_dir):
        folder_list = sorted(os.listdir(pcd_dir), key=lambda x:int(x))
        for fol in tqdm(folder_list):
            pcd_data_dir = osp.join(pcd_dir, fol, "lidar_half_filtered")
            veh_label_dir = osp.join(label_dir, fol, "car_label")
            ped_label_dir = osp.join(label_dir, fol, "ped_label")
            for pcd_file in sorted(os.listdir(pcd_data_dir)):
                pcd_file_path = osp.join(pcd_data_dir, pcd_file)
                veh_label_file_path = osp.join(veh_label_dir, pcd_file[:-4] + ".txt")
                ped_label_file_path = osp.join(ped_label_dir, pcd_file[:-4] + ".txt")
                annot = np.array([])
                if osp.exists(veh_label_file_path):
                    annot = np.loadtxt(veh_label_file_path, dtype=np.unicode_).reshape(-1, 8)
                    annot[:, [1,2,3,4,5,6]] = annot[:, [4,5,6,2,1,3]]
                    annot[:, 7] = math.pi/2 - annot[:, 7].astype(np.float32)
                    annot[annot == 'nan'] = '-1.00'
                if osp.exists(ped_label_file_path):
                    annot_ped = np.loadtxt(ped_label_file_path, dtype=np.unicode_).reshape(-1, 6)
                    if len(annot_ped) > 0:
                        annot_ped[annot_ped == 'nan'] = '-1.00'
                        annot_ped[annot_ped[:, 3] == '-1.00', 0] = '0.7'
                        annot_ped[annot_ped[:, 4] ==  '-1.00', 1] = '0.7'  
                        annot_cls = np.array([["Pedestrian"] for _ in range(len(annot_ped))])
                        annot_angle = np.array([[0] for _ in range(len(annot_ped))])
                        annot_ped = np.hstack((annot_cls, annot_ped, annot_angle))
                        annot = np.vstack((annot, annot_ped))
                
                if len(annot):
                    invalid_cond = (annot[:, 3] == '-1.00') & (annot[:, 6] == '-1.00')
                    cz, h = get_estimated_z_h(pcd_file_path, annot[invalid_cond])
                    annot[invalid_cond, 3] = cz
                    annot[invalid_cond, 6] = h
                    pcd_file_path = "/".join(pcd_file_path.split("/")[2:])
                    annot_dict = dict(
                        sample_idx= sample_idx,
                        lidar_points= {'lidar_path': pcd_file_path},
                        annos= {'box_type_3d': 'LiDAR',
                                'gt_bboxes_3d': annot[:, 1:].astype(np.float32),
                                'gt_names': annot[:, 0]
                                }
                    )
                    annot_deque.append(annot_dict)
                    sample_idx += 1
    else:
        print("no data dir")
        print("Please check data dir path")
        exit()

    annot_list = list(annot_deque)
    total_len = len(annot_list)
    train_len = int(total_len * 0.8)
    val_len = int(total_len * 0.1)
    rf_infos_train = annot_list[:train_len]
    filename = root_path / f'{info_prefix}_infos_train.pkl'
    print(f'RF2021 info train file is saved to {filename}')
    mmcv.dump(rf_infos_train, filename)

    rf_infos_val = annot_list[train_len:train_len + val_len]
    filename = root_path / f'{info_prefix}_infos_val.pkl'
    print(f'RF2021 info val file is saved to {filename}')
    mmcv.dump(rf_infos_val, filename)

    rf_infos_test = annot_list[train_len + val_len:]
    filename = root_path / f'{info_prefix}_infos_test.pkl'
    print(f'RF2021 info test file is saved to {filename}')
    mmcv.dump(rf_infos_test, filename)

    create_groundtruth_database('Custom3DDataset', root_path, info_prefix,
                            root_path / f'{info_prefix}_infos_train.pkl')


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

    info_train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(root_path, f'{info_prefix}_infos_val.pkl')
    info_trainval_path = osp.join(root_path,
                                  f'{info_prefix}_infos_trainval.pkl')
    info_test_path = osp.join(root_path, f'{info_prefix}_infos_test.pkl')
    kitti.export_2d_annotation(root_path, info_train_path)
    kitti.export_2d_annotation(root_path, info_val_path)
    kitti.export_2d_annotation(root_path, info_trainval_path)
    kitti.export_2d_annotation(root_path, info_test_path)

    create_groundtruth_database(
        'KittiDataset',
        root_path,
        info_prefix,
        f'{out_dir}/{info_prefix}_infos_train.pkl',
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
        info_test_path = osp.join(root_path, f'{info_prefix}_infos_test.pkl')
        nuscenes_converter.export_2d_annotation(
            root_path, info_test_path, version=version)
        return

    info_train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(root_path, f'{info_prefix}_infos_val.pkl')
    nuscenes_converter.export_2d_annotation(
        root_path, info_train_path, version=version)
    nuscenes_converter.export_2d_annotation(
        root_path, info_val_path, version=version)
    create_groundtruth_database(dataset_name, root_path, info_prefix,
                                f'{out_dir}/{info_prefix}_infos_train.pkl')


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
    from tools.data_converter import waymo_converter as waymo

    splits = ['training', 'validation', 'testing']
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
            test_mode=(split == 'testing'))
        converter.convert()
    # Generate waymo infos
    out_dir = osp.join(out_dir, 'kitti_format')
    kitti.create_waymo_info_file(
        out_dir, info_prefix, max_sweeps=max_sweeps, workers=workers)
    GTDatabaseCreater(
        'WaymoDataset',
        out_dir,
        info_prefix,
        f'{out_dir}/{info_prefix}_infos_train.pkl',
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
    if args.dataset == 'kitti':
        kitti_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir,
            with_plane=args.with_plane)
    elif args.dataset == 'rf2021':
        rf2021_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag
        )
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
