# Copyright (c) OpenMMLab. All rights reserved.
f"""Partly adapted from `once_devkit
    <https://github.com/once-for-auto-driving/once_devkit>`
"""

import shutil
import json
import functools
import numpy as np
import os.path as osp
from collections import defaultdict

import mmcv
from glob import glob
from os.path import join
from typing import List

# TODO: reformat code
def split_info_loader_helper(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        split_file_path = func(*args, **kwargs)
        if not osp.isfile(split_file_path):
            split_list = []
        else:
            split_list = set(map(lambda x: x.strip(), open(split_file_path).readlines()))
        return split_list
    return wrapper


class ONCE2KITTI(object):
    """Once to KITTI converter.

    This class serves as the converter to change the waymo raw data to KITTI
    format.

    Args:
        load_dir (str): Directory to load waymo raw data.
        save_dir (str): Directory to save data in KITTI format.
        prefix (str): Prefix of filename. In general, 0 for training, 1 for
            validation and 2 for testing.
        workers (int, optional): Number of workers for the parallel process.
        test_mode (bool, optional): Whether in the test_mode. Default: False.
    """
    def __init__(self,
                 load_dir,
                 save_dir,
                 prefix,
                 workers=64,
                 test_mode=False):
        self.filter_empy_3dboxes = True
        self.filter_no_label_zone_points = True

        self.selected_once_classes = ['Car', 'Bux', 'Truck', 'Pedestrian', 'Cyclist']
        self.once_to_kitti_class_map = {
            'Car': 'Car',
            'Bus': 'Bus',       # not in kitti
            'Truck': 'Truck',   # not in kitti
            'Pedestrian': 'Pedestrian',
            'Cyclist': 'Cyclist'
        }

        self.load_dir = load_dir
        self.save_dir = save_dir
        self.prefix = prefix
        self.workers = int(workers)
        self.test_mode = test_mode

        self.sequence_paths = sorted(
            glob(join(self.load_dir, 'data/*')))

        self.label_save_dir = f'{self.save_dir}/label_'
        self.image_save_dir = f'{self.save_dir}/image_'
        self.calib_save_dir = f'{self.save_dir}/calib'
        self.point_cloud_save_dir = f'{self.save_dir}/velodyne'
        self.pose_save_dir = f'{self.save_dir}/pose'
        self.timestamp_save_dir = f'{self.save_dir}/timestamp'

        self.camera_list = ['cam01', 'cam03', 'cam05', 'cam06', 'cam07', 'cam08', 'cam09']
        self.create_folder()
        self._collect_basic_infos()

    @property
    @split_info_loader_helper
    def train_split_list(self):
        return osp.join(self.dataset_root, 'ImageSets', 'train.txt')

    @property
    @split_info_loader_helper
    def val_split_list(self):
        return osp.join(self.dataset_root, 'ImageSets', 'val.txt')

    @property
    @split_info_loader_helper
    def test_split_list(self):
        return osp.join(self.dataset_root, 'ImageSets', 'test.txt')

    def _find_split_name(self, seq_id):
        if seq_id in self.train_split_list:
            return 'train'
        elif seq_id in self.test_split_list:
            return 'test'
        elif seq_id in self.val_split_list:
            return 'val'
        print("sequence id {} corresponding to no split".format(seq_id))
        raise NotImplementedError
    
    def _collect_basic_infos(self):
        self.train_info = defaultdict(dict)
        if self.train_split_list is not None:
            for train_seq in self.train_split_list:
                anno_file_path = osp.join(self.data_root, train_seq, '{}.json'.format(train_seq))
                if not osp.isfile(anno_file_path):
                    print("no annotation file for sequence {}".format(train_seq))
                    raise FileNotFoundError
                anno_file = json.load(open(anno_file_path, 'r'))
                for frame_anno in anno_file['frames']:
                    self.train_info[train_seq][frame_anno['frame_id']] = {
                        'pose': frame_anno['pose'],
                        'calib': anno_file['calib'],
                    }

    def __len__(self):
        """Length of the filename list."""
        return len(self.sequence_paths)

    def convert(self):
        """Convert action."""
        print('Start converting ...')
        mmcv.track_parallel_progress(self.convert_one, range(len(self)),
                                     self.workers)
        print('\nFinished ...')

    def convert_one(self, seqence_path: str):
        """Convert action for single file.

        Args:
            sequence_path (str): path to the sequence file.
        """
        frame_paths = sorted(glob(join(seqence_path, 'lidar_roof')))
        frame_ids = [frame_path.split('/')[-1].split('.')[0] for frame_path in frame_paths]
        self.save_image(seqence_path, frame_ids)
        self.save_calib(seqence_path, frame_ids)
        self.save_lidar(seqence_path, frame_ids)

    def save_image(self, sequence_path: str, frame_ids: List[str]):
        """Parse and save the images in jpg format. Jpg is the original format
        used by Once dataset.

        Args:
            sequence_path (str): path to the sequence file.
            frame_idss (list[str]): ids of the frame files in the current sequence.
        """
        sequence_id = sequence_path.split('/')[-1]
        for camera in self.camera_list:
            for frame_id in frame_ids:
                src_path = f'{sequence_path}/{camera}/{frame_id}.jpg'
                dist_path = f'{self.image_save_dir}{sequence_id}' + \
                            f'{frame_id}.jpg'
                shutil.copyfile(src_path, dist_path)

    def save_calib(self, sequence_path: str):
        """Parse and save the calibration data.

        Args:
            sequence_path (str): path to the sequence file.
        """
        sequence_id = sequence_path.split('/')[-1]
        # once camera 03 to kitti reference camera P0
        T_cam03_to_ref = np.eye(3)

        camera_calibs = []
        R0_rect = [f'{i:e}' for i in np.eye(3).flatten]
        Tr_velo_to_cams = []
        calib_context = ''

        original_calibs = \
            json.load(open(f'{sequence_path}/{sequence_path.split[-1]}.json'))['calib']

        # TODO: need to check
        for camera in self.camera_list:
            # extrinsic parameters
            Tr_cam_to_velo = original_calibs[camera]['cam_to_velo'].reshape(4, 4)
            Tr_velo_to_cam = np.linalg.inv(Tr_cam_to_velo)
            Tr_velo_to_cam = self.cart_to_homo(T_cam03_to_ref) @ Tr_velo_to_cam
            if camera == 'cam03':
                self.Tr_velo_to_cam = Tr_velo_to_cam.copy()
            Tr_velo_to_cams.append(Tr_velo_to_cam)

            # intrinsic parameters
            camera_calib = original_calibs[camera]['cam_intrinsic'].reshape(3, 3)
            camera_calib = np.hstack([camera_calib, np.zeros((3,1), dtype=np.float32)])
            camera_calib = list(camera_calib.reshape(12))
            camera_calib = [f'{i:e}' for i in camera_calib]
            camera_calibs.append(camera_calib)

        # save files
        for i in range(7):
            calib_context += 'P' + str(i) + ': ' + \
                ' '.join(camera_calibs[i]) + '\n'
        calib_context += 'R0_rect' + ': ' + ' '.join(R0_rect)
        for i in range(7):
            calib_context += 'Tr_velo_to_cam_' + str(i) + ': ' + \
                ' '.join(Tr_velo_to_cams[i]) + '\n'

        with open(
                f'{self.calib_save_dir}/' +
                f'{str(sequence_id)}.txt',
                'w+') as fp_calib:
            fp_calib.write(calib_context)
            fp_calib.close()

    def save_lidar(self, sequence_path: str, frame_ids: List[str]):
        """Parse and save the lidar data in psd format.

        Args:
            sequence_path (str): path to the sequence file.
            frame_idss (list[str]): ids of the frame files in the current sequence.
        """
        sequence_id = sequence_path.split('/')[-1]
        for frame_id in frame_ids:
            src_path = f'{sequence_path}/lidar_roof/{frame_id}.bin'
            dist_path = f'{self.point_cloud_save_dir}{self.prefix}' + \
                        f'{sequence_id}{frame_id}.bin'
            shutil.copyfile(src_path, dist_path)

    def save_label(self, sequence_path: str, frame_ids: List[str]):
        """Parse and save the label data in txt format, originally in json format
        The relation between once and kitti coordinates:
        1. x, y, z correspond to l, w, h (once) -> l, h, w (kitti)
        2. cx, cy, cz on the lidar coordinate (once) -> on the camera coordinate (kitti)
        3. x-y-z: right-down-front (once) = right-down-front (kitti)
        4. bbox origin at volumetric center (once) -> bottom center (kitti)
        5. rotation: +x around z-axis yaw angle (once) -> +x around y-axis roll angle(kitti)

        Args:
            sequence_path (str): path to the sequence file.   
            frame_idss (list[str]): ids of the frame files in the current sequence.
        """
        sequence_id = sequence_path.split['/'][-1]
        for frame_id in frame_ids:
            original_annos = self.get_frame_anno(sequence_id, frame_id)
            # TODO: frames that have no annos
            if not original_annos:
                return
            names = original_annos['names']
            boxes_2d = original_annos['boxes_2d']
            boxes_3d = original_annos['boxes_3d']

            line = ''
            for idx, once_type in enumerate(names):
                kitti_type = self.once_to_kitti_class_map[once_type]
                box_2d = boxes_2d[idx]
                box_3d = boxes_3d[idx]

                cx, cy, cz = box_3d[:3]
                height, width, length = box_3d[-2:2:-1]
                x, y, z = cx, cy, cz - height / 2

                # project bounding box to the reference image frame
                pt_ref = self.Tr_velo_to_cam @ \
                    np.array([x, y, z, 1]).reshape((4, 1))
                x, y, z, _ = pt_ref.flatten().tolist()
                # TODO: check if needed to project boxes 2d
                # the boxes 2d of kitti are on reference image frame
                # the boxes 2d of once are on each individual camera frame

                # TODO: need to check
                rotation_y = -box_3d[6] - np.pi / 2
                
                # not available for once
                truncated = 0
                occluded = 0
                alpha = -10

                line += kitti_type + \
                    ' {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(
                        round(truncated, 2), occluded, round(alpha, 2),
                        round(box_2d[0], 2), round(box_2d[1], 2),
                        round(box_2d[2], 2), round(box_2d[3], 2),
                        round(height, 2), round(width, 2), round(length, 2),
                        round(x, 2), round(y, 2), round(z, 2),
                        round(rotation_y, 2))
                
            fp_label = open(
                f'{self.label_save_dir}/' +
                f'{sequence_id}{frame_id}.txt', 'a')
            fp_label.write(line)
            fp_label.close()

    def get_frame_anno(self, sequence_id, frame_id):
        split_name = self._find_split_name(sequence_id)
        frame_info = getattr(self, '{}_info'.format(split_name))[sequence_id][frame_id]
        if 'annos' in frame_info:
            return frame_info['annos']
        return None

    def create_folder(self):
        """Create folder for data preprossing"""
        if not self.test_mode:
            dir_list1 = [
                self.label_all_save_dir, self.calib_save_dir,
                self.point_cloud_save_dir, self.pose_save_dir,
                self.timestamp_save_dir
            ]
            dir_list2 = [self.label_save_dir, self.image_save_dir]
        else:
            dir_list1 = [
                self.calib_save_dir, self.point_cloud_save_dir,
                self.pose_save_dir, self.timestamp_save_dir
            ]
            dir_list2 = [self.image_save_dir]
        for d in dir_list1:
            mmcv.mkdir_or_exist(d)
        for d in dir_list2:
            for i in range(5):
                mmcv.mkdir_or_exist(f'{d}{str(i)}')

    def cart_to_homo(self, mat):
        """Convert transformation matrix in Cartesian coordinates to
        homogeneous format.

        Args:
            mat (np.ndarray): Transformation matrix in Cartesian.
                The input matrix shape is 3x3 or 3x4.

        Returns:
            np.ndarray: Transformation matrix in homogeneous format.
                The matrix shape is 4x4.
        """
        ret = np.eye(4)
        if mat.shape == (3, 3):
            ret[:3, :3] = mat
        elif mat.shape == (3, 4):
            ret[:3, :] = mat
        else:
            raise ValueError(mat.shape)
        return ret
