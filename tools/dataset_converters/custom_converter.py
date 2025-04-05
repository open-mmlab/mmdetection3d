# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from pathlib import Path

import mmcv
import mmengine
import numpy as np
from nuscenes.utils.geometry_utils import view_points

from mmdet3d.structures import points_cam2img
from mmdet3d.structures.ops import box_np_ops
from .custom_data_utils import get_custom_velodyne_info
from .nuscenes_converter import post_process_coords

custom_categories = ('Car',)


def convert_to_custom_info_version2(info):
    """convert custom info v1 to v2 if possible.

    Args:
        info (dict): Info of the input custom data.
            - image (dict): image info
            - calib (dict): calibration info
            - point_cloud (dict): point cloud info
    """
    if 'point_cloud' not in info:
        info['point_cloud'] = {
            'velodyne_path': info['velodyne_path'],
        }

def _read_velodyne_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [str(line.strip()) for line in lines]

class _NumPointsInGTCalculater:
    """Calculate the number of points inside the ground truth box. This is the
    parallel version. For the serialized version, please refer to
    `_calculate_num_points_in_gt`.

    Args:
        data_path (str): Path of the data.
        relative_path (bool): Whether to use relative path.
        remove_outside (bool, optional): Whether to remove points which are
            outside of image. Default: True.
        num_features (int, optional): Number of features per point.
            Default: False.
        num_worker (int, optional): the number of parallel workers to use.
            Default: 8.
    """

    def __init__(self,
                 data_path,
                 relative_path,
                 num_features=4,
                 num_worker=8) -> None:
        self.data_path = data_path
        self.relative_path = relative_path
        self.num_features = num_features
        self.num_worker = num_worker

    def calculate_single(self, info):
        pc_info = info['point_cloud']
        if self.relative_path:
            v_path = str(Path(self.data_path) / pc_info['velodyne_path'])
        else:
            v_path = pc_info['velodyne_path']
        points_v = np.fromfile(
            v_path, dtype=np.float32,
            count=-1).reshape([-1, self.num_features])


        annos = info['annos']
        # gt_boxes_lidar = annos['gt_boxes_lidar']
        # print("gt_boxes_lidar:{}".format(gt_boxes_lidar))
        num_obj = len([n for n in annos['name']])
        dims = annos['dimensions'][:num_obj]
        loc = annos['location'][:num_obj]
        rots = annos['rotation_y'][:num_obj]
        gt_boxes_lidar = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                         axis=1)
        indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes_lidar)
        num_points_in_gt = indices.sum(0)
        annos['num_points_in_gt'] = num_points_in_gt.astype(np.int32)

        return info

    def calculate(self, infos):
        ret_infos = mmengine.track_parallel_progress(self.calculate_single,
                                                     infos, self.num_worker)
        for i, ret_info in enumerate(ret_infos):
            infos[i] = ret_info

def _calculate_num_points_in_gt(data_path,
                                infos,
                                relative_path,
                                remove_outside=True,
                                num_features=4):
    for info in mmengine.track_iter_progress(infos):
        pc_info = info['point_cloud']

        if relative_path:
            v_path = str(Path(data_path) / pc_info['velodyne_path'])
        else:
            v_path = pc_info['velodyne_path']
        points_v = np.fromfile(
            v_path, dtype=np.float32, count=-1).reshape([-1, num_features])


        # points_v = points_v[points_v[:, 0] > 0]
        annos = info['annos']
        num_obj = len([n for n in annos['name']])
        dims = annos['dimensions'][:num_obj]
        loc = annos['location'][:num_obj]
        rots = annos['rotation_y'][:num_obj]
        gt_boxes_lidar = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                         axis=1)
        indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes_lidar, origin=(0.5, 0.5, 0))
        num_points_in_gt = indices.sum(0)
        # 这里根据gt_boxes_lidar计算得到的num_points_in_gt不对,但是custom_dbinfos_train.pkl中是对的，为什么？
        annos['num_points_in_gt'] = num_points_in_gt.astype(np.int32)

def create_custom_info_file(data_path,
                           pkl_prefix='custom',
                           save_path=None,
                           relative_path=True):
    """Create info file of custom dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        data_path (str): Path of the data root.
        pkl_prefix (str, optional): Prefix of the info file to be generated.
            Default: 'custom'.
        save_path (str, optional): Path to save the info file.
            Default: None.
        relative_path (bool, optional): Whether to use relative path.
            Default: True.
    """
    imageset_folder = Path(data_path) / 'ImageSets'
    train_point_ids = _read_velodyne_file(str(imageset_folder / 'train.txt'))
    val_point_ids = _read_velodyne_file(str(imageset_folder / 'val.txt'))
    test_point_ids = _read_velodyne_file(str(imageset_folder / 'test.txt'))

    print('Generate info. this may take several minutes.')
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)
    custom_infos_train = get_custom_velodyne_info(
        data_path,
        training=True,
        velodyne=True,
        point_ids=train_point_ids,
        relative_path=relative_path)
    _calculate_num_points_in_gt(data_path, custom_infos_train, relative_path)
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'custom info train file is saved to {filename}')
    mmengine.dump(custom_infos_train, filename)
    custom_infos_val = get_custom_velodyne_info(
        data_path,
        training=True,
        velodyne=True,
        point_ids=val_point_ids,
        relative_path=relative_path)
    _calculate_num_points_in_gt(data_path, custom_infos_val, relative_path)
    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'custom info val file is saved to {filename}')
    mmengine.dump(custom_infos_val, filename)
    filename = save_path / f'{pkl_prefix}_infos_trainval.pkl'
    print(f'custom info trainval file is saved to {filename}')
    mmengine.dump(custom_infos_train + custom_infos_val, filename)

    custom_infos_test = get_custom_velodyne_info(
        data_path,
        training=False,
        label_info=False,
        velodyne=True,
        point_ids=test_point_ids,
        relative_path=relative_path)
    filename = save_path / f'{pkl_prefix}_infos_test.pkl'
    print(f'custom info test file is saved to {filename}')
    mmengine.dump(custom_infos_test, filename)

def _create_reduced_point_cloud(data_path,
                                info_path,
                                save_path=None,
                                back=False,
                                num_features=4,
                                front_camera_id=2):
    """Create reduced point clouds for given info.

    Args:
        data_path (str): Path of original data.
        info_path (str): Path of data info.
        save_path (str, optional): Path to save reduced point cloud
            data. Default: None.
        back (bool, optional): Whether to flip the points to back.
            Default: False.
        num_features (int, optional): Number of point features. Default: 4.
        front_camera_id (int, optional): The referenced/front camera ID.
            Default: 2.
    """
    custom_infos = mmengine.load(info_path)

    for info in mmengine.track_iter_progress(custom_infos):
        pc_info = info['point_cloud']
        # image_info = info['image']
        # calib = info['calib']

        v_path = pc_info['velodyne_path']
        v_path = Path(data_path) / v_path
        points_v = np.fromfile(
            str(v_path), dtype=np.float32,
            count=-1).reshape([-1, num_features])

        # first remove z < 0 points
        # keep = points_v[:, -1] > 0
        # points_v = points_v[keep]
        # then remove outside.
        if back:
            points_v[:, 0] = -points_v[:, 0]

        if save_path is None:
            save_dir = v_path.parent.parent / (v_path.parent.stem + '_reduced')
            if not save_dir.exists():
                save_dir.mkdir()
            save_filename = save_dir / v_path.name
            # save_filename = str(v_path) + '_reduced'
            if back:
                save_filename += '_back'
        else:
            save_filename = str(Path(save_path) / v_path.name)
            if back:
                save_filename += '_back'
        with open(save_filename, 'w') as f:
            points_v.tofile(f)

def create_reduced_point_cloud(data_path,
                               pkl_prefix,
                               train_info_path=None,
                               val_info_path=None,
                               test_info_path=None,
                               save_path=None,
                               with_back=False):
    """Create reduced point clouds for training/validation/testing.

    Args:
        data_path (str): Path of original data.
        pkl_prefix (str): Prefix of info files.
        train_info_path (str, optional): Path of training set info.
            Default: None.
        val_info_path (str, optional): Path of validation set info.
            Default: None.
        test_info_path (str, optional): Path of test set info.
            Default: None.
        save_path (str, optional): Path to save reduced point cloud data.
            Default: None.
        with_back (bool, optional): Whether to flip the points to back.
            Default: False.
    """
    if train_info_path is None:
        train_info_path = Path(data_path) / f'{pkl_prefix}_infos_train.pkl'
    if val_info_path is None:
        val_info_path = Path(data_path) / f'{pkl_prefix}_infos_val.pkl'
    if test_info_path is None:
        test_info_path = Path(data_path) / f'{pkl_prefix}_infos_test.pkl'

    print('create reduced point cloud for training set')
    _create_reduced_point_cloud(data_path, train_info_path, save_path)
    print('create reduced point cloud for validation set')
    _create_reduced_point_cloud(data_path, val_info_path, save_path)
    print('create reduced point cloud for testing set')
    _create_reduced_point_cloud(data_path, test_info_path, save_path)
    if with_back:
        _create_reduced_point_cloud(
            data_path, train_info_path, save_path, back=True)
        _create_reduced_point_cloud(
            data_path, val_info_path, save_path, back=True)
        _create_reduced_point_cloud(
            data_path, test_info_path, save_path, back=True)
