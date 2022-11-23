# Copyright (c) OpenMMLab. All rights reserved.
"""Convert the annotation pkl to the standard format in OpenMMLab V2.0.

Example:
    python tools/dataset_converters/update_infos_to_v2.py
        --dataset kitti
        --pkl-path ./data/kitti/kitti_infos_train.pkl
        --out-dir ./kitti_v2/
"""

import argparse
import copy
import time
from os import path as osp
from pathlib import Path

import mmengine
import numpy as np
from nuscenes.nuscenes import NuScenes

from mmdet3d.datasets.convert_utils import (convert_annos,
                                            get_kitti_style_2d_boxes,
                                            get_nuscenes_2d_boxes)
from mmdet3d.datasets.utils import convert_quaternion_to_matrix
from mmdet3d.structures import points_cam2img


def get_empty_instance():
    """Empty annotation for single instance."""
    instance = dict(
        # (list[float], required): list of 4 numbers representing
        # the bounding box of the instance, in (x1, y1, x2, y2) order.
        bbox=None,
        # (int, required): an integer in the range
        # [0, num_categories-1] representing the category label.
        bbox_label=None,
        #  (list[float], optional): list of 7 (or 9) numbers representing
        #  the 3D bounding box of the instance,
        #  in [x, y, z, w, h, l, yaw]
        #  (or [x, y, z, w, h, l, yaw, vx, vy]) order.
        bbox_3d=None,
        # (bool, optional): Whether to use the
        # 3D bounding box during training.
        bbox_3d_isvalid=None,
        # (int, optional): 3D category label
        # (typically the same as label).
        bbox_label_3d=None,
        # (float, optional): Projected center depth of the
        # 3D bounding box compared to the image plane.
        depth=None,
        #  (list[float], optional): Projected
        #  2D center of the 3D bounding box.
        center_2d=None,
        # (int, optional): Attribute labels
        # (fine-grained labels such as stopping, moving, ignore, crowd).
        attr_label=None,
        # (int, optional): The number of LiDAR
        # points in the 3D bounding box.
        num_lidar_pts=None,
        # (int, optional): The number of Radar
        # points in the 3D bounding box.
        num_radar_pts=None,
        # (int, optional): Difficulty level of
        # detecting the 3D bounding box.
        difficulty=None,
        unaligned_bbox_3d=None)
    return instance


def get_empty_multicamera_instances(camera_types):

    cam_instance = dict()
    for cam_type in camera_types:
        cam_instance[cam_type] = None
    return cam_instance


def get_empty_lidar_points():
    lidar_points = dict(
        # (int, optional) : Number of features for each point.
        num_pts_feats=None,
        # (str, optional): Path of LiDAR data file.
        lidar_path=None,
        # (list[list[float]], optional): Transformation matrix
        # from lidar to ego-vehicle
        # with shape [4, 4].
        # (Referenced camera coordinate system is ego in KITTI.)
        lidar2ego=None,
    )
    return lidar_points


def get_empty_radar_points():
    radar_points = dict(
        # (int, optional) : Number of features for each point.
        num_pts_feats=None,
        # (str, optional): Path of RADAR data file.
        radar_path=None,
        # Transformation matrix from lidar to
        # ego-vehicle with shape [4, 4].
        # (Referenced camera coordinate system is ego in KITTI.)
        radar2ego=None,
    )
    return radar_points


def get_empty_img_info():
    img_info = dict(
        # (str, required): the path to the image file.
        img_path=None,
        # (int) The height of the image.
        height=None,
        # (int) The width of the image.
        width=None,
        # (str, optional): Path of the depth map file
        depth_map=None,
        # (list[list[float]], optional) : Transformation
        # matrix from camera to image with
        # shape [3, 3], [3, 4] or [4, 4].
        cam2img=None,
        # (list[list[float]]): Transformation matrix from lidar
        # or depth to image with shape [4, 4].
        lidar2img=None,
        # (list[list[float]], optional) : Transformation
        # matrix from camera to ego-vehicle
        # with shape [4, 4].
        cam2ego=None)
    return img_info


def get_single_image_sweep(camera_types):
    single_image_sweep = dict(
        # (float, optional) : Timestamp of the current frame.
        timestamp=None,
        # (list[list[float]], optional) : Transformation matrix
        # from ego-vehicle to the global
        ego2global=None)
    # (dict): Information of images captured by multiple cameras
    images = dict()
    for cam_type in camera_types:
        images[cam_type] = get_empty_img_info()
    single_image_sweep['images'] = images
    return single_image_sweep


def get_single_lidar_sweep():
    single_lidar_sweep = dict(
        # (float, optional) : Timestamp of the current frame.
        timestamp=None,
        # (list[list[float]], optional) : Transformation matrix
        # from ego-vehicle to the global
        ego2global=None,
        # (dict): Information of images captured by multiple cameras
        lidar_points=get_empty_lidar_points())
    return single_lidar_sweep


def get_empty_standard_data_info(
        camera_types=['CAM0', 'CAM1', 'CAM2', 'CAM3', 'CAM4']):

    data_info = dict(
        # (str): Sample id of the frame.
        sample_idx=None,
        # (str, optional): '000010'
        token=None,
        **get_single_image_sweep(camera_types),
        # (dict, optional): dict contains information
        # of LiDAR point cloud frame.
        lidar_points=get_empty_lidar_points(),
        # (dict, optional) Each dict contains
        # information of Radar point cloud frame.
        radar_points=get_empty_radar_points(),
        # (list[dict], optional): Image sweeps data.
        image_sweeps=[],
        lidar_sweeps=[],
        instances=[],
        # (list[dict], optional): Required by object
        # detection, instance  to be ignored during training.
        instances_ignore=[],
        # (str, optional): Path of semantic labels for each point.
        pts_semantic_mask_path=None,
        # (str, optional): Path of instance labels for each point.
        pts_instance_mask_path=None)
    return data_info


def clear_instance_unused_keys(instance):
    keys = list(instance.keys())
    for k in keys:
        if instance[k] is None:
            del instance[k]
    return instance


def clear_data_info_unused_keys(data_info):
    keys = list(data_info.keys())
    empty_flag = True
    for key in keys:
        # we allow no annotations in datainfo
        if key in ['instances', 'cam_sync_instances', 'cam_instances']:
            empty_flag = False
            continue
        if isinstance(data_info[key], list):
            if len(data_info[key]) == 0:
                del data_info[key]
            else:
                empty_flag = False
        elif data_info[key] is None:
            del data_info[key]
        elif isinstance(data_info[key], dict):
            _, sub_empty_flag = clear_data_info_unused_keys(data_info[key])
            if sub_empty_flag is False:
                empty_flag = False
            else:
                # sub field is empty
                del data_info[key]
        else:
            empty_flag = False

    return data_info, empty_flag


def generate_nuscenes_camera_instances(info, nusc):

    # get bbox annotations for camera
    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]

    empty_multicamera_instance = get_empty_multicamera_instances(camera_types)

    for cam in camera_types:
        cam_info = info['cams'][cam]
        # list[dict]
        ann_infos = get_nuscenes_2d_boxes(
            nusc,
            cam_info['sample_data_token'],
            visibilities=['', '1', '2', '3', '4'])
        empty_multicamera_instance[cam] = ann_infos

    return empty_multicamera_instance


def update_nuscenes_infos(pkl_path, out_dir):
    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]
    print(f'{pkl_path} will be modified.')
    if out_dir in pkl_path:
        print(f'Warning, you may overwriting '
              f'the original data {pkl_path}.')
    print(f'Reading from input file: {pkl_path}.')
    data_list = mmengine.load(pkl_path)
    METAINFO = {
        'classes':
        ('car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'),
    }
    nusc = NuScenes(
        version=data_list['metadata']['version'],
        dataroot='./data/nuscenes',
        verbose=True)

    print('Start updating:')
    converted_list = []
    for i, ori_info_dict in enumerate(
            mmengine.track_iter_progress(data_list['infos'])):
        temp_data_info = get_empty_standard_data_info(
            camera_types=camera_types)
        temp_data_info['sample_idx'] = i
        temp_data_info['token'] = ori_info_dict['token']
        temp_data_info['ego2global'] = convert_quaternion_to_matrix(
            ori_info_dict['ego2global_rotation'],
            ori_info_dict['ego2global_translation'])
        temp_data_info['lidar_points']['num_pts_feats'] = ori_info_dict.get(
            'num_features', 5)
        temp_data_info['lidar_points']['lidar_path'] = Path(
            ori_info_dict['lidar_path']).name
        temp_data_info['lidar_points'][
            'lidar2ego'] = convert_quaternion_to_matrix(
                ori_info_dict['lidar2ego_rotation'],
                ori_info_dict['lidar2ego_translation'])
        # bc-breaking: Timestamp has divided 1e6 in pkl infos.
        temp_data_info['timestamp'] = ori_info_dict['timestamp'] / 1e6
        for ori_sweep in ori_info_dict['sweeps']:
            temp_lidar_sweep = get_single_lidar_sweep()
            temp_lidar_sweep['lidar_points'][
                'lidar2ego'] = convert_quaternion_to_matrix(
                    ori_sweep['sensor2ego_rotation'],
                    ori_sweep['sensor2ego_translation'])
            temp_lidar_sweep['ego2global'] = convert_quaternion_to_matrix(
                ori_sweep['ego2global_rotation'],
                ori_sweep['ego2global_translation'])
            lidar2sensor = np.eye(4)
            lidar2sensor[:3, :3] = ori_sweep['sensor2lidar_rotation'].T
            lidar2sensor[:3, 3] = -ori_sweep['sensor2lidar_translation']
            temp_lidar_sweep['lidar_points'][
                'lidar2sensor'] = lidar2sensor.astype(np.float32).tolist()
            temp_lidar_sweep['timestamp'] = ori_sweep['timestamp'] / 1e6
            temp_lidar_sweep['lidar_points']['lidar_path'] = ori_sweep[
                'data_path']
            temp_lidar_sweep['sample_data_token'] = ori_sweep[
                'sample_data_token']
            temp_data_info['lidar_sweeps'].append(temp_lidar_sweep)
        temp_data_info['images'] = {}
        for cam in ori_info_dict['cams']:
            empty_img_info = get_empty_img_info()
            empty_img_info['img_path'] = Path(
                ori_info_dict['cams'][cam]['data_path']).name
            empty_img_info['cam2img'] = ori_info_dict['cams'][cam][
                'cam_intrinsic'].tolist()
            empty_img_info['sample_data_token'] = ori_info_dict['cams'][cam][
                'sample_data_token']
            # bc-breaking: Timestamp has divided 1e6 in pkl infos.
            empty_img_info[
                'timestamp'] = ori_info_dict['cams'][cam]['timestamp'] / 1e6
            empty_img_info['cam2ego'] = convert_quaternion_to_matrix(
                ori_info_dict['cams'][cam]['sensor2ego_rotation'],
                ori_info_dict['cams'][cam]['sensor2ego_translation'])
            lidar2sensor = np.eye(4)
            lidar2sensor[:3, :3] = ori_info_dict['cams'][cam][
                'sensor2lidar_rotation'].T
            lidar2sensor[:3, 3] = -ori_info_dict['cams'][cam][
                'sensor2lidar_translation']
            empty_img_info['lidar2cam'] = lidar2sensor.astype(
                np.float32).tolist()
            temp_data_info['images'][cam] = empty_img_info
        num_instances = ori_info_dict['gt_boxes'].shape[0]
        ignore_class_name = set()
        for i in range(num_instances):
            empty_instance = get_empty_instance()
            empty_instance['bbox_3d'] = ori_info_dict['gt_boxes'][
                i, :].tolist()
            if ori_info_dict['gt_names'][i] in METAINFO['classes']:
                empty_instance['bbox_label'] = METAINFO['classes'].index(
                    ori_info_dict['gt_names'][i])
            else:
                ignore_class_name.add(ori_info_dict['gt_names'][i])
                empty_instance['bbox_label'] = -1
            empty_instance['bbox_label_3d'] = copy.deepcopy(
                empty_instance['bbox_label'])
            empty_instance['velocity'] = ori_info_dict['gt_velocity'][
                i, :].tolist()
            empty_instance['num_lidar_pts'] = ori_info_dict['num_lidar_pts'][i]
            empty_instance['num_radar_pts'] = ori_info_dict['num_radar_pts'][i]
            empty_instance['bbox_3d_isvalid'] = ori_info_dict['valid_flag'][i]
            empty_instance = clear_instance_unused_keys(empty_instance)
            temp_data_info['instances'].append(empty_instance)
        temp_data_info['cam_instances'] = generate_nuscenes_camera_instances(
            ori_info_dict, nusc)
        temp_data_info, _ = clear_data_info_unused_keys(temp_data_info)
        converted_list.append(temp_data_info)
    pkl_name = Path(pkl_path).name
    out_path = osp.join(out_dir, pkl_name)
    print(f'Writing to output file: {out_path}.')
    print(f'ignore classes: {ignore_class_name}')

    metainfo = dict()
    metainfo['categories'] = {k: i for i, k in enumerate(METAINFO['classes'])}
    if ignore_class_name:
        for ignore_class in ignore_class_name:
            metainfo['categories'][ignore_class] = -1
    metainfo['dataset'] = 'nuscenes'
    metainfo['version'] = data_list['metadata']['version']
    metainfo['info_version'] = '1.1'
    converted_data_info = dict(metainfo=metainfo, data_list=converted_list)

    mmengine.dump(converted_data_info, out_path, 'pkl')


def update_kitti_infos(pkl_path, out_dir):
    print(f'{pkl_path} will be modified.')
    if out_dir in pkl_path:
        print(f'Warning, you may overwriting '
              f'the original data {pkl_path}.')
        time.sleep(5)
    # TODO update to full label
    # TODO discuss how to process 'Van', 'DontCare'
    METAINFO = {
        'classes': ('Pedestrian', 'Cyclist', 'Car', 'Van', 'Truck',
                    'Person_sitting', 'Tram', 'Misc'),
    }
    print(f'Reading from input file: {pkl_path}.')
    data_list = mmengine.load(pkl_path)
    print('Start updating:')
    converted_list = []
    for ori_info_dict in mmengine.track_iter_progress(data_list):
        temp_data_info = get_empty_standard_data_info()

        if 'plane' in ori_info_dict:
            temp_data_info['plane'] = ori_info_dict['plane']

        temp_data_info['sample_idx'] = ori_info_dict['image']['image_idx']

        temp_data_info['images']['CAM0']['cam2img'] = ori_info_dict['calib'][
            'P0'].tolist()
        temp_data_info['images']['CAM1']['cam2img'] = ori_info_dict['calib'][
            'P1'].tolist()
        temp_data_info['images']['CAM2']['cam2img'] = ori_info_dict['calib'][
            'P2'].tolist()
        temp_data_info['images']['CAM3']['cam2img'] = ori_info_dict['calib'][
            'P3'].tolist()

        temp_data_info['images']['CAM2']['img_path'] = Path(
            ori_info_dict['image']['image_path']).name
        h, w = ori_info_dict['image']['image_shape']
        temp_data_info['images']['CAM2']['height'] = h
        temp_data_info['images']['CAM2']['width'] = w
        temp_data_info['lidar_points']['num_pts_feats'] = ori_info_dict[
            'point_cloud']['num_features']
        temp_data_info['lidar_points']['lidar_path'] = Path(
            ori_info_dict['point_cloud']['velodyne_path']).name

        rect = ori_info_dict['calib']['R0_rect'].astype(np.float32)
        Trv2c = ori_info_dict['calib']['Tr_velo_to_cam'].astype(np.float32)
        lidar2cam = rect @ Trv2c
        temp_data_info['images']['CAM2']['lidar2cam'] = lidar2cam.tolist()
        temp_data_info['images']['CAM0']['lidar2img'] = (
            ori_info_dict['calib']['P0'] @ lidar2cam).tolist()
        temp_data_info['images']['CAM1']['lidar2img'] = (
            ori_info_dict['calib']['P1'] @ lidar2cam).tolist()
        temp_data_info['images']['CAM2']['lidar2img'] = (
            ori_info_dict['calib']['P2'] @ lidar2cam).tolist()
        temp_data_info['images']['CAM3']['lidar2img'] = (
            ori_info_dict['calib']['P3'] @ lidar2cam).tolist()

        temp_data_info['lidar_points']['Tr_velo_to_cam'] = Trv2c.tolist()

        # for potential usage
        temp_data_info['images']['R0_rect'] = ori_info_dict['calib'][
            'R0_rect'].astype(np.float32).tolist()
        temp_data_info['lidar_points']['Tr_imu_to_velo'] = ori_info_dict[
            'calib']['Tr_imu_to_velo'].astype(np.float32).tolist()

        anns = ori_info_dict['annos']
        num_instances = len(anns['name'])
        cam2img = ori_info_dict['calib']['P2']

        ignore_class_name = set()
        instance_list = []
        for instance_id in range(num_instances):
            empty_instance = get_empty_instance()
            empty_instance['bbox'] = anns['bbox'][instance_id].tolist()

            if anns['name'][instance_id] in METAINFO['classes']:
                empty_instance['bbox_label'] = METAINFO['classes'].index(
                    anns['name'][instance_id])
            else:
                ignore_class_name.add(anns['name'][instance_id])
                empty_instance['bbox_label'] = -1

            empty_instance['bbox'] = anns['bbox'][instance_id].tolist()

            loc = anns['location'][instance_id]
            dims = anns['dimensions'][instance_id]
            rots = anns['rotation_y'][:, None][instance_id]

            dst = np.array([0.5, 0.5, 0.5])
            src = np.array([0.5, 1.0, 0.5])

            center_3d = loc + dims * (dst - src)
            center_2d = points_cam2img(
                center_3d.reshape([1, 3]), cam2img, with_depth=True)
            center_2d = center_2d.squeeze().tolist()
            empty_instance['center_2d'] = center_2d[:2]
            empty_instance['depth'] = center_2d[2]

            gt_bboxes_3d = np.concatenate([loc, dims, rots]).tolist()
            empty_instance['bbox_3d'] = gt_bboxes_3d
            empty_instance['bbox_label_3d'] = copy.deepcopy(
                empty_instance['bbox_label'])
            empty_instance['bbox'] = anns['bbox'][instance_id].tolist()
            empty_instance['truncated'] = anns['truncated'][
                instance_id].tolist()
            empty_instance['occluded'] = anns['occluded'][instance_id].tolist()
            empty_instance['alpha'] = anns['alpha'][instance_id].tolist()
            empty_instance['score'] = anns['score'][instance_id].tolist()
            empty_instance['index'] = anns['index'][instance_id].tolist()
            empty_instance['group_id'] = anns['group_ids'][instance_id].tolist(
            )
            empty_instance['difficulty'] = anns['difficulty'][
                instance_id].tolist()
            empty_instance['num_lidar_pts'] = anns['num_points_in_gt'][
                instance_id].tolist()
            empty_instance = clear_instance_unused_keys(empty_instance)
            instance_list.append(empty_instance)
        temp_data_info['instances'] = instance_list
        cam_instances = generate_kitti_camera_instances(ori_info_dict)
        temp_data_info['cam_instances'] = cam_instances
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
    metainfo['dataset'] = 'kitti'
    metainfo['info_version'] = '1.1'
    converted_data_info = dict(metainfo=metainfo, data_list=converted_list)

    mmengine.dump(converted_data_info, out_path, 'pkl')


def update_s3dis_infos(pkl_path, out_dir):
    print(f'{pkl_path} will be modified.')
    if out_dir in pkl_path:
        print(f'Warning, you may overwriting '
              f'the original data {pkl_path}.')
        time.sleep(5)
    METAINFO = {'classes': ('table', 'chair', 'sofa', 'bookcase', 'board')}
    print(f'Reading from input file: {pkl_path}.')
    data_list = mmengine.load(pkl_path)
    print('Start updating:')
    converted_list = []
    for i, ori_info_dict in enumerate(mmengine.track_iter_progress(data_list)):
        temp_data_info = get_empty_standard_data_info()
        temp_data_info['sample_idx'] = i
        temp_data_info['lidar_points']['num_pts_feats'] = ori_info_dict[
            'point_cloud']['num_features']
        temp_data_info['lidar_points']['lidar_path'] = Path(
            ori_info_dict['pts_path']).name
        temp_data_info['pts_semantic_mask_path'] = Path(
            ori_info_dict['pts_semantic_mask_path']).name
        temp_data_info['pts_instance_mask_path'] = Path(
            ori_info_dict['pts_instance_mask_path']).name

        # TODO support camera
        # np.linalg.inv(info['axis_align_matrix'] @ extrinsic): depth2cam
        anns = ori_info_dict.get('annos', None)
        ignore_class_name = set()
        if anns is not None:
            if anns['gt_num'] == 0:
                instance_list = []
            else:
                num_instances = len(anns['class'])
                instance_list = []
                for instance_id in range(num_instances):
                    empty_instance = get_empty_instance()
                    empty_instance['bbox_3d'] = anns['gt_boxes_upright_depth'][
                        instance_id].tolist()

                    if anns['class'][instance_id] < len(METAINFO['classes']):
                        empty_instance['bbox_label_3d'] = anns['class'][
                            instance_id]
                    else:
                        ignore_class_name.add(
                            METAINFO['classes'][anns['class'][instance_id]])
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
    metainfo['dataset'] = 's3dis'
    metainfo['info_version'] = '1.1'

    converted_data_info = dict(metainfo=metainfo, data_list=converted_list)

    mmengine.dump(converted_data_info, out_path, 'pkl')


def update_scannet_infos(pkl_path, out_dir):
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
        temp_data_info['lidar_points']['num_pts_feats'] = ori_info_dict[
            'point_cloud']['num_features']
        temp_data_info['lidar_points']['lidar_path'] = Path(
            ori_info_dict['pts_path']).name
        temp_data_info['pts_semantic_mask_path'] = Path(
            ori_info_dict['pts_semantic_mask_path']).name
        temp_data_info['pts_instance_mask_path'] = Path(
            ori_info_dict['pts_instance_mask_path']).name

        # TODO support camera
        # np.linalg.inv(info['axis_align_matrix'] @ extrinsic): depth2cam
        anns = ori_info_dict['annos']
        temp_data_info['axis_align_matrix'] = anns['axis_align_matrix'].tolist(
        )
        if anns['gt_num'] == 0:
            instance_list = []
        else:
            num_instances = len(anns['name'])
            ignore_class_name = set()
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


def update_sunrgbd_infos(pkl_path, out_dir):
    print(f'{pkl_path} will be modified.')
    if out_dir in pkl_path:
        print(f'Warning, you may overwriting '
              f'the original data {pkl_path}.')
        time.sleep(5)
    METAINFO = {
        'classes': ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk',
                    'dresser', 'night_stand', 'bookshelf', 'bathtub')
    }
    print(f'Reading from input file: {pkl_path}.')
    data_list = mmengine.load(pkl_path)
    print('Start updating:')
    converted_list = []
    for ori_info_dict in mmengine.track_iter_progress(data_list):
        temp_data_info = get_empty_standard_data_info()
        temp_data_info['lidar_points']['num_pts_feats'] = ori_info_dict[
            'point_cloud']['num_features']
        temp_data_info['lidar_points']['lidar_path'] = Path(
            ori_info_dict['pts_path']).name
        calib = ori_info_dict['calib']
        rt_mat = calib['Rt']
        # follow Coord3DMode.convert_point
        rt_mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]
                           ]) @ rt_mat.transpose(1, 0)
        depth2img = calib['K'] @ rt_mat
        temp_data_info['images']['CAM0']['depth2img'] = depth2img.tolist()
        temp_data_info['images']['CAM0']['img_path'] = Path(
            ori_info_dict['image']['image_path']).name
        h, w = ori_info_dict['image']['image_shape']
        temp_data_info['images']['CAM0']['height'] = h
        temp_data_info['images']['CAM0']['width'] = w

        anns = ori_info_dict['annos']
        if anns['gt_num'] == 0:
            instance_list = []
        else:
            num_instances = len(anns['name'])
            ignore_class_name = set()
            instance_list = []
            for instance_id in range(num_instances):
                empty_instance = get_empty_instance()
                empty_instance['bbox_3d'] = anns['gt_boxes_upright_depth'][
                    instance_id].tolist()
                empty_instance['bbox'] = anns['bbox'][instance_id].tolist()
                if anns['name'][instance_id] in METAINFO['classes']:
                    empty_instance['bbox_label_3d'] = METAINFO[
                        'classes'].index(anns['name'][instance_id])
                    empty_instance['bbox_label'] = empty_instance[
                        'bbox_label_3d']
                else:
                    ignore_class_name.add(anns['name'][instance_id])
                    empty_instance['bbox_label_3d'] = -1
                    empty_instance['bbox_label'] = -1
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
    metainfo['dataset'] = 'sunrgbd'
    metainfo['info_version'] = '1.1'

    converted_data_info = dict(metainfo=metainfo, data_list=converted_list)

    mmengine.dump(converted_data_info, out_path, 'pkl')


def update_lyft_infos(pkl_path, out_dir):
    print(f'{pkl_path} will be modified.')
    if out_dir in pkl_path:
        print(f'Warning, you may overwriting '
              f'the original data {pkl_path}.')
    print(f'Reading from input file: {pkl_path}.')
    data_list = mmengine.load(pkl_path)
    METAINFO = {
        'classes':
        ('car', 'truck', 'bus', 'emergency_vehicle', 'other_vehicle',
         'motorcycle', 'bicycle', 'pedestrian', 'animal'),
    }
    print('Start updating:')
    converted_list = []
    for i, ori_info_dict in enumerate(
            mmengine.track_iter_progress(data_list['infos'])):
        temp_data_info = get_empty_standard_data_info()
        temp_data_info['sample_idx'] = i
        temp_data_info['token'] = ori_info_dict['token']
        temp_data_info['ego2global'] = convert_quaternion_to_matrix(
            ori_info_dict['ego2global_rotation'],
            ori_info_dict['ego2global_translation'])
        temp_data_info['lidar_points']['num_pts_feats'] = ori_info_dict.get(
            'num_features', 5)
        temp_data_info['lidar_points']['lidar_path'] = Path(
            ori_info_dict['lidar_path']).name
        temp_data_info['lidar_points'][
            'lidar2ego'] = convert_quaternion_to_matrix(
                ori_info_dict['lidar2ego_rotation'],
                ori_info_dict['lidar2ego_translation'])
        # bc-breaking: Timestamp has divided 1e6 in pkl infos.
        temp_data_info['timestamp'] = ori_info_dict['timestamp'] / 1e6
        for ori_sweep in ori_info_dict['sweeps']:
            temp_lidar_sweep = get_single_lidar_sweep()
            temp_lidar_sweep['lidar_points'][
                'lidar2ego'] = convert_quaternion_to_matrix(
                    ori_sweep['sensor2ego_rotation'],
                    ori_sweep['sensor2ego_translation'])
            temp_lidar_sweep['ego2global'] = convert_quaternion_to_matrix(
                ori_sweep['ego2global_rotation'],
                ori_sweep['ego2global_translation'])
            lidar2sensor = np.eye(4)
            lidar2sensor[:3, :3] = ori_sweep['sensor2lidar_rotation'].T
            lidar2sensor[:3, 3] = -ori_sweep['sensor2lidar_translation']
            temp_lidar_sweep['lidar_points'][
                'lidar2sensor'] = lidar2sensor.astype(np.float32).tolist()
            # bc-breaking: Timestamp has divided 1e6 in pkl infos.
            temp_lidar_sweep['timestamp'] = ori_sweep['timestamp'] / 1e6
            temp_lidar_sweep['lidar_points']['lidar_path'] = ori_sweep[
                'data_path']
            temp_lidar_sweep['sample_data_token'] = ori_sweep[
                'sample_data_token']
            temp_data_info['lidar_sweeps'].append(temp_lidar_sweep)
        temp_data_info['images'] = {}
        for cam in ori_info_dict['cams']:
            empty_img_info = get_empty_img_info()
            empty_img_info['img_path'] = Path(
                ori_info_dict['cams'][cam]['data_path']).name
            empty_img_info['cam2img'] = ori_info_dict['cams'][cam][
                'cam_intrinsic'].tolist()
            empty_img_info['sample_data_token'] = ori_info_dict['cams'][cam][
                'sample_data_token']
            empty_img_info[
                'timestamp'] = ori_info_dict['cams'][cam]['timestamp'] / 1e6
            empty_img_info['cam2ego'] = convert_quaternion_to_matrix(
                ori_info_dict['cams'][cam]['sensor2ego_rotation'],
                ori_info_dict['cams'][cam]['sensor2ego_translation'])
            lidar2sensor = np.eye(4)
            lidar2sensor[:3, :3] = ori_info_dict['cams'][cam][
                'sensor2lidar_rotation'].T
            lidar2sensor[:3, 3] = -ori_info_dict['cams'][cam][
                'sensor2lidar_translation']
            empty_img_info['lidar2cam'] = lidar2sensor.astype(
                np.float32).tolist()
            temp_data_info['images'][cam] = empty_img_info
        num_instances = ori_info_dict['gt_boxes'].shape[0]
        ignore_class_name = set()
        for i in range(num_instances):
            empty_instance = get_empty_instance()
            empty_instance['bbox_3d'] = ori_info_dict['gt_boxes'][
                i, :].tolist()
            if ori_info_dict['gt_names'][i] in METAINFO['classes']:
                empty_instance['bbox_label'] = METAINFO['classes'].index(
                    ori_info_dict['gt_names'][i])
            else:
                ignore_class_name.add(ori_info_dict['gt_names'][i])
                empty_instance['bbox_label'] = -1
            empty_instance['bbox_label_3d'] = copy.deepcopy(
                empty_instance['bbox_label'])
            empty_instance = clear_instance_unused_keys(empty_instance)
            temp_data_info['instances'].append(empty_instance)
        temp_data_info, _ = clear_data_info_unused_keys(temp_data_info)
        converted_list.append(temp_data_info)
    pkl_name = Path(pkl_path).name
    out_path = osp.join(out_dir, pkl_name)
    print(f'Writing to output file: {out_path}.')
    print(f'ignore classes: {ignore_class_name}')

    metainfo = dict()
    metainfo['categories'] = {k: i for i, k in enumerate(METAINFO['classes'])}
    if ignore_class_name:
        for ignore_class in ignore_class_name:
            metainfo['categories'][ignore_class] = -1
    metainfo['dataset'] = 'lyft'
    metainfo['version'] = data_list['metadata']['version']
    metainfo['info_version'] = '1.1'
    converted_data_info = dict(metainfo=metainfo, data_list=converted_list)

    mmengine.dump(converted_data_info, out_path, 'pkl')


def update_waymo_infos(pkl_path, out_dir):
    # the input pkl is based on the
    # pkl generated in the waymo cam only challenage.
    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_SIDE_RIGHT',
        'CAM_SIDE_LEFT',
    ]
    print(f'{pkl_path} will be modified.')
    if out_dir in pkl_path:
        print(f'Warning, you may overwriting '
              f'the original data {pkl_path}.')
        time.sleep(5)
    # TODO update to full label
    # TODO discuss how to process 'Van', 'DontCare'
    METAINFO = {
        'classes': ('Car', 'Pedestrian', 'Cyclist', 'Sign'),
    }
    print(f'Reading from input file: {pkl_path}.')
    data_list = mmengine.load(pkl_path)
    print('Start updating:')
    converted_list = []
    for ori_info_dict in mmengine.track_iter_progress(data_list):
        temp_data_info = get_empty_standard_data_info(camera_types)

        if 'plane' in ori_info_dict:
            temp_data_info['plane'] = ori_info_dict['plane']
        temp_data_info['sample_idx'] = ori_info_dict['image']['image_idx']

        # calib matrix
        for cam_idx, cam_key in enumerate(camera_types):
            temp_data_info['images'][cam_key]['cam2img'] =\
                 ori_info_dict['calib'][f'P{cam_idx}'].tolist()

        for cam_idx, cam_key in enumerate(camera_types):
            rect = ori_info_dict['calib']['R0_rect'].astype(np.float32)
            velo_to_cam = 'Tr_velo_to_cam'
            if cam_idx != 0:
                velo_to_cam += str(cam_idx)
            Trv2c = ori_info_dict['calib'][velo_to_cam].astype(np.float32)

            lidar2cam = rect @ Trv2c
            temp_data_info['images'][cam_key]['lidar2cam'] = lidar2cam.tolist()
            temp_data_info['images'][cam_key]['lidar2img'] = (
                ori_info_dict['calib'][f'P{cam_idx}'] @ lidar2cam).tolist()

        # image path
        base_img_path = Path(ori_info_dict['image']['image_path']).name

        for cam_idx, cam_key in enumerate(camera_types):
            temp_data_info['images'][cam_key]['timestamp'] = ori_info_dict[
                'timestamp']
            temp_data_info['images'][cam_key]['img_path'] = base_img_path

        h, w = ori_info_dict['image']['image_shape']

        # for potential usage
        temp_data_info['images'][camera_types[0]]['height'] = h
        temp_data_info['images'][camera_types[0]]['width'] = w
        temp_data_info['lidar_points']['num_pts_feats'] = ori_info_dict[
            'point_cloud']['num_features']
        temp_data_info['lidar_points']['timestamp'] = ori_info_dict[
            'timestamp']
        temp_data_info['lidar_points']['lidar_path'] = Path(
            ori_info_dict['point_cloud']['velodyne_path']).name

        # TODO discuss the usage of Tr_velo_to_cam in lidar
        Trv2c = ori_info_dict['calib']['Tr_velo_to_cam'].astype(np.float32)

        temp_data_info['lidar_points']['Tr_velo_to_cam'] = Trv2c.tolist()

        # for potential usage
        # temp_data_info['images']['R0_rect'] = ori_info_dict['calib'][
        #     'R0_rect'].astype(np.float32).tolist()

        # for the sweeps part:
        temp_data_info['timestamp'] = ori_info_dict['timestamp']
        temp_data_info['ego2global'] = ori_info_dict['pose']

        for ori_sweep in ori_info_dict['sweeps']:
            # lidar sweeps
            lidar_sweep = get_single_lidar_sweep()
            lidar_sweep['ego2global'] = ori_sweep['pose']
            lidar_sweep['timestamp'] = ori_sweep['timestamp']
            lidar_sweep['lidar_points']['lidar_path'] = Path(
                ori_sweep['velodyne_path']).name
            # image sweeps
            image_sweep = get_single_image_sweep(camera_types)
            image_sweep['ego2global'] = ori_sweep['pose']
            image_sweep['timestamp'] = ori_sweep['timestamp']
            img_path = Path(ori_sweep['image_path']).name
            for cam_idx, cam_key in enumerate(camera_types):
                image_sweep['images'][cam_key]['img_path'] = img_path

            temp_data_info['lidar_sweeps'].append(lidar_sweep)
            temp_data_info['image_sweeps'].append(image_sweep)

        anns = ori_info_dict['annos']
        num_instances = len(anns['name'])

        ignore_class_name = set()
        instance_list = []
        for instance_id in range(num_instances):
            empty_instance = get_empty_instance()
            empty_instance['bbox'] = anns['bbox'][instance_id].tolist()

            if anns['name'][instance_id] in METAINFO['classes']:
                empty_instance['bbox_label'] = METAINFO['classes'].index(
                    anns['name'][instance_id])
            else:
                ignore_class_name.add(anns['name'][instance_id])
                empty_instance['bbox_label'] = -1

            empty_instance['bbox'] = anns['bbox'][instance_id].tolist()

            loc = anns['location'][instance_id]
            dims = anns['dimensions'][instance_id]
            rots = anns['rotation_y'][:, None][instance_id]
            gt_bboxes_3d = np.concatenate([loc, dims,
                                           rots]).astype(np.float32).tolist()
            empty_instance['bbox_3d'] = gt_bboxes_3d
            empty_instance['bbox_label_3d'] = copy.deepcopy(
                empty_instance['bbox_label'])
            empty_instance['bbox'] = anns['bbox'][instance_id].tolist()
            empty_instance['truncated'] = int(
                anns['truncated'][instance_id].tolist())
            empty_instance['occluded'] = anns['occluded'][instance_id].tolist()
            empty_instance['alpha'] = anns['alpha'][instance_id].tolist()
            empty_instance['index'] = anns['index'][instance_id].tolist()
            empty_instance['group_id'] = anns['group_ids'][instance_id].tolist(
            )
            empty_instance['difficulty'] = anns['difficulty'][
                instance_id].tolist()
            empty_instance['num_lidar_pts'] = anns['num_points_in_gt'][
                instance_id].tolist()
            empty_instance['camera_id'] = anns['camera_id'][
                instance_id].tolist()
            empty_instance = clear_instance_unused_keys(empty_instance)
            instance_list.append(empty_instance)
        temp_data_info['instances'] = instance_list

        # waymo provide the labels that sync with cam
        anns = ori_info_dict['cam_sync_annos']
        num_instances = len(anns['name'])
        ignore_class_name = set()
        instance_list = []
        for instance_id in range(num_instances):
            empty_instance = get_empty_instance()
            empty_instance['bbox'] = anns['bbox'][instance_id].tolist()

            if anns['name'][instance_id] in METAINFO['classes']:
                empty_instance['bbox_label'] = METAINFO['classes'].index(
                    anns['name'][instance_id])
            else:
                ignore_class_name.add(anns['name'][instance_id])
                empty_instance['bbox_label'] = -1

            empty_instance['bbox'] = anns['bbox'][instance_id].tolist()

            loc = anns['location'][instance_id]
            dims = anns['dimensions'][instance_id]
            rots = anns['rotation_y'][:, None][instance_id]
            gt_bboxes_3d = np.concatenate([loc, dims,
                                           rots]).astype(np.float32).tolist()
            empty_instance['bbox_3d'] = gt_bboxes_3d
            empty_instance['bbox_label_3d'] = copy.deepcopy(
                empty_instance['bbox_label'])
            empty_instance['bbox'] = anns['bbox'][instance_id].tolist()
            empty_instance['truncated'] = int(
                anns['truncated'][instance_id].tolist())
            empty_instance['occluded'] = anns['occluded'][instance_id].tolist()
            empty_instance['alpha'] = anns['alpha'][instance_id].tolist()
            empty_instance['index'] = anns['index'][instance_id].tolist()
            empty_instance['group_id'] = anns['group_ids'][instance_id].tolist(
            )
            empty_instance['camera_id'] = anns['camera_id'][
                instance_id].tolist()
            empty_instance = clear_instance_unused_keys(empty_instance)
            instance_list.append(empty_instance)
        temp_data_info['cam_sync_instances'] = instance_list

        cam_instances = generate_waymo_camera_instances(
            ori_info_dict, camera_types)
        temp_data_info['cam_instances'] = cam_instances

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
    metainfo['dataset'] = 'waymo'
    metainfo['version'] = '1.2'
    metainfo['info_version'] = '1.1'

    converted_data_info = dict(metainfo=metainfo, data_list=converted_list)

    mmengine.dump(converted_data_info, out_path, 'pkl')


def generate_kitti_camera_instances(ori_info_dict):

    cam_key = 'CAM2'
    empty_camera_instances = get_empty_multicamera_instances([cam_key])
    annos = copy.deepcopy(ori_info_dict['annos'])
    ann_infos = get_kitti_style_2d_boxes(
        ori_info_dict, occluded=[0, 1, 2, 3], annos=annos)
    empty_camera_instances[cam_key] = ann_infos

    return empty_camera_instances


def generate_waymo_camera_instances(ori_info_dict, cam_keys):

    empty_multicamera_instances = get_empty_multicamera_instances(cam_keys)

    for cam_idx, cam_key in enumerate(cam_keys):
        annos = copy.deepcopy(ori_info_dict['cam_sync_annos'])
        if cam_idx != 0:
            annos = convert_annos(ori_info_dict, cam_idx)

        ann_infos = get_kitti_style_2d_boxes(
            ori_info_dict, cam_idx, occluded=[0], annos=annos, dataset='waymo')

        empty_multicamera_instances[cam_key] = ann_infos
    return empty_multicamera_instances


def parse_args():
    parser = argparse.ArgumentParser(description='Arg parser for data coords '
                                     'update due to coords sys refactor.')
    parser.add_argument(
        '--dataset', type=str, default='kitti', help='name of dataset')
    parser.add_argument(
        '--pkl-path',
        type=str,
        default='./data/kitti/kitti_infos_train.pkl ',
        help='specify the root dir of dataset')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='converted_annotations',
        required=False,
        help='output direction of info pkl')
    args = parser.parse_args()
    return args


def update_pkl_infos(dataset, out_dir, pkl_path):
    if dataset.lower() == 'kitti':
        update_kitti_infos(pkl_path=pkl_path, out_dir=out_dir)
    elif dataset.lower() == 'waymo':
        update_waymo_infos(pkl_path=pkl_path, out_dir=out_dir)
    elif dataset.lower() == 'scannet':
        update_scannet_infos(pkl_path=pkl_path, out_dir=out_dir)
    elif dataset.lower() == 'sunrgbd':
        update_sunrgbd_infos(pkl_path=pkl_path, out_dir=out_dir)
    elif dataset.lower() == 'lyft':
        update_lyft_infos(pkl_path=pkl_path, out_dir=out_dir)
    elif dataset.lower() == 'nuscenes':
        update_nuscenes_infos(pkl_path=pkl_path, out_dir=out_dir)
    elif dataset.lower() == 's3dis':
        update_s3dis_infos(pkl_path=pkl_path, out_dir=out_dir)
    else:
        raise NotImplementedError(f'Do not support convert {dataset} to v2.')


if __name__ == '__main__':
    args = parse_args()
    if args.out_dir is None:
        args.out_dir = args.root_dir
    update_pkl_infos(
        dataset=args.dataset, out_dir=args.out_dir, pkl_path=args.pkl_path)
