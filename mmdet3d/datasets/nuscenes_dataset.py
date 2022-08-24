# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp
from typing import Dict, List

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.structures.bbox_3d.cam_box3d import CameraInstance3DBoxes
from .det3d_dataset import Det3DDataset


@DATASETS.register_module()
class NuScenesDataset(Det3DDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        box_type_3d (str): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to dict(use_camera=False,use_lidar=True).
        filter_empty_gt (bool): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
        with_velocity (bool): Whether include velocity prediction
            into the experiments. Defaults to True.
        use_valid_flag (bool): Whether to use `use_valid_flag` key
            in the info file as mask to filter gt_boxes and gt_names.
            Defaults to False.
    """
    METAINFO = {
        'CLASSES':
        ('car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'),
        'version':
        'v1.0-trainval'
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 task: str = '3d',
                 pipeline: List[dict] = None,
                 box_type_3d: str = 'LiDAR',
                 modality: Dict = dict(
                     use_camera=False,
                     use_lidar=True,
                 ),
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 with_velocity: bool = True,
                 use_valid_flag: bool = False,
                 **kwargs):
        self.use_valid_flag = use_valid_flag
        self.with_velocity = with_velocity

        # TODO: Redesign multi-view data process in the future
        assert task in ('3d', 'mono3d', 'multi-view')
        self.task = task

        assert box_type_3d.lower() in ('lidar', 'camera')
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            modality=modality,
            pipeline=pipeline,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)

    def parse_ann_info(self, info: dict) -> dict:
        """Get annotation info according to the given index.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
        """
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            # empty instance
            anns_results = dict()
            anns_results['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            anns_results['gt_labels_3d'] = np.zeros(0, dtype=np.int64)
            return anns_results

        if self.use_valid_flag:
            mask = ann_info['bbox_3d_isvalid']
        else:
            mask = ann_info['num_lidar_pts'] > 0
        gt_bboxes_3d = ann_info['gt_bboxes_3d'][mask]
        gt_labels_3d = ann_info['gt_labels_3d'][mask]

        if 'gt_bboxes' in ann_info:
            gt_bboxes = ann_info['gt_bboxes'][mask]
            gt_labels = ann_info['gt_labels'][mask]
            attr_labels = ann_info['attr_labels'][mask]
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            attr_labels = np.array([], dtype=np.int64)

        if 'centers_2d' in ann_info:
            centers_2d = ann_info['centers_2d'][mask]
            depths = ann_info['depths'][mask]
        else:
            centers_2d = np.zeros((0, 2), dtype=np.float32)
            depths = np.zeros((0), dtype=np.float32)

        if self.with_velocity:
            gt_velocity = ann_info['velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # TODO: Unify the coordinates
        if self.task == 'mono3d':
            gt_bboxes_3d = CameraInstance3DBoxes(
                gt_bboxes_3d,
                box_dim=gt_bboxes_3d.shape[-1],
                origin=(0.5, 0.5, 0.5))
        else:
            gt_bboxes_3d = LiDARInstance3DBoxes(
                gt_bboxes_3d,
                box_dim=gt_bboxes_3d.shape[-1],
                origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            attr_labels=attr_labels,
            centers_2d=centers_2d,
            depths=depths)

        return anns_results

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        The only difference with it in `Det3DDataset`
        is the specific process for `plane`.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        if self.task == 'mono3d':
            data_list = []
            if self.modality['use_lidar']:
                info['lidar_points']['lidar_path'] = \
                    osp.join(
                        self.data_prefix.get('pts', ''),
                        info['lidar_points']['lidar_path'])

            if self.modality['use_camera']:
                for cam_id, img_info in info['images'].items():
                    if 'img_path' in img_info:
                        if cam_id in self.data_prefix:
                            cam_prefix = self.data_prefix[cam_id]
                        else:
                            cam_prefix = self.data_prefix.get('img', '')
                        img_info['img_path'] = osp.join(
                            cam_prefix, img_info['img_path'])

            for idx, (cam_id, img_info) in enumerate(info['images'].items()):
                camera_info = dict()
                camera_info['images'] = dict()
                camera_info['images'][cam_id] = img_info
                if 'cam_instances' in info and cam_id in info['cam_instances']:
                    camera_info['instances'] = info['cam_instances'][cam_id]
                else:
                    camera_info['instances'] = []
                # TODO: check whether to change sample_idx for 6 cameras
                #  in one frame
                camera_info['sample_idx'] = info['sample_idx'] * 6 + idx
                camera_info['token'] = info['token']
                camera_info['ego2global'] = info['ego2global']

                if not self.test_mode:
                    # used in traing
                    camera_info['ann_info'] = self.parse_ann_info(camera_info)
                if self.test_mode and self.load_eval_anns:
                    camera_info['eval_ann_info'] = \
                        self.parse_ann_info(camera_info)
                data_list.append(camera_info)
            return data_list
        else:
            data_info = super().parse_data_info(info)
            return data_info
