# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import numpy as np

from mmdet3d.registry import DATASETS
from ..core.bbox import LiDARInstance3DBoxes
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
        assert box_type_3d.lower() == 'lidar'
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

        if self.with_velocity:
            gt_velocity = ann_info['velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d)
        return anns_results
