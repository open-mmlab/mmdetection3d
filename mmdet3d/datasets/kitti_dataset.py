# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Union

import numpy as np

from mmdet3d.datasets import DATASETS
from mmdet3d.structures import CameraInstance3DBoxes
from .det3d_dataset import Det3DDataset


@DATASETS.register_module()
class KittiDataset(Det3DDataset):
    r"""KITTI Dataset.

    This class serves as the API for experiments on the `KITTI Dataset
    <http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d>`_.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict]): Pipeline used for data processing.
            Defaults to [].
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to `dict(use_lidar=True)`.
        default_cam_key (str): The default camera name adopted.
            Defaults to 'CAM2'.
        box_type_3d (str): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes:

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
        filter_cfg (dict, optional): Config for filter data.
            the filter keys of kitti dataset include:
                - filter_dontcare: bool, whether to filter the 'DontCare'
                    objects in kitti dataset.
                - filter_class: bool, whether to filter not required classes
                    (classes not in metainfo).
                - filter_empty_gt: bool, whether to filter the data sample
                    without annotations.
            Defaults to dict(filter_class=False, filter_empty_gt=False
                             filter_dontcare=True).
    """
    # TODO: use full classes of kitti
    METAINFO = {
        'classes': ('Pedestrian', 'Cyclist', 'Car', 'Van', 'Truck',
                    'Person_sitting', 'Tram', 'Misc')
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True),
                 default_cam_key: str = 'CAM2',
                 task: str = 'lidar_det',
                 box_type_3d: str = 'LiDAR',
                 test_mode: bool = False,
                 filter_cfg: dict = dict(
                     filter_class=False,
                     filter_empty_gt=False,
                     filter_dontcare=True),
                 **kwargs) -> None:

        assert task in ('lidar_det', 'mono_det')
        self.task = task
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            modality=modality,
            default_cam_key=default_cam_key,
            box_type_3d=box_type_3d,
            test_mode=test_mode,
            filter_cfg=filter_cfg,
            **kwargs)

        assert self.modality is not None
        assert box_type_3d.lower() in ('lidar', 'camera')
        # the original label of `DontCare` in kitti is -1 in info file,
        # to distinguish it with other not required classes, we map it
        # from -1 to -2
        self.label_mapping[-1] = -2

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
        if self.modality['use_lidar']:
            if 'plane' in info:
                # convert ground plane to velodyne coordinates
                plane = np.array(info['plane'])
                lidar2cam = np.array(
                    info['images']['CAM2']['lidar2cam'], dtype=np.float32)
                reverse = np.linalg.inv(lidar2cam)

                (plane_norm_cam, plane_off_cam) = (plane[:3],
                                                   -plane[:3] * plane[3])
                plane_norm_lidar = \
                    (reverse[:3, :3] @ plane_norm_cam[:, None])[:, 0]
                plane_off_lidar = (
                    reverse[:3, :3] @ plane_off_cam[:, None][:, 0] +
                    reverse[:3, 3])
                plane_lidar = np.zeros_like(plane_norm_lidar, shape=(4, ))
                plane_lidar[:3] = plane_norm_lidar
                plane_lidar[3] = -plane_norm_lidar.T @ plane_off_lidar
            else:
                plane_lidar = None

            info['plane'] = plane_lidar

        if self.task == 'mono_det' and self.load_eval_anns:
            info['instances'] = info['cam_instances'][self.default_cam_key]

        info = super().parse_data_info(info)

        return info

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)

        valid_data_infos = []
        for data_info in self.data_list:
            ann_info = data_info['ann_info']
            valid_mask = np.ones(ann_info['gt_labels_3d'].shape, dtype=np.bool)

            # Whether to filter `DontCare` objects, which is set to be True
            if self.filter_cfg.get('filter_dontcare', True):
                valid_mask &= ann_info['gt_labels_3d'] > -2

            # Whether to filter object classes not required (not in metainfo)
            # 'filter_class' should be False if ground truth database
            # sampling is used in data augmentation, since objects with
            # labels -1 are kept to for collision detection
            if self.filter_cfg.get('filter_class', False):
                valid_mask &= ann_info['gt_labels_3d'] > -1

            for key in ann_info.keys():
                if key != 'instances':
                    ann_info[key] = (ann_info[key][valid_mask])

            if filter_empty_gt and len(ann_info['gt_labels_3d']) == 0:
                continue
            valid_data_infos.append(data_info)

        return valid_data_infos

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                  3D ground truth bboxes.
                - bbox_labels_3d (np.ndarray): Labels of ground truths.
                - gt_bboxes (np.ndarray): 2D ground truth bboxes.
                - gt_labels (np.ndarray): Labels of ground truths.
                - difficulty (int): Difficulty defined by KITTI.
                  0, 1, 2 represent xxxxx respectively.
        """
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            ann_info = dict()
            # empty instance
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

        # in kitti, lidar2cam = R0_rect @ Tr_velo_to_cam
        lidar2cam = np.array(info['images']['CAM2']['lidar2cam'])
        # convert gt_bboxes_3d to velodyne coordinates with `lidar2cam`
        gt_bboxes_3d = CameraInstance3DBoxes(
            ann_info['gt_bboxes_3d']).convert_to(self.box_mode_3d,
                                                 np.linalg.inv(lidar2cam))
        ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        return ann_info
