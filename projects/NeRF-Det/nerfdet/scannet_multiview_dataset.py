# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from os import path as osp
from typing import Callable, List, Optional, Union

import numpy as np

from mmdet3d.datasets import Det3DDataset
from mmdet3d.registry import DATASETS
from mmdet3d.structures import DepthInstance3DBoxes


@DATASETS.register_module()
class MultiViewScanNetDataset(Det3DDataset):
    r"""Multi-View ScanNet Dataset for NeRF-detection Task

    This class serves as the API for experiments on the ScanNet Dataset.

    Please refer to the `github repo <https://github.com/ScanNet/ScanNet>`_
    for data downloading.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        pipeline (List[dict]): Pipeline used for data processing.
            Defaults to [].
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to dict(use_camera=True, use_lidar=False).
        box_type_3d (str): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'Depth' in this dataset. Available options includes:

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool): Whether to filter the data with empty GT.
            If it's set to be True, the example with empty annotations after
            data pipeline will be dropped and a random example will be chosen
            in `__getitem__`. Defaults to True.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
    """
    METAINFO = {
        'classes':
        ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
         'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
         'showercurtrain', 'toilet', 'sink', 'bathtub', 'garbagebin')
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 metainfo: Optional[dict] = None,
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_camera=True, use_lidar=False),
                 box_type_3d: str = 'Depth',
                 filter_empty_gt: bool = True,
                 remove_dontcare: bool = False,
                 test_mode: bool = False,
                 **kwargs) -> None:

        self.remove_dontcare = remove_dontcare

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            metainfo=metainfo,
            pipeline=pipeline,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)

        assert 'use_camera' in self.modality and \
               'use_lidar' in self.modality
        assert self.modality['use_camera'] or self.modality['use_lidar']

    @staticmethod
    def _get_axis_align_matrix(info: dict) -> np.ndarray:
        """Get axis_align_matrix from info. If not exist, return identity mat.

        Args:
            info (dict): Info of a single sample data.

        Returns:
            np.ndarray: 4x4 transformation matrix.
        """
        if 'axis_align_matrix' in info:
            return np.array(info['axis_align_matrix'])
        else:
            warnings.warn(
                'axis_align_matrix is not found in ScanNet data info, please '
                'use new pre-process scripts to re-generate ScanNet data')
            return np.eye(4).astype(np.float32)

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        Convert all relative path of needed modality data file to
        the absolute path.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        if self.modality['use_depth']:
            info['depth_info'] = []
        if self.modality['use_neuralrecon_depth']:
            info['depth_info'] = []

        if self.modality['use_lidar']:
            # implement lidar processing in the future
            raise NotImplementedError(
                'Please modified '
                '`MultiViewPipeline` to support lidar processing')

        info['axis_align_matrix'] = self._get_axis_align_matrix(info)
        info['img_info'] = []
        info['lidar2img'] = []
        info['c2w'] = []
        info['camrotc2w'] = []
        info['lightpos'] = []
        # load img and depth_img
        for i in range(len(info['img_paths'])):
            img_filename = osp.join(self.data_root, info['img_paths'][i])

            info['img_info'].append(dict(filename=img_filename))
            if 'depth_info' in info.keys():
                if self.modality['use_neuralrecon_depth']:
                    info['depth_info'].append(
                        dict(filename=img_filename[:-4] + '.npy'))
                else:
                    info['depth_info'].append(
                        dict(filename=img_filename[:-4] + '.png'))
            # implement lidar_info in input.keys() in the future.
            extrinsic = np.linalg.inv(
                info['axis_align_matrix'] @ info['lidar2cam'][i])
            info['lidar2img'].append(extrinsic.astype(np.float32))
            if self.modality['use_ray']:
                c2w = (
                    info['axis_align_matrix'] @ info['lidar2cam'][i]).astype(
                        np.float32)  # noqa
                info['c2w'].append(c2w)
                info['camrotc2w'].append(c2w[0:3, 0:3])
                info['lightpos'].append(c2w[0:3, 3])
        origin = np.array([.0, .0, .5])
        info['lidar2img'] = dict(
            extrinsic=info['lidar2img'],
            intrinsic=info['cam2img'].astype(np.float32),
            origin=origin.astype(np.float32))

        if self.modality['use_ray']:
            info['ray_info'] = []

        if not self.test_mode:
            info['ann_info'] = self.parse_ann_info(info)
        if self.test_mode and self.load_eval_anns:
            info['ann_info'] = self.parse_ann_info(info)
            info['eval_ann_info'] = self._remove_dontcare(info['ann_info'])

        return info

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Info dict.

        Returns:
            dict: Processed `ann_info`.
        """
        ann_info = super().parse_ann_info(info)

        if self.remove_dontcare:
            ann_info = self._remove_dontcare(ann_info)

        # empty gt
        if ann_info is None:
            ann_info = dict()
            ann_info['gt_bboxes_3d'] = np.zeros((0, 6), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros((0, ), dtype=np.int64)

        ann_info['gt_bboxes_3d'] = DepthInstance3DBoxes(
            ann_info['gt_bboxes_3d'],
            box_dim=ann_info['gt_bboxes_3d'].shape[-1],
            with_yaw=False,
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        # count the numbers
        for label in ann_info['gt_labels_3d']:
            if label != -1:
                cat_name = self.metainfo['classes'][label]
                self.num_ins_per_cat[cat_name] += 1

        return ann_info
