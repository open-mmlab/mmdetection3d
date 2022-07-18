# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Optional, Union

import numpy as np

from mmdet3d.core.bbox import DepthInstance3DBoxes
from mmdet3d.registry import DATASETS
from .det3d_dataset import Det3DDataset


@DATASETS.register_module()
class SUNRGBDDataset(Det3DDataset):
    r"""SUNRGBD Dataset.

    This class serves as the API for experiments on the SUNRGBD Dataset.

    See the `download page <http://rgbd.cs.princeton.edu/challenge.html>`_
    for data downloading.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_prefix (dict, optional): Prefix for data. Defaults to
            `dict(pts='points',img='sunrgbd_trainval')`.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to `dict(use_camera=True, use_lidar=True)`.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'Depth' in this dataset. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """
    METAINFO = {
        'CLASSES': ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk',
                    'dresser', 'night_stand', 'bookshelf', 'bathtub')
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(
                     pts='points', img='sunrgbd_trainval'),
                 pipeline: List[Union[dict, Callable]] = [],
                 modality=dict(use_camera=True, use_lidar=True),
                 box_type_3d: str = 'Depth',
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 **kwargs):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            metainfo=metainfo,
            data_prefix=data_prefix,
            pipeline=pipeline,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)
        assert 'use_camera' in self.modality and \
            'use_lidar' in self.modality
        assert self.modality['use_camera'] or self.modality['use_lidar']

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`

        Args:
            info (dict): Info dict.

        Returns:
            dict: Processed `ann_info`
        """
        ann_info = super().parse_ann_info(info)
        # empty gt
        if ann_info is None:
            ann_info = dict()
            ann_info['gt_bboxes_3d'] = np.zeros((0, 6), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros((0, ), dtype=np.int64)

        # to target box structure
        ann_info['gt_bboxes_3d'] = DepthInstance3DBoxes(
            ann_info['gt_bboxes_3d'],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        return ann_info
