# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Optional, Union

import numpy as np

from mmdet3d.registry import DATASETS
from .seg3d_dataset import Seg3DDataset


@DATASETS.register_module()
class SemanticKittiDataset(Seg3DDataset):
    r"""SemanticKitti Dataset.

    This class serves as the API for experiments on the SemanticKITTI Dataset
    Please refer to <http://www.semantic-kitti.org/dataset.html>`_
    for data downloading

    Args:
        data_root (str, optional): Path of dataset root. Defaults to None.
        ann_file (str): Path of annotation file. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_prefix (dict): Prefix for training data. Defaults to
            dict(pts='',
                 img='',
                 pts_instance_mask='',
                 pts_semantic_mask='').
        pipeline (List[dict]): Pipeline used for data processing.
            Defaults to [].
        modality (dict): Modality to specify the sensor data used as input,
            it usually has following keys:

                - use_camera: bool
                - use_lidar: bool
            Defaults to dict(use_lidar=True, use_camera=False).
        ignore_index (int, optional): The label index to be ignored, e.g.
            unannotated points. If None is given, set to len(self.classes) to
            be consistent with PointSegClassMapping function in pipeline.
            Defaults to None.
        scene_idxs (np.ndarray or str, optional): Precomputed index to load
            data. For scenes with many points, we may sample it several times.
            Defaults to None.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
    """
    METAINFO = {
        'classes': ('car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person',
                    'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
                    'other-ground', 'building', 'fence', 'vegetation',
                    'trunck', 'terrian', 'pole', 'traffic-sign'),
        'palette': [[100, 150, 245], [100, 230, 245], [30, 60, 150],
                    [80, 30, 180], [100, 80, 250], [155, 30, 30],
                    [255, 40, 200], [150, 30, 90], [255, 0, 255],
                    [255, 150, 255], [75, 0, 75], [175, 0, 75], [255, 200, 0],
                    [255, 120, 50], [0, 175, 0], [135, 60, 0], [150, 240, 80],
                    [255, 240, 150], [255, 0, 0]],
        'seg_valid_class_ids':
        tuple(range(19)),
        'seg_all_class_ids':
        tuple(range(19)),
    }

    def __init__(self,
                 data_root: Optional[str] = None,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(
                     pts='',
                     img='',
                     pts_instance_mask='',
                     pts_semantic_mask=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True, use_camera=False),
                 ignore_index: Optional[int] = None,
                 scene_idxs: Optional[Union[str, np.ndarray]] = None,
                 test_mode: bool = False,
                 **kwargs) -> None:

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            metainfo=metainfo,
            data_prefix=data_prefix,
            pipeline=pipeline,
            modality=modality,
            ignore_index=ignore_index,
            scene_idxs=scene_idxs,
            test_mode=test_mode,
            **kwargs)

    def get_seg_label_mapping(self, metainfo):
        seg_label_mapping = np.zeros(metainfo['max_label'] + 1, dtype=np.int64)
        for idx in metainfo['seg_label_mapping']:
            seg_label_mapping[idx] = metainfo['seg_label_mapping'][idx]
        return seg_label_mapping
