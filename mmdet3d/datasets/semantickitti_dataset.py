# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Optional, Union

from mmdet3d.registry import DATASETS
from .seg3d_dataset import Seg3DDataset


@DATASETS.register_module()
class SemanticKITTIDataset(Seg3DDataset):
    r"""SemanticKITTI Dataset.

    This class serves as the API for experiments on the SemanticKITTI Dataset
    Please refer to <http://www.semantic-kitti.org/dataset.html>`_
    for data downloading

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): NO 3D box for this dataset.
            You can choose any type
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """
    METAINFO = {
        'CLASSES': ('unlabeled', 'car', 'bicycle', 'motorcycle', 'truck',
                    'bus', 'person', 'bicyclist', 'motorcyclist', 'road',
                    'parking', 'sidewalk', 'other-ground', 'building', 'fence',
                    'vegetation', 'trunck', 'terrian', 'pole', 'traffic-sign'),
        'seg_valid_class_ids':
        tuple(range(20)),
        'seg_all_class_ids':
        tuple(range(20))
    }

    def __init__(self,
                 data_root: Optional[str] = None,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(
                     pts='points', img='', instance_mask='', semantic_mask=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True, use_camera=False),
                 ignore_index=None,
                 scene_idxs=None,
                 test_mode=False,
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
