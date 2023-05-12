# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.structures import DepthInstance3DBoxes
from .det3d_dataset import Det3DDataset
from .seg3d_dataset import Seg3DDataset


@DATASETS.register_module()
class S3DISDataset(Det3DDataset):
    r"""S3DIS Dataset for Detection Task.

    This class is the inner dataset for S3DIS. Since S3DIS has 6 areas, we
    often train on 5 of them and test on the remaining one. The one for
    test is Area_5 as suggested in `GSDN <https://arxiv.org/abs/2006.12356>`_.
    To concatenate 5 areas during training
    `mmengine.datasets.dataset_wrappers.ConcatDataset` should be used.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_prefix (dict): Prefix for data. Defaults to
            dict(pts='points',
                 pts_instance_mask='instance_mask',
                 pts_semantic_mask='semantic_mask').
        pipeline (List[dict]): Pipeline used for data processing.
            Defaults to [].
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to dict(use_camera=False, use_lidar=True).
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
        'classes': ('table', 'chair', 'sofa', 'bookcase', 'board'),
        # the valid ids of segmentation annotations
        'seg_valid_class_ids': (7, 8, 9, 10, 11),
        'seg_all_class_ids':
        tuple(range(1, 14)),  # possibly with 'stair' class
        'palette': [(170, 120, 200), (255, 0, 0), (200, 100, 100),
                    (10, 200, 100), (200, 200, 200)]
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(
                     pts='points',
                     pts_instance_mask='instance_mask',
                     pts_semantic_mask='semantic_mask'),
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_camera=False, use_lidar=True),
                 box_type_3d: str = 'Depth',
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 **kwargs) -> None:

        # construct seg_label_mapping for semantic mask
        seg_max_cat_id = len(self.METAINFO['seg_all_class_ids'])
        seg_valid_cat_ids = self.METAINFO['seg_valid_class_ids']
        neg_label = len(seg_valid_cat_ids)
        seg_label_mapping = np.ones(
            seg_max_cat_id + 1, dtype=np.int64) * neg_label
        for cls_idx, cat_id in enumerate(seg_valid_cat_ids):
            seg_label_mapping[cat_id] = cls_idx
        self.seg_label_mapping = seg_label_mapping

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

        self.metainfo['seg_label_mapping'] = self.seg_label_mapping
        assert 'use_camera' in self.modality and \
               'use_lidar' in self.modality
        assert self.modality['use_camera'] or self.modality['use_lidar']

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        info['pts_instance_mask_path'] = osp.join(
            self.data_prefix.get('pts_instance_mask', ''),
            info['pts_instance_mask_path'])
        info['pts_semantic_mask_path'] = osp.join(
            self.data_prefix.get('pts_semantic_mask', ''),
            info['pts_semantic_mask_path'])

        info = super().parse_data_info(info)
        # only be used in `PointSegClassMapping` in pipeline
        # to map original semantic class to valid category ids.
        info['seg_label_mapping'] = self.seg_label_mapping
        return info

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Info dict.

        Returns:
            dict: Processed `ann_info`.
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
            box_dim=ann_info['gt_bboxes_3d'].shape[-1],
            with_yaw=False,
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        return ann_info


class _S3DISSegDataset(Seg3DDataset):
    r"""S3DIS Dataset for Semantic Segmentation Task.

    This class is the inner dataset for S3DIS. Since S3DIS has 6 areas, we
    often train on 5 of them and test on the remaining one.
    However, there is not a fixed train-test split of S3DIS. People often test
    on Area_5 as suggested by `SEGCloud <https://arxiv.org/abs/1710.07563>`_.
    But many papers also report the average results of 6-fold cross validation
    over the 6 areas (e.g. `DGCNN <https://arxiv.org/abs/1801.07829>`_).
    Therefore, we use an inner dataset for one area, and further use a dataset
    wrapper to concat all the provided data in different areas.

    Args:
        data_root (str, optional): Path of dataset root, Defaults to None.
        ann_file (str): Path of annotation file. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_prefix (dict): Prefix for training data. Defaults to
            dict(pts='points', pts_instance_mask='', pts_semantic_mask='').
        pipeline (List[dict]): Pipeline used for data processing.
            Defaults to [].
        modality (dict): Modality to specify the sensor data used as input.
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
        'classes':
        ('ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
         'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter'),
        'palette': [[0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 255, 0],
                    [255, 0, 255], [100, 100, 255], [200, 200, 100],
                    [170, 120, 200], [255, 0, 0], [200, 100, 100],
                    [10, 200, 100], [200, 200, 200], [50, 50, 50]],
        'seg_valid_class_ids':
        tuple(range(13)),
        'seg_all_class_ids':
        tuple(range(14))  # possibly with 'stair' class
    }

    def __init__(self,
                 data_root: Optional[str] = None,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(
                     pts='points', pts_instance_mask='', pts_semantic_mask=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True, use_camera=False),
                 ignore_index: Optional[int] = None,
                 scene_idxs: Optional[Union[np.ndarray, str]] = None,
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

    def get_scene_idxs(self, scene_idxs: Union[np.ndarray, str,
                                               None]) -> np.ndarray:
        """Compute scene_idxs for data sampling.

        We sample more times for scenes with more points.
        """
        # when testing, we load one whole scene every time
        if not self.test_mode and scene_idxs is None:
            raise NotImplementedError(
                'please provide re-sampled scene indexes for training')

        return super().get_scene_idxs(scene_idxs)


@DATASETS.register_module()
class S3DISSegDataset(_S3DISSegDataset):
    r"""S3DIS Dataset for Semantic Segmentation Task.

    This class serves as the API for experiments on the S3DIS Dataset.
    It wraps the provided datasets of different areas.
    We don't use `mmdet.datasets.dataset_wrappers.ConcatDataset` because we
    need to concat the `scene_idxs` of different areas.

    Please refer to the `google form <https://docs.google.com/forms/d/e/1FAIpQL
    ScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1>`_ for
    data downloading.

    Args:
        data_root (str, optional): Path of dataset root. Defaults to None.
        ann_files (List[str]): Path of several annotation files.
            Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_prefix (dict): Prefix for training data. Defaults to
            dict(pts='points', pts_instance_mask='', pts_semantic_mask='').
        pipeline (List[dict]): Pipeline used for data processing.
            Defaults to [].
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to dict(use_lidar=True, use_camera=False).
        ignore_index (int, optional): The label index to be ignored, e.g.
            unannotated points. If None is given, set to len(self.classes) to
            be consistent with PointSegClassMapping function in pipeline.
            Defaults to None.
        scene_idxs (List[np.ndarray] | List[str], optional): Precomputed index
            to load data. For scenes with many points, we may sample it
            several times. Defaults to None.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
    """

    def __init__(self,
                 data_root: Optional[str] = None,
                 ann_files: List[str] = '',
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(
                     pts='points', pts_instance_mask='', pts_semantic_mask=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True, use_camera=False),
                 ignore_index: Optional[int] = None,
                 scene_idxs: Optional[Union[List[np.ndarray],
                                            List[str]]] = None,
                 test_mode: bool = False,
                 **kwargs) -> None:

        # make sure that ann_files and scene_idxs have same length
        ann_files = self._check_ann_files(ann_files)
        scene_idxs = self._check_scene_idxs(scene_idxs, len(ann_files))

        # initialize some attributes as datasets[0]
        super().__init__(
            data_root=data_root,
            ann_file=ann_files[0],
            metainfo=metainfo,
            data_prefix=data_prefix,
            pipeline=pipeline,
            modality=modality,
            ignore_index=ignore_index,
            scene_idxs=scene_idxs[0],
            test_mode=test_mode,
            **kwargs)

        datasets = [
            _S3DISSegDataset(
                data_root=data_root,
                ann_file=ann_files[i],
                metainfo=metainfo,
                data_prefix=data_prefix,
                pipeline=pipeline,
                modality=modality,
                ignore_index=ignore_index,
                scene_idxs=scene_idxs[i],
                test_mode=test_mode,
                **kwargs) for i in range(len(ann_files))
        ]

        # data_list and scene_idxs need to be concat
        self.concat_data_list([dst.data_list for dst in datasets])

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

    def concat_data_list(self, data_lists: List[List[dict]]) -> None:
        """Concat data_list from several datasets to form self.data_list.

        Args:
            data_lists (List[List[dict]]): List of dict containing
                annotation information.
        """
        self.data_list = [
            data for data_list in data_lists for data in data_list
        ]

    @staticmethod
    def _duplicate_to_list(x: Any, num: int) -> list:
        """Repeat x `num` times to form a list."""
        return [x for _ in range(num)]

    def _check_ann_files(
            self, ann_file: Union[List[str], Tuple[str], str]) -> List[str]:
        """Make ann_files as list/tuple."""
        # ann_file could be str
        if not isinstance(ann_file, (list, tuple)):
            ann_file = self._duplicate_to_list(ann_file, 1)
        return ann_file

    def _check_scene_idxs(self, scene_idx: Union[str, List[Union[list, tuple,
                                                                 np.ndarray]],
                                                 List[str], None],
                          num: int) -> List[np.ndarray]:
        """Make scene_idxs as list/tuple."""
        if scene_idx is None:
            return self._duplicate_to_list(scene_idx, num)
        # scene_idx could be str, np.ndarray, list or tuple
        if isinstance(scene_idx, str):  # str
            return self._duplicate_to_list(scene_idx, num)
        if isinstance(scene_idx[0], str):  # list of str
            return scene_idx
        if isinstance(scene_idx[0], (list, tuple, np.ndarray)):  # list of idx
            return scene_idx
        # single idx
        return self._duplicate_to_list(scene_idx, num)
