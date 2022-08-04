# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp
from typing import Callable, List, Optional, Union

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.structures import DepthInstance3DBoxes
from .det3d_dataset import Det3DDataset
from .seg3d_dataset import Seg3DDataset
from .transforms import Compose


@DATASETS.register_module()
class S3DISDataset(Det3DDataset):
    r"""S3DIS Dataset for Detection Task.

    This class is the inner dataset for S3DIS. Since S3DIS has 6 areas, we
    often train on 5 of them and test on the remaining one. The one for
    test is Area_5 as suggested in `GSDN <https://arxiv.org/abs/2006.12356>`_.
    To concatenate 5 areas during training
    `mmdet.datasets.dataset_wrappers.ConcatDataset` should be used.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
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
    CLASSES = ('table', 'chair', 'sofa', 'bookcase', 'board')

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='Depth',
                 filter_empty_gt=True,
                 test_mode=False,
                 *kwargs):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            *kwargs)

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`DepthInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - pts_instance_mask_path (str): Path of instance masks.
                - pts_semantic_mask_path (str): Path of semantic masks.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]
        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(
                np.float32)  # k, 6
            gt_labels_3d = info['annos']['class'].astype(np.int64)
        else:
            gt_bboxes_3d = np.zeros((0, 6), dtype=np.float32)
            gt_labels_3d = np.zeros((0, ), dtype=np.int64)

        # to target box structure
        gt_bboxes_3d = DepthInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            with_yaw=False,
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        pts_instance_mask_path = osp.join(self.data_root,
                                          info['pts_instance_mask_path'])
        pts_semantic_mask_path = osp.join(self.data_root,
                                          info['pts_semantic_mask_path'])

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            pts_instance_mask_path=pts_instance_mask_path,
            pts_semantic_mask_path=pts_semantic_mask_path)
        return anns_results

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing transforms. It includes the following keys:

                - pts_filename (str): Filename of point clouds.
                - file_name (str): Filename of point clouds.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        pts_filename = osp.join(self.data_root, info['pts_path'])
        input_dict = dict(pts_filename=pts_filename)

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if self.filter_empty_gt and ~(annos['gt_labels_3d'] != -1).any():
                return None
        return input_dict

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                load_dim=6,
                use_dim=[0, 1, 2, 3, 4, 5]),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        return Compose(pipeline)


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
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        palette (list[list[int]], optional): The palette of segmentation map.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        ignore_index (int, optional): The label index to be ignored, e.g.
            unannotated points. If None is given, set to len(self.CLASSES).
            Defaults to None.
        scene_idxs (np.ndarray | str, optional): Precomputed index to load
            data. For scenes with many points, we may sample it several times.
            Defaults to None.
    """
    METAINFO = {
        'CLASSES':
        ('ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
         'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter'),
        'PALETTE': [[0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 255, 0],
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

    def get_scene_idxs(self, scene_idxs):
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
        data_root (str): Path of dataset root.
        ann_files (list[str]): Path of several annotation files.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        palette (list[list[int]], optional): The palette of segmentation map.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        ignore_index (int, optional): The label index to be ignored, e.g.
            unannotated points. If None is given, set to len(self.CLASSES).
            Defaults to None.
        scene_idxs (list[np.ndarray] | list[str], optional): Precomputed index
            to load data. For scenes with many points, we may sample it several
            times. Defaults to None.
    """

    def __init__(self,
                 data_root: Optional[str] = None,
                 ann_files: str = '',
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(
                     pts='points', img='', instance_mask='', semantic_mask=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True, use_camera=False),
                 ignore_index=None,
                 scene_idxs=None,
                 test_mode=False,
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
            serialize_data=False,
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
                serialize_data=False,
                **kwargs) for i in range(len(ann_files))
        ]

        # data_list and scene_idxs need to be concat
        self.concat_data_list([dst.data_list for dst in datasets])
        self.concat_scene_idxs([dst.scene_idxs for dst in datasets])

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

    def concat_data_list(self, data_lists):
        """Concat data_list from several datasets to form self.data_list.

        Args:
            data_lists (list[list[dict]])
        """
        self.data_list = [
            data for data_list in data_lists for data in data_list
        ]

    def concat_scene_idxs(self, scene_idxs):
        """Concat scene_idxs from several datasets to form self.scene_idxs.

        Needs to manually add offset to scene_idxs[1, 2, ...].

        Args:
            scene_idxs (list[np.ndarray])
        """
        self.scene_idxs = np.array([], dtype=np.int32)
        offset = 0
        for one_scene_idxs in scene_idxs:
            self.scene_idxs = np.concatenate(
                [self.scene_idxs, one_scene_idxs + offset]).astype(np.int32)
            offset = np.unique(self.scene_idxs).max() + 1

    @staticmethod
    def _duplicate_to_list(x, num):
        """Repeat x `num` times to form a list."""
        return [x for _ in range(num)]

    def _check_ann_files(self, ann_file):
        """Make ann_files as list/tuple."""
        # ann_file could be str
        if not isinstance(ann_file, (list, tuple)):
            ann_file = self._duplicate_to_list(ann_file, 1)
        return ann_file

    def _check_scene_idxs(self, scene_idx, num):
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
