# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from os import path as osp
from typing import Callable, List, Optional, Union

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.structures import DepthInstance3DBoxes
from .det3d_dataset import Det3DDataset
from .seg3d_dataset import Seg3DDataset


@DATASETS.register_module()
class ScanNetDataset(Det3DDataset):
    r"""ScanNet Dataset for Detection Task.

    This class serves as the API for experiments on the ScanNet Dataset.

    Please refer to the `github repo <https://github.com/ScanNet/ScanNet>`_
    for data downloading.

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
        'classes':
        ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
         'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
         'showercurtrain', 'toilet', 'sink', 'bathtub', 'garbagebin'),
        # the valid ids of segmentation annotations
        'seg_valid_class_ids':
        (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39),
        'seg_all_class_ids':
        tuple(range(1, 41)),
        'palette': [(31, 119, 180), (255, 187, 120), (188, 189, 34),
                    (140, 86, 75), (255, 152, 150), (214, 39, 40),
                    (197, 176, 213), (148, 103, 189), (196, 156, 148),
                    (23, 190, 207), (247, 182, 210), (219, 219, 141),
                    (255, 127, 14), (158, 218, 229), (44, 160, 44),
                    (112, 128, 144), (227, 119, 194), (82, 84, 163)]
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

        The only difference with it in `Det3DDataset`
        is the specific process for `axis_align_matrix'.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        info['axis_align_matrix'] = self._get_axis_align_matrix(info)
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


@DATASETS.register_module()
class ScanNetSegDataset(Seg3DDataset):
    r"""ScanNet Dataset for Semantic Segmentation Task.

    This class serves as the API for experiments on the ScanNet Dataset.

    Please refer to the `github repo <https://github.com/ScanNet/ScanNet>`_
    for data downloading.

    Args:
        data_root (str, optional): Path of dataset root. Defaults to None.
        ann_file (str): Path of annotation file. Defaults to ''.
        pipeline (List[dict]): Pipeline used for data processing.
            Defaults to [].
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_prefix (dict): Prefix for training data. Defaults to
            dict(pts='points',
                 img='',
                 pts_instance_mask='',
                 pts_semantic_mask='').
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
        ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
         'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
         'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
         'otherfurniture'),
        'palette': [
            [174, 199, 232],
            [152, 223, 138],
            [31, 119, 180],
            [255, 187, 120],
            [188, 189, 34],
            [140, 86, 75],
            [255, 152, 150],
            [214, 39, 40],
            [197, 176, 213],
            [148, 103, 189],
            [196, 156, 148],
            [23, 190, 207],
            [247, 182, 210],
            [219, 219, 141],
            [255, 127, 14],
            [158, 218, 229],
            [44, 160, 44],
            [112, 128, 144],
            [227, 119, 194],
            [82, 84, 163],
        ],
        'seg_valid_class_ids': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16,
                                24, 28, 33, 34, 36, 39),
        'seg_all_class_ids':
        tuple(range(41)),
    }

    def __init__(self,
                 data_root: Optional[str] = None,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(
                     pts='points',
                     img='',
                     pts_instance_mask='',
                     pts_semantic_mask=''),
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
class ScanNetInstanceSegDataset(Seg3DDataset):

    METAINFO = {
        'classes':
        ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
         'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
         'showercurtrain', 'toilet', 'sink', 'bathtub', 'garbagebin'),
        'palette': [
            [174, 199, 232],
            [152, 223, 138],
            [31, 119, 180],
            [255, 187, 120],
            [188, 189, 34],
            [140, 86, 75],
            [255, 152, 150],
            [214, 39, 40],
            [197, 176, 213],
            [148, 103, 189],
            [196, 156, 148],
            [23, 190, 207],
            [247, 182, 210],
            [219, 219, 141],
            [255, 127, 14],
            [158, 218, 229],
            [44, 160, 44],
            [112, 128, 144],
            [227, 119, 194],
            [82, 84, 163],
        ],
        'seg_valid_class_ids':
        (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39),
        'seg_all_class_ids':
        tuple(range(41))
    }

    def __init__(self,
                 data_root: Optional[str] = None,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(
                     pts='points',
                     img='',
                     pts_instance_mask='',
                     pts_semantic_mask=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True, use_camera=False),
                 test_mode: bool = False,
                 ignore_index: Optional[int] = None,
                 scene_idxs: Optional[Union[np.ndarray, str]] = None,
                 backend_args: Optional[dict] = None,
                 **kwargs) -> None:
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            metainfo=metainfo,
            pipeline=pipeline,
            data_prefix=data_prefix,
            modality=modality,
            test_mode=test_mode,
            ignore_index=ignore_index,
            scene_idxs=scene_idxs,
            backend_args=backend_args,
            **kwargs)


@DATASETS.register_module()
class ScanNetMultiViewDataset(Det3DDataset):
    r"""ScanNet Dataset for NeRF-detection Task
    
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
        'classes':
        ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
         'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
         'showercurtrain', 'toilet', 'sink', 'bathtub', 'garbagebin'),
        # the valid ids of segmentation annotations
        'seg_valid_class_ids':
        (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39),
        'seg_all_class_ids':
        tuple(range(1, 41)),
        'palette': [(31, 119, 180), (255, 187, 120), (188, 189, 34),
                    (140, 86, 75), (255, 152, 150), (214, 39, 40),
                    (197, 176, 213), (148, 103, 189), (196, 156, 148),
                    (23, 190, 207), (247, 182, 210), (219, 219, 141),
                    (255, 127, 14), (158, 218, 229), (44, 160, 44),
                    (112, 128, 144), (227, 119, 194), (82, 84, 163)]
    }
    
    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 metainfo: Optional[dict] = None,
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict=dict(use_camera=False, use_depth=True),
                 box_type_3d: str = 'Depth',
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 **kwargs) -> None:
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            metainfo=metainfo,
            pipeline=pipeline,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)
        
        assert 'use_camera' in self.modality and \
               'use_lidar' in self.modality
        assert self.modality['use_camera'] or self.modality['use_depth']
        
    def get_data_info(self, index):
        """Get data info according to the given index.
        
        Args:
            index(int): Index of the sample data to get.
            
        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:
                
                -depth_info
                -img_prefix
                -img_info
                -lidar2img
                    --extrinsic
                    --intrinsic
                    --origin
                -c2w
                -camrotc2w
                -lightpos
                -ray_info
                    --c2w
                    --camrotc2w
                    --lightpos
                -anno_info
                    --gt_bboxes_3d
                    --gt_labels_3d
                    --axis_align_matrix
        """
        import mmengine
        from collections import defaultdict
        info = mmengine.load(self.ann_file)
        input_dict = defaultdict(list)
        
        if self.modality is not None:
            if self.modality["use_depth"]:
                input_dict['depth_info'] = []
            if self.modality["use_neuralrecon_depth"]:
                input_dict['depth_info'] = []
            if self.modality["use_lidar"]:
                sample_idx = info['point_cloud']['lidar_idx']
                pts_filename = osp.join(self.data_root, info['pts_path'])
                input_dict['pts_filename'] = pts_filename
                input_dict['sample_idx'] = sample_idx
                input_dict['file_name'] = pts_filename
                
        axis_align_matrix = info['annos']['axis_align_matrix'].astype(np.float32)
        
        for i in range(len(info['img_paths'])):
            img_filename = osp.join(self.data_root, info['img_paths'][i])
            input_dict['img_prefix'].append(None)
            input_dict['img_info'].append(dict(filename=img_filename))
            if "depth_info" in input_dict.keys():
                if self.modality["use_neuralrecon_depth"]:
                    input_dict['depth_info'].append(dict(filename=img_filename[:-4]+'.npy'))
                else:
                    input_dict['depth_info'].append(dict(filename=img_filename[:-4]+'.png'))
            # implement lidar_info in input.keys() in the future.
            extrinsic = np.linalg.inv(axis_align_matrix @ info['extrinsics'][i])
            input_dict['lidar2img'].append(extrinsic.astype(np.float32))
            if self.modality["use_ray"]:
                c2w = (axis_align_matrix @ info['extrinsics'][i]).astype(np.float32)
                input_dict['c2w'].append(c2w)
                input_dict['camrotc2w'].append(c2w[0:3, 0:3])
                input_dict['lightpos'].append(c2w[0:3, 3])
                
        input_dict = dict(input_dict)
        origin = np.array([.0, .0, .5])
        input_dict['lidar2img'] = dict(
            extrinsic=input_dict['lidar2img'],
            intrinsic=info['intrinsics'].astype(np.float32),
            origin=origin.astype(np.float32)
        )
        
        if self.modality["use_ray"]:
            input_dict['ray_info'] = dict(
            c2w=input_dict['c2w'],
            camrotc2w=input_dict['camrotc2w'],
            lightpos=input_dict['lightpos'],
        )

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if self.filter_empty_gt and len(annos['gt_bboxes_3d']) == 0:
                return None
        else:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict
    
    def get_ann_info(self, index: int) -> dict:
        """Get annotation info according to the given index.
        
        Use index to get the corresponding annotations, thus the 
        evalhook could use this api.
        
        Args:
            index (int): Index of the annotation data to get.
            
        Returns:
            dict: Annotation information.
        """
        # info = self.get_data_info(index) # need double check
        import mmengine
        info = mmengine.load(self.ann_file)
        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(
                np.float32)  #k, 6
            gt_labels_3d = info['annos']['class'].astype(np.long)
        else:
            gt_bboxes_3d = np.zeros((0, 6), dtype=np.float32)
            gt_labels_3d = np.zeros((0,), dtype=np.long)
            
        from mmdet3d.structures.bbox_3d import DepthInstance3DBoxes
        # to target box structure
        gt_bboxes_3d = DepthInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            with_yaw=False,
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        
        axis_align_matrix = info['annos']['axis_align_matrix'].astype(np.float32)
        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            axis_align_matrix=axis_align_matrix)
        return anns_results
        