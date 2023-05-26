# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
from mmengine.dataset import BaseDataset
from mmengine.fileio import get_local_path

from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class Seg3DDataset(BaseDataset):
    """Base Class for 3D semantic segmentation dataset.

    This is the base dataset of ScanNet, S3DIS and SemanticKITTI dataset.

    Args:
        data_root (str, optional): Path of dataset root. Defaults to None.
        ann_file (str): Path of annotation file. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_prefix (dict): Prefix for training data. Defaults to
            dict(pts='points',
                 img='',
                 pts_instance_mask='',
                 pts_semantic_mask='').
        pipeline (List[dict]): Pipeline used for data processing.
            Defaults to [].
        modality (dict): Modality to specify the sensor data used
            as input, it usually has following keys:

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
        serialize_data (bool): Whether to hold memory using serialized objects,
            when enabled, data loader workers can use shared RAM from master
            process instead of making a copy.
            Defaults to False for 3D Segmentation datasets.
        load_eval_anns (bool): Whether to load annotations in test_mode,
            the annotation will be save in `eval_ann_infos`, which can be used
            in Evaluator. Defaults to True.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """
    METAINFO = {
        'classes': None,  # names of all classes data used for the task
        'palette': None,  # official color for visualization
        'seg_valid_class_ids': None,  # class_ids used for training
        'seg_all_class_ids': None,  # all possible class_ids in loaded seg mask
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
                 scene_idxs: Optional[Union[str, np.ndarray]] = None,
                 test_mode: bool = False,
                 serialize_data: bool = False,
                 load_eval_anns: bool = True,
                 backend_args: Optional[dict] = None,
                 **kwargs) -> None:
        self.backend_args = backend_args
        self.modality = modality
        self.load_eval_anns = load_eval_anns

        # TODO: We maintain the ignore_index attributes,
        # but we may consider to remove it in the future.
        self.ignore_index = len(self.METAINFO['classes']) if \
            ignore_index is None else ignore_index

        # Get label mapping for custom classes
        new_classes = metainfo.get('classes', None)

        self.label_mapping, self.label2cat, seg_valid_class_ids = \
            self.get_label_mapping(new_classes)

        metainfo['label_mapping'] = self.label_mapping
        metainfo['label2cat'] = self.label2cat
        metainfo['ignore_index'] = self.ignore_index
        metainfo['seg_valid_class_ids'] = seg_valid_class_ids

        # generate palette if it is not defined based on
        # label mapping, otherwise directly use palette
        # defined in dataset config.
        palette = metainfo.get('palette', None)
        updated_palette = self._update_palette(new_classes, palette)

        metainfo['palette'] = updated_palette

        # construct seg_label_mapping for semantic mask
        self.seg_label_mapping = self.get_seg_label_mapping(metainfo)

        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=pipeline,
            test_mode=test_mode,
            serialize_data=serialize_data,
            **kwargs)

        self.metainfo['seg_label_mapping'] = self.seg_label_mapping
        if not kwargs.get('lazy_init', False):
            self.scene_idxs = self.get_scene_idxs(scene_idxs)
            self.data_list = [self.data_list[i] for i in self.scene_idxs]

            # set group flag for the sampler
            if not self.test_mode:
                self._set_group_flag()

    def get_label_mapping(self,
                          new_classes: Optional[Sequence] = None) -> tuple:
        """Get label mapping.

        The ``label_mapping`` is a dictionary, its keys are the old label ids
        and its values are the new label ids, and is used for changing pixel
        labels in load_annotations. If and only if old classes in cls.METAINFO
        is not equal to new classes in self._metainfo and nether of them is not
        None, `label_mapping` is not None.

        Args:
            new_classes (list or tuple, optional): The new classes name from
                metainfo. Defaults to None.

        Returns:
            tuple: The mapping from old classes in cls.METAINFO to
            new classes in metainfo
        """
        old_classes = self.METAINFO.get('classes', None)
        if (new_classes is not None and old_classes is not None
                and list(new_classes) != list(old_classes)):
            if not set(new_classes).issubset(old_classes):
                raise ValueError(
                    f'new classes {new_classes} is not a '
                    f'subset of classes {old_classes} in METAINFO.')

            # obtain true id from valid_class_ids
            valid_class_ids = [
                self.METAINFO['seg_valid_class_ids'][old_classes.index(
                    cls_name)] for cls_name in new_classes
            ]
            label_mapping = {
                cls_id: self.ignore_index
                for cls_id in self.METAINFO['seg_all_class_ids']
            }
            label_mapping.update(
                {cls_id: i
                 for i, cls_id in enumerate(valid_class_ids)})
            label2cat = {i: cat_name for i, cat_name in enumerate(new_classes)}
        else:
            label_mapping = {
                cls_id: self.ignore_index
                for cls_id in self.METAINFO['seg_all_class_ids']
            }
            label_mapping.update({
                cls_id: i
                for i, cls_id in enumerate(
                    self.METAINFO['seg_valid_class_ids'])
            })
            # map label to category name
            label2cat = {
                i: cat_name
                for i, cat_name in enumerate(self.METAINFO['classes'])
            }
            valid_class_ids = self.METAINFO['seg_valid_class_ids']

        return label_mapping, label2cat, valid_class_ids

    def get_seg_label_mapping(self, metainfo=None):
        """Get segmentation label mapping.

        The ``seg_label_mapping`` is an array, its indices are the old label
        ids and its values are the new label ids, and is specifically used
        for changing point labels in PointSegClassMapping.

        Args:
            metainfo (dict, optional): Meta information to set
            seg_label_mapping. Defaults to None.

        Returns:
            tuple: The mapping from old classes to new classes.
        """
        seg_max_cat_id = len(self.METAINFO['seg_all_class_ids'])
        seg_valid_cat_ids = self.METAINFO['seg_valid_class_ids']
        neg_label = len(seg_valid_cat_ids)
        seg_label_mapping = np.ones(
            seg_max_cat_id + 1, dtype=np.int64) * neg_label
        for cls_idx, cat_id in enumerate(seg_valid_cat_ids):
            seg_label_mapping[cat_id] = cls_idx
        return seg_label_mapping

    def _update_palette(self, new_classes: list, palette: Union[None,
                                                                list]) -> list:
        """Update palette according to metainfo.

        If length of palette is equal to classes, just return the palette.
        If palette is not defined, it will randomly generate a palette.
        If classes is updated by customer, it will return the subset of
        palette.

        Returns:
            Sequence: Palette for current dataset.
        """
        if palette is None:
            # If palette is not defined, it generate a palette according
            # to the original palette and classes.
            old_classes = self.METAINFO.get('classes', None)
            palette = [
                self.METAINFO['palette'][old_classes.index(cls_name)]
                for cls_name in new_classes
            ]
            return palette

        # palette does match classes
        if len(palette) == len(new_classes):
            return palette
        else:
            raise ValueError('Once palette in set in metainfo, it should'
                             'match classes in metainfo')

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        Convert all relative path of needed modality data file to
        the absolute path. And process
        the `instances` field to `ann_info` in training stage.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        if self.modality['use_lidar']:
            info['lidar_points']['lidar_path'] = \
                osp.join(
                    self.data_prefix.get('pts', ''),
                    info['lidar_points']['lidar_path'])
            if 'num_pts_feats' in info['lidar_points']:
                info['num_pts_feats'] = info['lidar_points']['num_pts_feats']
            info['lidar_path'] = info['lidar_points']['lidar_path']

        if self.modality['use_camera']:
            for cam_id, img_info in info['images'].items():
                if 'img_path' in img_info:
                    img_info['img_path'] = osp.join(
                        self.data_prefix.get('img', ''), img_info['img_path'])

        if 'pts_instance_mask_path' in info:
            info['pts_instance_mask_path'] = \
                osp.join(self.data_prefix.get('pts_instance_mask', ''),
                         info['pts_instance_mask_path'])

        if 'pts_semantic_mask_path' in info:
            info['pts_semantic_mask_path'] = \
                osp.join(self.data_prefix.get('pts_semantic_mask', ''),
                         info['pts_semantic_mask_path'])

        # only be used in `PointSegClassMapping` in pipeline
        # to map original semantic class to valid category ids.
        info['seg_label_mapping'] = self.seg_label_mapping

        # 'eval_ann_info' will be updated in loading transforms
        if self.test_mode and self.load_eval_anns:
            info['eval_ann_info'] = dict()

        return info

    def prepare_data(self, idx: int) -> dict:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            dict: Results passed through ``self.pipeline``.
        """
        if not self.test_mode:
            data_info = self.get_data_info(idx)
            # Pass the dataset to the pipeline during training to support mixed
            # data augmentation, such as polarmix and lasermix.
            data_info['dataset'] = self
            return self.pipeline(data_info)
        else:
            return super().prepare_data(idx)

    def get_scene_idxs(self, scene_idxs: Union[None, str,
                                               np.ndarray]) -> np.ndarray:
        """Compute scene_idxs for data sampling.

        We sample more times for scenes with more points.
        """
        if self.test_mode:
            # when testing, we load one whole scene every time
            return np.arange(len(self)).astype(np.int32)

        # we may need to re-sample different scenes according to scene_idxs
        # this is necessary for indoor scene segmentation such as ScanNet
        if scene_idxs is None:
            scene_idxs = np.arange(len(self))
        if isinstance(scene_idxs, str):
            scene_idxs = osp.join(self.data_root, scene_idxs)
            with get_local_path(
                    scene_idxs, backend_args=self.backend_args) as local_path:
                scene_idxs = np.load(local_path)
        else:
            scene_idxs = np.array(scene_idxs)

        return scene_idxs.astype(np.int32)

    def _set_group_flag(self) -> None:
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
