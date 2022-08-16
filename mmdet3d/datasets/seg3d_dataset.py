# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp
from typing import Callable, Dict, List, Optional, Sequence, Union

import mmengine
import numpy as np
from mmengine.dataset import BaseDataset

from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class Seg3DDataset(BaseDataset):
    """Base Class for 3D semantic segmentation dataset.

    This is the base dataset of ScanNet, S3DIS and SemanticKITTI dataset.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_prefix (dict, optional): Prefix for training data. Defaults to
            dict(pts='velodyne', img='', instance_mask='', semantic_mask='').
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input, it usually has following keys.

                - use_camera: bool
                - use_lidar: bool
            Defaults to `dict(use_lidar=True, use_camera=False)`
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        ignore_index (int, optional): The label index to be ignored, e.g.
            unannotated points. If None is given, set to len(self.CLASSES) to
            be consistent with PointSegClassMapping function in pipeline.
            Defaults to None.
        scene_idxs (np.ndarray | str, optional): Precomputed index to load
            data. For scenes with many points, we may sample it several times.
            Defaults to None.
        load_eval_anns (bool): Whether to load annotations
            in test_mode, the annotation will be save in
            `eval_ann_infos`, which can be use in Evaluator.
        file_client_args (dict): Configuration of file client.
            Defaults to `dict(backend='disk')`.
    """
    METAINFO = {
        'CLASSES': None,  # names of all classes data used for the task
        'PALETTE': None,  # official color for visualization
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
                     pts_emantic_mask=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True, use_camera=False),
                 ignore_index: Optional[int] = None,
                 scene_idxs: Optional[str] = None,
                 test_mode: bool = False,
                 load_eval_anns: bool = True,
                 file_client_args: dict = dict(backend='disk'),
                 **kwargs) -> None:
        # init file client
        self.file_client = mmengine.FileClient(**file_client_args)
        self.modality = modality
        self.load_eval_anns = load_eval_anns

        # TODO: We maintain the ignore_index attributes,
        # but we may consider to remove it in the future.
        self.ignore_index = len(self.METAINFO['CLASSES']) if \
            ignore_index is None else ignore_index

        # Get label mapping for custom classes
        new_classes = metainfo.get('CLASSES', None)

        self.label_mapping, self.label2cat, seg_valid_class_ids = \
            self.get_label_mapping(new_classes)

        metainfo['label_mapping'] = self.label_mapping
        metainfo['label2cat'] = self.label2cat
        metainfo['ignore_index'] = self.ignore_index
        metainfo['seg_valid_class_ids'] = seg_valid_class_ids

        # generate palette if it is not defined based on
        # label mapping, otherwise directly use palette
        # defined in dataset config.
        palette = metainfo.get('PALETTE', None)
        updated_palette = self._update_palette(new_classes, palette)

        metainfo['PALETTE'] = updated_palette

        # construct seg_label_mapping for semantic mask
        seg_max_cat_id = len(self.METAINFO['seg_all_class_ids'])
        seg_valid_cat_ids = self.METAINFO['seg_valid_class_ids']
        neg_label = len(seg_valid_cat_ids)
        seg_label_mapping = np.ones(
            seg_max_cat_id + 1, dtype=np.int) * neg_label
        for cls_idx, cat_id in enumerate(seg_valid_cat_ids):
            seg_label_mapping[cat_id] = cls_idx
        self.seg_label_mapping = seg_label_mapping

        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwargs)

        self.metainfo['seg_label_mapping'] = self.seg_label_mapping
        self.scene_idxs = self.get_scene_idxs(scene_idxs)

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

    def get_label_mapping(self,
                          new_classes: Optional[Sequence] = None
                          ) -> Union[Dict, None]:
        """Get label mapping.

        The ``label_mapping`` is a dictionary, its keys are the old label ids
        and its values are the new label ids, and is used for changing pixel
        labels in load_annotations. If and only if old classes in cls.METAINFO
        is not equal to new classes in self._metainfo and nether of them is not
        None, `label_mapping` is not None.

        Args:
            new_classes (list, tuple, optional): The new classes name from
                metainfo. Default to None.


        Returns:
            tuple: The mapping from old classes in cls.METAINFO to
                new classes in metainfo
        """
        old_classes = self.METAINFO.get('CLASSES', None)
        if (new_classes is not None and old_classes is not None
                and list(new_classes) != list(old_classes)):
            if not set(new_classes).issubset(old_classes):
                raise ValueError(
                    f'new classes {new_classes} is not a '
                    f'subset of CLASSES {old_classes} in METAINFO.')

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
                for i, cat_name in enumerate(self.METAINFO['CLASSES'])
            }
            valid_class_ids = self.METAINFO['seg_valid_class_ids']

        return label_mapping, label2cat, valid_class_ids

    def _update_palette(self, new_classes, palette) -> list:
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
            # to the original PALETTE and classes.
            old_classes = self.METAINFO.get('CLASSES', None)
            palette = [
                self.METAINFO['PALETTE'][old_classes.index(cls_name)]
                for cls_name in new_classes
            ]
            return palette

        # palette does match classes
        if len(palette) == len(new_classes):
            return palette
        else:
            raise ValueError('Once PLATTE in set in metainfo, it should'
                             'match CLASSES in metainfo')

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

    def get_scene_idxs(self, scene_idxs):
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
            with self.file_client.get_local_path(scene_idxs) as local_path:
                scene_idxs = np.load(local_path)
        else:
            scene_idxs = np.array(scene_idxs)

        return scene_idxs.astype(np.int32)

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
