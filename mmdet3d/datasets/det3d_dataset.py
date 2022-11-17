# Copyright (c) OpenMMLab. All rights reserved.
import copy
from os import path as osp
from typing import Callable, Dict, List, Optional, Sequence, Set, Union

import mmengine
import numpy as np
import torch
from mmengine.dataset import BaseDataset
from mmengine.logging import print_log
from terminaltables import AsciiTable

from mmdet3d.datasets import DATASETS
from mmdet3d.structures import get_box_type


@DATASETS.register_module()
class Det3DDataset(BaseDataset):
    """Base Class of 3D dataset.

    This is the base dataset of SUNRGB-D, ScanNet, nuScenes, and KITTI
    dataset.
    # TODO: doc link here for the standard data format

    Args:
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to None.
        ann_file (str): Annotation file path. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_prefix (dict): Prefix for training data. Defaults to
            dict(pts='velodyne', img='').
        pipeline (list[dict]): Pipeline used for data processing.
            Defaults to [].
        modality (dict): Modality to specify the sensor data used as input,
            it usually has following keys:

                - use_camera: bool
                - use_lidar: bool
            Defaults to `dict(use_lidar=True, use_camera=False)`
        point_cloud_range (list[float]): The range of point cloud used to
            filter points and 3D bboxes. Defaults to None.
        default_cam_key (str, optional): The default camera name adopted.
            Defaults to None.
        box_type_3d (str): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes:

            - 'LiDAR': Box in LiDAR coordinates, usually for
              outdoor point cloud 3d detection.
            - 'Depth': Box in depth coordinates, usually for
              indoor point cloud 3d detection.
            - 'Camera': Box in camera coordinates, usually
              for vision-based 3d detection.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
        load_eval_anns (bool): Whether to load annotations in test_mode,
            the annotation will be save in `eval_ann_infos`, which can be
            used in Evaluator. Defaults to True.
        file_client_args (dict): Configuration of file client.
            Defaults to dict(backend='disk').
        merge_cfg (dict, optional): Config for merge dataset classes.
            Defaults to None.
            A typical merge config should be like:
                If we want to merge classes `Truck` and `Van` into
                `Car`, it should be:
                dict(class_merge=dict(Car=('Truck','Van')))
        show_ins_var (bool): For debug purpose. Whether to show variation
            of the number of instances before and after through pipeline.
            Defaults to False.
    """

    def __init__(self,
                 data_root: Optional[str] = None,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(pts='velodyne', img=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True, use_camera=False),
                 point_cloud_range: List[float] = None,
                 default_cam_key: str = None,
                 box_type_3d: dict = 'LiDAR',
                 test_mode: bool = False,
                 load_eval_anns=True,
                 file_client_args: dict = dict(backend='disk'),
                 merge_cfg: Optional[dict] = None,
                 show_ins_var: bool = False,
                 **kwargs) -> None:
        # init file client
        self.file_client = mmengine.FileClient(**file_client_args)
        self.load_eval_anns = load_eval_anns
        self.merge_cfg = merge_cfg
        _default_modality_keys = ('use_lidar', 'use_camera')
        if modality is None:
            modality = dict()

        # Defaults to False if not specify
        for key in _default_modality_keys:
            if key not in modality:
                modality[key] = False
        self.modality = modality
        self.point_cloud_range = point_cloud_range
        self.default_cam_key = default_cam_key
        assert self.modality['use_lidar'] or self.modality['use_camera'], (
            'Please specify the `modality` (`use_lidar` '
            f', `use_camera`) for {self.__class__.__name__}')

        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)

        new_classes = metainfo.get('classes', None)
        self.label_mapping, self.num_ins_per_cat = self.get_label_mapping(
            new_classes)

        if self.merge_cfg is not None:
            self.merge_mapping = self.parse_merge_cfg()

        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwargs)

        # can be accessed by other component in runner
        self.metainfo['box_type_3d'] = box_type_3d
        self.metainfo['label_mapping'] = self.label_mapping

        # used for showing variation of the number of instances before and
        # after through the pipeline
        self.show_ins_var = show_ins_var

        # show statistics of this dataset
        print_log('-' * 30, 'current')
        print_log(f'The length of the dataset: {len(self)}', 'current')
        content_show = [['category', 'number']]
        for cat_name, num in self.num_ins_per_cat.items():
            content_show.append([cat_name, num])
        table = AsciiTable(content_show)
        print_log(
            f'The number of instances per category in the dataset:\n{table.table}',  # noqa: E501
            'current')

    def get_label_mapping(self,
                          new_classes: Optional[Sequence] = None
                          ) -> Union[Dict, None]:
        """Get label mapping.

        The ``label_mapping`` is a dictionary, its keys are the old label
        ids and its values are the new label ids.

        Args:
            new_classes (list, tuple, optional): The new classes name from
                metainfo. Default to None.

        Returns:
            tuple: The mapping from old classes in cls.METAINFO to
            new classes in metainfo
        """
        # we allow to train on subset of self.METAINFO['classes']
        # map unselected labels to -1
        old_classes = self.METAINFO.get('classes', None)
        if (new_classes is not None and old_classes is not None
                and list(new_classes) != list(old_classes)):
            if not set(new_classes).issubset(old_classes):
                raise ValueError(
                    f'new classes {new_classes} is not a '
                    f'subset of classes {old_classes} in METAINFO.')

            label_mapping = {i: -1 for i in range(len(old_classes))}

            for label_idx, name in enumerate(new_classes):
                ori_label = self.METAINFO['classes'].index(name)
                label_mapping[ori_label] = label_idx

            num_ins_per_cat = {name: 0 for name in new_classes}
        else:
            label_mapping = {i: i for i in range(len(old_classes))}
            num_ins_per_cat = {name: 0 for name in old_classes}

        return label_mapping, num_ins_per_cat

    def get_ann_info(self, index: int) -> dict:
        """Get annotation info according to the given index.

        Use index to get the corresponding annotations, thus the
        evalhook could use this api.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information.
        """
        data_info = self.get_data_info(index)
        # test model
        if 'ann_info' not in data_info:
            ann_info = self.parse_ann_info(data_info)
        else:
            ann_info = data_info['ann_info']

        return ann_info

    def parse_merge_cfg(self) -> dict:
        """Parse `self.merge_cfg`.

        Sometimes we want to merge certain classes into target class
        to increase ground truth number and improve training performance.

        Returns:
            dict: Processed `merge_mapping`
        """

        if not isinstance(self.merge_cfg, dict):
            raise TypeError(
                f'merge_cfg should be a dict, but got {type(self.merge_cfg)}')
        merge_mapping = dict()
        for merge_name, names in self.merge_cfg['class_merge'].items():
            merge_label = self.METAINFO['classes'].index(merge_name)
            if isinstance(names, (list, tuple)):
                for name in names:
                    ori_label = self.METAINFO['classes'].index(name)
                    merge_mapping[ori_label] = merge_label
            elif isinstance(names, str):
                ori_label = self.METAINFO['classes'].index(names)
                merge_mapping[ori_label] = merge_label
            else:
                raise TypeError(
                    f'class names to be merged should be a list, tuple'
                    f'or str, but got {type(names)}')

        return merge_mapping

    def parse_ann_info(self, info: dict) -> Optional[dict]:
        """Process the `instances` in data info to `ann_info`.

        In `Custom3DDataset`, we simply concatenate all the field
        in `instances` to `np.ndarray`, you can do the specific
        process in subclass. You have to convert `gt_bboxes_3d`
        to different coordinates according to the task.

        Args:
            info (dict): Info dict.

        Returns:
            dict | None: Processed `ann_info`
        """
        # add s or gt prefix for most keys after concat
        # we only process 3d annotations here, the corresponding
        # 2d annotation process is in the `LoadAnnotations3D`
        # in `transforms`
        name_mapping = {
            'bbox_label_3d': 'gt_labels_3d',
            'bbox_label': 'gt_bboxes_labels',
            'bbox': 'gt_bboxes',
            'bbox_3d': 'gt_bboxes_3d',
            'depth': 'depths',
            'center_2d': 'centers_2d',
            'attr_label': 'attr_labels',
            'velocity': 'velocities',
        }
        instances = info['instances']
        # empty gt
        if len(instances) == 0:
            return None
        else:
            keys = list(instances[0].keys())
            ann_info = dict()
            for ann_name in keys:
                temp_anns = [item[ann_name] for item in instances]
                # map the original dataset label to training label
                if 'label' in ann_name and ann_name != 'attr_label':
                    # if the merge_cfg is not None, we need to merge
                    # specified dataset labels to the target dataset
                    # label before mapping the original dataset label
                    # to training label
                    if self.merge_cfg is not None:
                        merge_mask = [
                            item in self.merge_mapping for item in temp_anns
                        ]
                        ann_info['merge_mask'] = np.array(merge_mask)

                        temp_anns = [
                            self.merge_mapping[item]
                            if item in self.merge_mapping else item
                            for item in temp_anns
                        ]

                    temp_anns = [
                        self.label_mapping[item] for item in temp_anns
                    ]
                if ann_name in name_mapping:
                    mapped_ann_name = name_mapping[ann_name]
                else:
                    mapped_ann_name = ann_name

                if 'label' in ann_name:
                    temp_anns = np.array(temp_anns).astype(np.int64)
                elif ann_name in name_mapping:
                    temp_anns = np.array(temp_anns).astype(np.float32)
                else:
                    temp_anns = np.array(temp_anns)

                ann_info[mapped_ann_name] = temp_anns
            ann_info['instances'] = info['instances']

            for label in ann_info['gt_labels_3d']:
                if label != -1:
                    cat_name = self.metainfo['classes'][label]
                    self.num_ins_per_cat[cat_name] += 1

        return ann_info

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        Convert all relative path of needed modality data file to
        the absolute path. And process the `instances` field to
        `ann_info` in training stage.

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

            info['num_pts_feats'] = info['lidar_points']['num_pts_feats']
            info['lidar_path'] = info['lidar_points']['lidar_path']
            if self.point_cloud_range is not None:
                info['point_cloud_range'] = self.point_cloud_range

            if 'lidar_sweeps' in info:
                for sweep in info['lidar_sweeps']:
                    file_suffix = sweep['lidar_points']['lidar_path'].split(
                        '/')[-1]
                    if 'samples' in sweep['lidar_points']['lidar_path']:
                        sweep['lidar_points']['lidar_path'] = osp.join(
                            self.data_prefix['pts'], file_suffix)
                    else:
                        sweep['lidar_points']['lidar_path'] = osp.join(
                            self.data_prefix['sweeps'], file_suffix)

        if self.modality['use_camera']:
            for cam_id, img_info in info['images'].items():
                if 'img_path' in img_info:
                    if cam_id in self.data_prefix:
                        cam_prefix = self.data_prefix[cam_id]
                    else:
                        cam_prefix = self.data_prefix.get('img', '')
                    img_info['img_path'] = osp.join(cam_prefix,
                                                    img_info['img_path'])
            if self.default_cam_key is not None:
                info['img_path'] = info['images'][
                    self.default_cam_key]['img_path']
                if 'lidar2cam' in info['images'][self.default_cam_key]:
                    info['lidar2cam'] = np.array(
                        info['images'][self.default_cam_key]['lidar2cam'])
                if 'cam2img' in info['images'][self.default_cam_key]:
                    info['cam2img'] = np.array(
                        info['images'][self.default_cam_key]['cam2img'])
                if 'lidar2img' in info['images'][self.default_cam_key]:
                    info['lidar2img'] = np.array(
                        info['images'][self.default_cam_key]['lidar2img'])
                else:
                    info['lidar2img'] = info['cam2img'] @ info['lidar2cam']

        if not self.test_mode:
            # used in training
            info['ann_info'] = self.parse_ann_info(info)
        if self.test_mode and self.load_eval_anns:
            info['eval_ann_info'] = self.parse_ann_info(info)

        return info

    def _show_ins_var(self, old_labels: np.ndarray, new_labels: torch.Tensor):
        """Show variation of the number of instances before and after through
        the pipeline.

        Args:
            old_labels (np.ndarray): The labels before through the pipeline.
            new_labels (torch.Tensor): The labels after through the pipeline.
        """
        ori_num_per_cat = dict()
        for label in old_labels:
            if label != -1:
                cat_name = self.metainfo['classes'][label]
                ori_num_per_cat[cat_name] = ori_num_per_cat.get(cat_name,
                                                                0) + 1
        new_num_per_cat = dict()
        for label in new_labels:
            if label != -1:
                cat_name = self.metainfo['classes'][label]
                new_num_per_cat[cat_name] = new_num_per_cat.get(cat_name,
                                                                0) + 1
        content_show = [['category', 'new number', 'ori number']]
        for cat_name, num in ori_num_per_cat.items():
            new_num = new_num_per_cat.get(cat_name, 0)
            content_show.append([cat_name, new_num, num])
        table = AsciiTable(content_show)
        print_log(
            'The number of instances per category after and before '
            f'through pipeline:\n{table.table}', 'current')

    def prepare_data(self, index: int) -> Optional[dict]:
        """Data preparation for both training and testing stage.

        Called by `__getitem__`  of dataset.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict | None: Data dict of the corresponding index.
        """
        ori_input_dict = self.get_data_info(index)

        # deepcopy here to avoid inplace modification in pipeline.
        input_dict = copy.deepcopy(ori_input_dict)

        # box_type_3d (str): 3D box type.
        input_dict['box_type_3d'] = self.box_type_3d
        # box_mode_3d (str): 3D box mode.
        input_dict['box_mode_3d'] = self.box_mode_3d

        # pre-pipline return None to random another in `__getitem__`
        if not self.test_mode:
            if len(input_dict['ann_info']['gt_labels_3d']) == 0:
                return None

        example = self.pipeline(input_dict)

        if not self.test_mode:
            # after pipeline drop the example with empty annotations
            # return None to random another in `__getitem__`
            if example is None or len(
                    example['data_samples'].gt_instances_3d.labels_3d) == 0:
                return None

        if self.show_ins_var:
            if 'ann_info' in ori_input_dict:
                self._show_ins_var(
                    ori_input_dict['ann_info']['gt_labels_3d'],
                    example['data_samples'].gt_instances_3d.labels_3d)
            else:
                print_log(
                    "'ann_info' is not in the input dict. It's probably that "
                    'the data is not in training mode',
                    'current',
                    level=30)

        return example

    def get_cat_ids(self, idx: int) -> Set[int]:
        """Get category ids by index. Dataset wrapped by ClassBalancedDataset
        must implement this method.

        The ``CBGSDataset`` or ``ClassBalancedDataset``requires a subclass
        which implements this method.

        Args:
            idx (int): The index of data.

        Returns:
            set[int]: All categories in the sample of specified index.
        """
        info = self.get_data_info(idx)
        gt_labels = info['ann_info']['gt_labels_3d'].tolist()
        return set(gt_labels)
