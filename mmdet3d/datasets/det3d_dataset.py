# Copyright (c) OpenMMLab. All rights reserved.
import copy
from os import path as osp
from typing import Callable, List, Optional, Union

import mmengine
import numpy as np
from mmengine.dataset import BaseDataset

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
        data_prefix (dict, optional): Prefix for training data. Defaults to
            dict(pts='velodyne', img="").
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input, it usually has following keys.

                - use_camera: bool
                - use_lidar: bool
            Defaults to `dict(use_lidar=True, use_camera=False)`
        default_cam_key (str, optional): The default camera name adopted.
            Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR'. Available options includes

            - 'LiDAR': Box in LiDAR coordinates, usually for
              outdoor point cloud 3d detection.
            - 'Depth': Box in depth coordinates, usually for
              indoor point cloud 3d detection.
            - 'Camera': Box in camera coordinates, usually
              for vision-based 3d detection.

        filter_empty_gt (bool): Whether to filter the data with
            empty GT. Defaults to True.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
        load_eval_anns (bool): Whether to load annotations
            in test_mode, the annotation will be save in
            `eval_ann_infos`, which can be use in Evaluator.
        file_client_args (dict): Configuration of file client.
            Defaults to `dict(backend='disk')`.
    """

    def __init__(self,
                 data_root: Optional[str] = None,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(pts='velodyne', img=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True, use_camera=False),
                 default_cam_key: str = None,
                 box_type_3d: dict = 'LiDAR',
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 load_eval_anns=True,
                 file_client_args: dict = dict(backend='disk'),
                 **kwargs):
        # init file client
        self.file_client = mmengine.FileClient(**file_client_args)
        self.filter_empty_gt = filter_empty_gt
        self.load_eval_anns = load_eval_anns
        _default_modality_keys = ('use_lidar', 'use_camera')
        if modality is None:
            modality = dict()

        # Defaults to False if not specify
        for key in _default_modality_keys:
            if key not in modality:
                modality[key] = False
        self.modality = modality
        self.default_cam_key = default_cam_key
        assert self.modality['use_lidar'] or self.modality['use_camera'], (
            'Please specify the `modality` (`use_lidar` '
            f', `use_camera`) for {self.__class__.__name__}')

        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)

        if metainfo is not None and 'CLASSES' in metainfo:
            # we allow to train on subset of self.METAINFO['CLASSES']
            # map unselected labels to -1
            self.label_mapping = {
                i: -1
                for i in range(len(self.METAINFO['CLASSES']))
            }
            self.label_mapping[-1] = -1
            for label_idx, name in enumerate(metainfo['CLASSES']):
                ori_label = self.METAINFO['CLASSES'].index(name)
                self.label_mapping[ori_label] = label_idx
        else:
            self.label_mapping = {
                i: i
                for i in range(len(self.METAINFO['CLASSES']))
            }
            self.label_mapping[-1] = -1

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

    def _remove_dontcare(self, ann_info):
        """Remove annotations that do not need to be cared.

        -1 indicate dontcare in MMDet3d.

        Args:
            ann_info (dict): Dict of annotation infos. The
                instance with label `-1` will be removed.

        Returns:
            dict: Annotations after filtering.
        """
        img_filtered_annotations = {}
        filter_mask = ann_info['gt_labels_3d'] > -1
        for key in ann_info.keys():
            if key != 'instances':
                img_filtered_annotations[key] = (ann_info[key][filter_mask])
            else:
                img_filtered_annotations[key] = ann_info[key]
        return img_filtered_annotations

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

    def parse_ann_info(self, info: dict) -> Optional[dict]:
        """Process the `instances` in data info to `ann_info`

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
            'attr_label': 'attr_labels'
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
                    temp_anns = [
                        self.label_mapping[item] for item in temp_anns
                    ]
                if ann_name in name_mapping:
                    ann_name = name_mapping[ann_name]

                if 'label' in ann_name:
                    temp_anns = np.array(temp_anns).astype(np.int64)
                else:
                    temp_anns = np.array(temp_anns).astype(np.float32)

                ann_info[ann_name] = temp_anns
            ann_info['instances'] = info['instances']
        return ann_info

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

            info['lidar_path'] = info['lidar_points']['lidar_path']
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

    def prepare_data(self, index):
        """Data preparation for both training and testing stage.

        Called by `__getitem__`  of dataset.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)

        # deepcopy here to avoid inplace modification in pipeline.
        input_dict = copy.deepcopy(input_dict)

        # box_type_3d (str): 3D box type.
        input_dict['box_type_3d'] = self.box_type_3d
        # box_mode_3d (str): 3D box mode.
        input_dict['box_mode_3d'] = self.box_mode_3d

        # pre-pipline return None to random another in `__getitem__`
        if not self.test_mode and self.filter_empty_gt:
            if len(input_dict['ann_info']['gt_labels_3d']) == 0:
                return None

        example = self.pipeline(input_dict)
        if not self.test_mode and self.filter_empty_gt:
            # after pipeline drop the example with empty annotations
            # return None to random another in `__getitem__`
            if example is None or len(
                    example['data_samples'].gt_instances_3d.labels_3d) == 0:
                return None
        return example

    def get_cat_ids(self, idx: int) -> List[int]:
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
