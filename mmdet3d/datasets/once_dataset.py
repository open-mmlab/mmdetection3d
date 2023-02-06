# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import tempfile
from os import path as osp

import mmcv
import numpy as np
import torch
from mmcv.utils import print_log

from ..core import show_multi_modality_result, show_result
from ..core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                         LiDARInstance3DBoxes, points_cam2img)
from .builder import DATASETS
from .custom_3d import Custom3DDataset


@DATASETS.register_module()
class OnceDataset(Custom3DDataset):
    r"""ONCE Dataset.

    This class serves as the API for experiments on the `ONCE Dataset
    <https://once-for-auto-driving.github.io/download>`_.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        split (str): Split of input data.
        pts_prefix (str, optional): Prefix of points files.
            Defaults to 'velodyne'.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
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
        pcd_limit_range (list, optional): The range of point cloud used to
            filter invalid predicted boxes.
            Default: [0, -40, -3, 70.4, 40, 0.0].
    """
    CLASSES = ('Car', 'Bus', 'Truck', 'Pedestrian', 'Cyclist')

    def __init__(self,
                 data_root,
                 ann_file,
                 split,
                 pts_prefix='velodyne',
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 pcd_limit_range=[0, -40, -3, 70.4, 40, 0.0],
                 **kwargs):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            classes=classes,
            pipeline=pipeline,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)

        self.split = split
        self.root_split = os.path.join(self.data_root, split)
        assert self.modality is not None
        self.pcd_limit_range = pcd_limit_range
        self.pts_prefix = pts_prefix

        self.camera_list = ['cam01', 'cam03', 'cam05', 'cam06', 'cam07', 'cam08', 'cam09']
        self.data_infos = list(filter(self._check_annos, self.data_infos))
    
    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of filtered data infos.
        """
        return len(self.data_infos)
    
    def _check_annos(self, info):
        return 'annos' in info

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - img_prefix (str): Prefix of image files.
                - img_filename (str, optional): image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        sample_idx = info['frame_id']
        pts_filename = info['lidar_path']
        
        img_filenames = []
        lidar2imgs = []
        seq_id = info['sequence_id']
        for camera in self.camera_list:
            img_filename = os.path.join(self.data_root, 'data', \
                                        seq_id, camera, f'{sample_idx}.jpg')
            img_filenames.append(img_filename)
            # obtain lidar to image transformation matrix
            cam2lidar = info['calib'][camera]['cam_to_velo']
            lidar2cam = np.linalg.inv(cam2lidar)
            intrinsic = info['calib'][camera]['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:3, :3] = intrinsic
            lidar2img = viewpad @ lidar2cam.T
            lidar2imgs.append(lidar2img)

        input_dict = dict(
            sample_idx=sample_idx,
            pts_filename=pts_filename,
            img_prefix=None,
            img_filename=img_filenames,
            lidar2img=lidar2imgs,
        )

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_bboxes (np.ndarray): 2D ground truth bboxes.
                - gt_labels (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]
        annos = info['annos']

        gt_bboxes_3d = annos['boxes_3d']
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        gt_names = annos['name']
        gt_labels = []
        for cat in gt_names:
            if cat in self.CLASSES:
                gt_labels.append(self.CLASSES.index(cat))
            else:
                gt_labels.append(-1)
        gt_labels = np.array(gt_labels).astype(np.int64)
        gt_labels_3d = copy.deepcopy(gt_labels)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_labels=gt_labels,
            gt_names=gt_names,
        )
        return anns_results

    def _format_results(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
                - boxes_3d (:obj:`LiDARInstance3DBoxes`): Detection bbox.
                - scores_3d (torch.Tensor): Detection scores.
                - labels_3d (torch.Tensor): Predicted box labels.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            list[dict]: A list of dictionaries with the once format
            str: Path of the output json file.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        annos = []
        print('\nConverting prediction to ONCE format')
        for idx, result in enumerate(
                mmcv.track_iter_progress(results)):
            info = self.data_infos[idx]
            sample_idx = info['frame_id']
            pred_scores = result['scores_3d'].numpy()
            pred_labels = result['labels_3d'].numpy()
            pred_boxes = result['boxes_3d'].tensor.numpy()

            num_samples = pred_scores.shape[0]
            pred_dict = {
                'name': np.zeros(num_samples),
                'score': np.zeros(num_samples),
                'boxes_3d': np.zeros((num_samples, 7))
            }
            if num_samples != 0:
                pred_dict['name'] = np.array(self.CLASSES)[pred_labels]
                pred_dict['score'] = pred_scores
                # the predicted box center is [0.5, 0.5, 0], we change it to be
                # the same as OCNE (0.5, 0.5, 0.5)
                pred_boxes[:, 2] += pred_boxes[:, 5] / 2
                pred_dict['boxes_3d'] = pred_boxes

            pred_dict['frame_id'] = sample_idx
            annos.append(pred_dict)

        res_path = None
        if jsonfile_prefix is not None:
            mmcv.mkdir_or_exist(jsonfile_prefix)
            res_path = osp.join(jsonfile_prefix, 'results_once.json')
            print('Results writes to', res_path)
            mmcv.dump(annos, res_path)
        return annos, res_path

    def evaluate(self,
                 results,
                 eval_mode='Overall&Distance',
                 logger=None,
                 jsonfile_prefix=None,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in ONCE protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            eval_mode (str, optional): Mode to eval.
                Default: 'Overall&Distance'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str, optional): The prefix of json files, including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        results_list, tmp_dir = self._format_results(results, jsonfile_prefix)

        from mmdet3d.core.evaluation import once_eval
        gt_annos = [info['annos'] for info in self.data_infos]

        ap_result_str, ap_dict = once_eval(gt_annos, results_list, self.CLASSES)
        print_log('\n' + ap_result_str, logger=logger)
        
        if tmp_dir is not None:
            tmp_dir.cleanup()

        return ap_dict