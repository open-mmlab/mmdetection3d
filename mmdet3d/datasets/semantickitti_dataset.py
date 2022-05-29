# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp

import numpy as np
from mmcv.utils import print_log

from mmdet3d.core.evaluation import SemanticKITTIEval
from .builder import DATASETS
from .custom_3d import Custom3DDataset


@DATASETS.register_module()
class SemanticKITTIDataset(Custom3DDataset):
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
    CLASSES = ('unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'bus',
               'person', 'bicyclist', 'motorcyclist', 'road', 'parking',
               'sidewalk', 'other-ground', 'building', 'fence', 'vegetation',
               'trunck', 'terrian', 'pole', 'traffic-sign')
    labels_map = {
        0: 0,  # "unlabeled"
        1: 0,  # "outlier" mapped to "unlabeled" --------------mapped
        10: 1,  # "car"
        11: 2,  # "bicycle"
        13: 5,  # "bus" mapped to "other-vehicle" --------------mapped
        15: 3,  # "motorcycle"
        16: 5,  # "on-rails" mapped to "other-vehicle" ---------mapped
        18: 4,  # "truck"
        20: 5,  # "other-vehicle"
        30: 6,  # "person"
        31: 7,  # "bicyclist"
        32: 8,  # "motorcyclist"
        40: 9,  # "road"
        44: 10,  # "parking"
        48: 11,  # "sidewalk"
        49: 12,  # "other-ground"
        50: 13,  # "building"
        51: 14,  # "fence"
        52: 0,  # "other-structure" mapped to "unlabeled" ------mapped
        60: 9,  # "lane-marking" to "road" ---------------------mapped
        70: 15,  # "vegetation"
        71: 16,  # "trunk"
        72: 17,  # "terrain"
        80: 18,  # "pole"
        81: 19,  # "traffic-sign"
        99: 0,  # "other-object" to "unlabeled" ----------------mapped
        252: 1,  # "moving-car" to "car" ------------------------mapped
        253: 7,  # "moving-bicyclist" to "bicyclist" ------------mapped
        254: 6,  # "moving-person" to "person" ------------------mapped
        255: 8,  # "moving-motorcyclist" to "motorcyclist" ------mapped
        256: 5,  # "moving-on-rails" mapped to "other-vehic------mapped
        257: 5,  # "moving-bus" mapped to "other-vehicle" -------mapped
        258: 4,  # "moving-truck" to "truck" --------------------mapped
        259: 5  # "moving-other"-vehicle to "other-vehicle"-----mapped
    }

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 offset=2**32,
                 min_points=30,
                 ignore=[0],
                 modality=None,
                 box_type_3d='Lidar',
                 filter_empty_gt=False,
                 test_mode=False):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)

        self.init_eval = SemanticKITTIEval(classes, offset, min_points, ignore)

    def get_data_info(self, index):
        """Get data info according to the given index.
        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:
                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - file_name (str): Filename of point clouds.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        sample_idx = info['point_cloud']['lidar_idx']
        pts_filename = osp.join(self.data_root, info['pts_path'])

        input_dict = dict(
            pts_filename=pts_filename,
            sample_idx=sample_idx,
            file_name=pts_filename)

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if self.filter_empty_gt and ~(annos['gt_labels_3d'] != -1).any():
                return None
        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - pts_semantic_mask_path (str): Path of semantic masks.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]

        pts_semantic_mask_path = osp.join(self.data_root,
                                          info['pts_semantic_mask_path'])

        anns_results = dict(pts_semantic_mask_path=pts_semantic_mask_path)
        return anns_results

    def parse_label(self, filename):
        class_remap = self.labels_map
        gt_annos = []
        for i, seq in enumerate(filename):

            gt = np.fromfile(
                osp.join(self.data_root, seq), dtype=np.int32).reshape(-1, 1)
            gt_sem = gt & 0xFFFF
            gt_sem = np.vectorize(class_remap.__getitem__)(gt_sem)
            gt_annos.append(dict(gt_sem=gt_sem, gt_inst=gt))

        return gt_annos

    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in KITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str, optional): The prefix of pkl files, including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str, optional): The prefix of submission data.
                If not specified, the submission data will not be generated.
                Default: None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_files, tmp_dir = self.format_results(results, pklfile_prefix)

        # get the labels
        gt_files = [info['pts_semantic_mask_path'] for info in self.data_infos]
        gt_annos = self.parse_label(gt_files)

        if isinstance(result_files, dict):
            ap_dict_list = []
            ap_dict = dict()
            for name, result_files_ in result_files.items():

                ap_result_str, ap_dict_ = self.init_eval.evaluate(
                    gt_annos, result_files_, self.CLASSES)
                for ap_type, ap in ap_dict_.items():
                    ap_dict[f'{name}/{ap_type}'] = float('{:.4f}'.format(ap))
                ap_dict_list.append(ap_dict)
                print_log(
                    f'Results of {name}:\n' + ap_result_str, logger=logger)

        else:
            ap_result_str, ap_dict_list = self.init_eval.evaluate(
                gt_annos, result_files)
            print_log('\n' + ap_result_str, logger=logger)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)
        return ap_dict_list

    def show(self, results, out_dir, show=False, pipeline=None):
        pass
