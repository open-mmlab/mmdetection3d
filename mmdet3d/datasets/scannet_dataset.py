# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
import warnings
from os import path as osp

import numpy as np

from mmdet3d.core import instance_seg_eval, show_result, show_seg_result
from mmdet3d.core.bbox import DepthInstance3DBoxes
from mmseg.datasets import DATASETS as SEG_DATASETS
from .builder import DATASETS
from .custom_3d import Custom3DDataset
from .custom_3d_seg import Custom3DSegDataset
from .pipelines import Compose


@DATASETS.register_module()
class ScanNetDataset(Custom3DDataset):
    r"""ScanNet Dataset for Detection Task.

    This class serves as the API for experiments on the ScanNet Dataset.

    Please refer to the `github repo <https://github.com/ScanNet/ScanNet>`_
    for data downloading.

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
    CLASSES = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
               'garbagebin')

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 modality=dict(use_camera=False, use_depth=True),
                 box_type_3d='Depth',
                 filter_empty_gt=True,
                 test_mode=False,
                 **kwargs):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)
        assert 'use_camera' in self.modality and \
               'use_depth' in self.modality
        assert self.modality['use_camera'] or self.modality['use_depth']

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
                - img_prefix (str, optional): Prefix of image files.
                - img_info (dict, optional): Image info.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        sample_idx = info['point_cloud']['lidar_idx']
        pts_filename = osp.join(self.data_root, info['pts_path'])
        input_dict = dict(sample_idx=sample_idx)

        if self.modality['use_depth']:
            input_dict['pts_filename'] = pts_filename
            input_dict['file_name'] = pts_filename

        if self.modality['use_camera']:
            img_info = []
            for img_path in info['img_paths']:
                img_info.append(
                    dict(filename=osp.join(self.data_root, img_path)))
            intrinsic = info['intrinsics']
            axis_align_matrix = self._get_axis_align_matrix(info)
            depth2img = []
            for extrinsic in info['extrinsics']:
                depth2img.append(
                    intrinsic @ np.linalg.inv(axis_align_matrix @ extrinsic))

            input_dict['img_prefix'] = None
            input_dict['img_info'] = img_info
            input_dict['depth2img'] = depth2img

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

                - gt_bboxes_3d (:obj:`DepthInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - pts_instance_mask_path (str): Path of instance masks.
                - pts_semantic_mask_path (str): Path of semantic masks.
                - axis_align_matrix (np.ndarray): Transformation matrix for
                    global scene alignment.
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

        axis_align_matrix = self._get_axis_align_matrix(info)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            pts_instance_mask_path=pts_instance_mask_path,
            pts_semantic_mask_path=pts_semantic_mask_path,
            axis_align_matrix=axis_align_matrix)
        return anns_results

    def prepare_test_data(self, index):
        """Prepare data for testing.

        We should take axis_align_matrix from self.data_infos since we need
            to align point clouds.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        # take the axis_align_matrix from data_infos
        input_dict['ann_info'] = dict(
            axis_align_matrix=self._get_axis_align_matrix(
                self.data_infos[index]))
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    @staticmethod
    def _get_axis_align_matrix(info):
        """Get axis_align_matrix from info. If not exist, return identity mat.

        Args:
            info (dict): one data info term.

        Returns:
            np.ndarray: 4x4 transformation matrix.
        """
        if 'axis_align_matrix' in info['annos'].keys():
            return info['annos']['axis_align_matrix'].astype(np.float32)
        else:
            warnings.warn(
                'axis_align_matrix is not found in ScanNet data info, please '
                'use new pre-process scripts to re-generate ScanNet data')
            return np.eye(4).astype(np.float32)

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                load_dim=6,
                use_dim=[0, 1, 2]),
            dict(type='GlobalAlignment', rotation_axis=2),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        return Compose(pipeline)

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            data_info = self.data_infos[i]
            pts_path = data_info['pts_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points').numpy()
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            pred_bboxes = result['boxes_3d'].tensor.numpy()
            show_result(points, gt_bboxes, pred_bboxes, out_dir, file_name,
                        show)


@DATASETS.register_module()
@SEG_DATASETS.register_module()
class ScanNetSegDataset(Custom3DSegDataset):
    r"""ScanNet Dataset for Semantic Segmentation Task.

    This class serves as the API for experiments on the ScanNet Dataset.

    Please refer to the `github repo <https://github.com/ScanNet/ScanNet>`_
    for data downloading.

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
    CLASSES = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
               'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
               'curtain', 'refrigerator', 'showercurtrain', 'toilet', 'sink',
               'bathtub', 'otherfurniture')

    VALID_CLASS_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28,
                       33, 34, 36, 39)

    ALL_CLASS_IDS = tuple(range(41))

    PALETTE = [
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
    ]

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 palette=None,
                 modality=None,
                 test_mode=False,
                 ignore_index=None,
                 scene_idxs=None,
                 **kwargs):

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            palette=palette,
            modality=modality,
            test_mode=test_mode,
            ignore_index=ignore_index,
            scene_idxs=scene_idxs,
            **kwargs)

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

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                use_color=True,
                load_dim=6,
                use_dim=[0, 1, 2, 3, 4, 5]),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=False,
                with_label_3d=False,
                with_mask_3d=False,
                with_seg_3d=True),
            dict(
                type='PointSegClassMapping',
                valid_cat_ids=self.VALID_CLASS_IDS,
                max_cat_id=np.max(self.ALL_CLASS_IDS)),
            dict(
                type='DefaultFormatBundle3D',
                with_label=False,
                class_names=self.CLASSES),
            dict(type='Collect3D', keys=['points', 'pts_semantic_mask'])
        ]
        return Compose(pipeline)

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            data_info = self.data_infos[i]
            pts_path = data_info['pts_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points, gt_sem_mask = self._extract_data(
                i, pipeline, ['points', 'pts_semantic_mask'], load_annos=True)
            points = points.numpy()
            pred_sem_mask = result['semantic_mask'].numpy()
            show_seg_result(points, gt_sem_mask,
                            pred_sem_mask, out_dir, file_name,
                            np.array(self.PALETTE), self.ignore_index, show)

    def get_scene_idxs(self, scene_idxs):
        """Compute scene_idxs for data sampling.

        We sample more times for scenes with more points.
        """
        # when testing, we load one whole scene every time
        if not self.test_mode and scene_idxs is None:
            raise NotImplementedError(
                'please provide re-sampled scene indexes for training')

        return super().get_scene_idxs(scene_idxs)

    def format_results(self, results, txtfile_prefix=None):
        r"""Format the results to txt file. Refer to `ScanNet documentation
        <http://kaldir.vc.in.tum.de/scannet_benchmark/documentation>`_.

        Args:
            outputs (list[dict]): Testing results of the dataset.
            txtfile_prefix (str): The prefix of saved files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (outputs, tmp_dir), outputs is the detection results,
                tmp_dir is the temporal directory created for saving submission
                files when ``submission_prefix`` is not specified.
        """
        import mmcv

        if txtfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            txtfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        mmcv.mkdir_or_exist(txtfile_prefix)

        # need to map network output to original label idx
        pred2label = np.zeros(len(self.VALID_CLASS_IDS)).astype(np.int)
        for original_label, output_idx in self.label_map.items():
            if output_idx != self.ignore_index:
                pred2label[output_idx] = original_label

        outputs = []
        for i, result in enumerate(results):
            info = self.data_infos[i]
            sample_idx = info['point_cloud']['lidar_idx']
            pred_sem_mask = result['semantic_mask'].numpy().astype(np.int)
            pred_label = pred2label[pred_sem_mask]
            curr_file = f'{txtfile_prefix}/{sample_idx}.txt'
            np.savetxt(curr_file, pred_label, fmt='%d')
            outputs.append(dict(seg_mask=pred_label))

        return outputs, tmp_dir


@DATASETS.register_module()
@SEG_DATASETS.register_module()
class ScanNetInstanceSegDataset(Custom3DSegDataset):
    CLASSES = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
               'garbagebin')

    VALID_CLASS_IDS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34,
                       36, 39)

    ALL_CLASS_IDS = tuple(range(41))

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:
                - pts_semantic_mask_path (str): Path of semantic masks.
                - pts_instance_mask_path (str): Path of instance masks.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]

        pts_instance_mask_path = osp.join(self.data_root,
                                          info['pts_instance_mask_path'])
        pts_semantic_mask_path = osp.join(self.data_root,
                                          info['pts_semantic_mask_path'])

        anns_results = dict(
            pts_instance_mask_path=pts_instance_mask_path,
            pts_semantic_mask_path=pts_semantic_mask_path)
        return anns_results

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset. Palette is simply ignored for
        instance segmentation.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
                Defaults to None.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Defaults to None.
        """
        if classes is not None:
            return classes, None
        return self.CLASSES, None

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                use_color=True,
                load_dim=6,
                use_dim=[0, 1, 2, 3, 4, 5]),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=False,
                with_label_3d=False,
                with_mask_3d=True,
                with_seg_3d=True),
            dict(
                type='PointSegClassMapping',
                valid_cat_ids=self.VALID_CLASS_IDS,
                max_cat_id=40),
            dict(
                type='DefaultFormatBundle3D',
                with_label=False,
                class_names=self.CLASSES),
            dict(
                type='Collect3D',
                keys=['points', 'pts_semantic_mask', 'pts_instance_mask'])
        ]
        return Compose(pipeline)

    def evaluate(self,
                 results,
                 metric=None,
                 options=None,
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in instance segmentation protocol.

        Args:
            results (list[dict]): List of results.
            metric (str | list[str]): Metrics to be evaluated.
            options (dict, optional): options for instance_seg_eval.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Defaults to None.
            show (bool, optional): Whether to visualize.
                Defaults to False.
            out_dir (str, optional): Path to save the visualization results.
                Defaults to None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict: Evaluation results.
        """
        assert isinstance(
            results, list), f'Expect results to be list, got {type(results)}.'
        assert len(results) > 0, 'Expect length of results > 0.'
        assert len(results) == len(self.data_infos)
        assert isinstance(
            results[0], dict
        ), f'Expect elements in results to be dict, got {type(results[0])}.'

        load_pipeline = self._get_pipeline(pipeline)
        pred_instance_masks = [result['instance_mask'] for result in results]
        pred_instance_labels = [result['instance_label'] for result in results]
        pred_instance_scores = [result['instance_score'] for result in results]
        gt_semantic_masks, gt_instance_masks = zip(*[
            self._extract_data(
                index=i,
                pipeline=load_pipeline,
                key=['pts_semantic_mask', 'pts_instance_mask'],
                load_annos=True) for i in range(len(self.data_infos))
        ])
        ret_dict = instance_seg_eval(
            gt_semantic_masks,
            gt_instance_masks,
            pred_instance_masks,
            pred_instance_labels,
            pred_instance_scores,
            valid_class_ids=self.VALID_CLASS_IDS,
            class_labels=self.CLASSES,
            options=options,
            logger=logger)

        if show:
            raise NotImplementedError('show is not implemented for now')

        return ret_dict
