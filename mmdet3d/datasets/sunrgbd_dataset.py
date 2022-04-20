# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from os import path as osp

import numpy as np

from mmdet3d.core import show_multi_modality_result, show_result
from mmdet3d.core.bbox import DepthInstance3DBoxes
from mmdet.core import eval_map
from .builder import DATASETS
from .custom_3d import Custom3DDataset
from .pipelines import Compose


@DATASETS.register_module()
class SUNRGBDDataset(Custom3DDataset):
    r"""SUNRGBD Dataset.

    This class serves as the API for experiments on the SUNRGBD Dataset.

    See the `download page <http://rgbd.cs.princeton.edu/challenge.html>`_
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
    CLASSES = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
               'night_stand', 'bookshelf', 'bathtub')

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 modality=dict(use_camera=True, use_lidar=True),
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
            'use_lidar' in self.modality
        assert self.modality['use_camera'] or self.modality['use_lidar']

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str, optional): Filename of point clouds.
                - file_name (str, optional): Filename of point clouds.
                - img_prefix (str, optional): Prefix of image files.
                - img_info (dict, optional): Image info.
                - calib (dict, optional): Camera calibration info.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        sample_idx = info['point_cloud']['lidar_idx']
        assert info['point_cloud']['lidar_idx'] == info['image']['image_idx']
        input_dict = dict(sample_idx=sample_idx)

        if self.modality['use_lidar']:
            pts_filename = osp.join(self.data_root, info['pts_path'])
            input_dict['pts_filename'] = pts_filename
            input_dict['file_name'] = pts_filename

        if self.modality['use_camera']:
            img_filename = osp.join(
                osp.join(self.data_root, 'sunrgbd_trainval'),
                info['image']['image_path'])
            input_dict['img_prefix'] = None
            input_dict['img_info'] = dict(filename=img_filename)
            calib = info['calib']
            rt_mat = calib['Rt']
            # follow Coord3DMode.convert_point
            rt_mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]
                               ]) @ rt_mat.transpose(1, 0)
            depth2img = calib['K'] @ rt_mat
            input_dict['depth2img'] = depth2img

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if self.filter_empty_gt and len(annos['gt_bboxes_3d']) == 0:
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
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]
        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(
                np.float32)  # k, 6
            gt_labels_3d = info['annos']['class'].astype(np.int64)
        else:
            gt_bboxes_3d = np.zeros((0, 7), dtype=np.float32)
            gt_labels_3d = np.zeros((0, ), dtype=np.int64)

        # to target box structure
        gt_bboxes_3d = DepthInstance3DBoxes(
            gt_bboxes_3d, origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d)

        if self.modality['use_camera']:
            if info['annos']['gt_num'] != 0:
                gt_bboxes_2d = info['annos']['bbox'].astype(np.float32)
            else:
                gt_bboxes_2d = np.zeros((0, 4), dtype=np.float32)
            anns_results['bboxes'] = gt_bboxes_2d
            anns_results['labels'] = gt_labels_3d

        return anns_results

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                load_dim=6,
                use_dim=[0, 1, 2]),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        if self.modality['use_camera']:
            pipeline.insert(0, dict(type='LoadImageFromFile'))
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
            points, img_metas, img = self._extract_data(
                i, pipeline, ['points', 'img_metas', 'img'])
            # scale colors to [0, 255]
            points = points.numpy()
            points[:, 3:] *= 255

            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            pred_bboxes = result['boxes_3d'].tensor.numpy()
            show_result(points, gt_bboxes.copy(), pred_bboxes.copy(), out_dir,
                        file_name, show)

            # multi-modality visualization
            if self.modality['use_camera']:
                img = img.numpy()
                # need to transpose channel to first dim
                img = img.transpose(1, 2, 0)
                pred_bboxes = DepthInstance3DBoxes(
                    pred_bboxes, origin=(0.5, 0.5, 0))
                gt_bboxes = DepthInstance3DBoxes(
                    gt_bboxes, origin=(0.5, 0.5, 0))
                show_multi_modality_result(
                    img,
                    gt_bboxes,
                    pred_bboxes,
                    None,
                    out_dir,
                    file_name,
                    box_mode='depth',
                    img_metas=img_metas,
                    show=show)

    def evaluate(self,
                 results,
                 metric=None,
                 iou_thr=(0.25, 0.5),
                 iou_thr_2d=(0.5, ),
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluate.

        Evaluation in indoor protocol.

        Args:
            results (list[dict]): List of results.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: None.
            iou_thr (list[float], optional): AP IoU thresholds for 3D
                evaluation. Default: (0.25, 0.5).
            iou_thr_2d (list[float], optional): AP IoU thresholds for 2D
                evaluation. Default: (0.5, ).
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict: Evaluation results.
        """
        # evaluate 3D detection performance
        if isinstance(results[0], dict):
            return super().evaluate(results, metric, iou_thr, logger, show,
                                    out_dir, pipeline)
        # evaluate 2D detection performance
        else:
            eval_results = OrderedDict()
            annotations = [self.get_ann_info(i) for i in range(len(self))]
            iou_thr_2d = (iou_thr_2d) if isinstance(iou_thr_2d,
                                                    float) else iou_thr_2d
            for iou_thr_2d_single in iou_thr_2d:
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=None,
                    iou_thr=iou_thr_2d_single,
                    dataset=self.CLASSES,
                    logger=logger)
                eval_results['mAP_' + str(iou_thr_2d_single)] = mean_ap
            return eval_results
