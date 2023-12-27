# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from mmengine import Config
from mmengine.device import get_device
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log

from mmdet3d.models.layers import box3d_multiclass_nms
from mmdet3d.registry import METRICS
from mmdet3d.structures import (Box3DMode, CameraInstance3DBoxes,
                                LiDARInstance3DBoxes, points_cam2img,
                                xywhr2xyxyr)


@METRICS.register_module()
class WaymoMetric(BaseMetric):
    """Waymo evaluation metric.

    Args:
        waymo_bin_file (str): The path of the annotation file in waymo format.
        metric (str or List[str]): Metrics to be evaluated. Defaults to 'mAP'.
        load_type (str): Type of loading mode during training.
            - 'frame_based': Load all of the instances in the frame.
            - 'mv_image_based': Load all of the instances in the frame and need
              to convert to the FOV-based data type to support image-based
              detector.
            - 'fov_image_based': Only load the instances inside the default cam
              and need to convert to the FOV-based data type to support image-
              based detector.
        result_prefix (str, optional): The prefix of result '*.bin' file,
            including the file path and the prefix of filename, e.g.,
            "a/b/prefix". If not specified, a temp file will be created.
            Defaults to None.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result to a
            specific format and submit it to the test server.
            Defaults to False.
        nms_cfg (dict): The configuration of non-maximum suppression for
            the mergence of multi-image predicted bboxes, only use when
            load_type == 'mv_image_based'. Defaults to None.
    """
    num_cams = 5
    default_prefix = 'Waymo metric'

    def __init__(self,
                 waymo_bin_file: str,
                 metric: Union[str, List[str]] = 'mAP',
                 load_type: str = 'frame_based',
                 result_prefix: Optional[str] = None,
                 format_only: bool = False,
                 nms_cfg=None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.waymo_bin_file = waymo_bin_file
        self.metrics = metric if isinstance(metric, list) else [metric]
        self.load_type = load_type
        self.result_prefix = result_prefix
        self.format_only = format_only
        if self.format_only:
            assert result_prefix is not None, 'result_prefix must be not '
            'None when format_only is True, otherwise the result files will '
            'be saved to a temp directory which will be cleaned up at the end.'
        if nms_cfg is not None:
            assert load_type == 'mv_image_based', 'nms_cfg in WaymoMetric '
            'only use when load_type == \'mv_image_based\'.'
            self.nms_cfg = Config(nms_cfg)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        for data_sample in data_samples:
            result = dict()
            bboxes_3d = data_sample['pred_instances_3d']['bboxes_3d']
            bboxes_3d.limit_yaw(offset=0.5, period=np.pi * 2)
            scores_3d = data_sample['pred_instances_3d']['scores_3d']
            labels_3d = data_sample['pred_instances_3d']['labels_3d']
            # TODO: check lidar post-processing
            if isinstance(bboxes_3d, CameraInstance3DBoxes):
                box_corners = bboxes_3d.corners
                cam2img = box_corners.new_tensor(
                    np.array(data_sample['cam2img']))
                box_corners_in_image = points_cam2img(box_corners, cam2img)
                # box_corners_in_image: [N, 8, 2]
                minxy = torch.min(box_corners_in_image, dim=1)[0]
                maxxy = torch.max(box_corners_in_image, dim=1)[0]
                # check minxy & maxxy
                # if the projected 2d bbox has intersection
                # with the image, we keep it, otherwise, we omit it.
                img_shape = data_sample['img_shape']
                valid_inds = ((minxy[:, 0] < img_shape[1]) &
                              (minxy[:, 1] < img_shape[0]) & (maxxy[:, 0] > 0)
                              & (maxxy[:, 1] > 0))

                if valid_inds.sum() > 0:
                    lidar2cam = data_sample['lidar2cam']
                    bboxes_3d = bboxes_3d.convert_to(
                        Box3DMode.LIDAR,
                        np.linalg.inv(lidar2cam),
                        correct_yaw=True)
                    bboxes_3d = bboxes_3d[valid_inds]
                    scores_3d = scores_3d[valid_inds]
                    labels_3d = labels_3d[valid_inds]
                else:
                    bboxes_3d = torch.zeros([0, 7])
                    scores_3d = torch.zeros([0])
                    labels_3d = torch.zeros([0])
            result['bboxes_3d'] = bboxes_3d.tensor.cpu().numpy()
            result['scores_3d'] = scores_3d.cpu().numpy()
            result['labels_3d'] = labels_3d.cpu().numpy()
            result['sample_idx'] = data_sample['sample_idx']
            result['context_name'] = data_sample['context_name']
            result['timestamp'] = data_sample['timestamp']
            self.results.append(result)

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List[dict]): The processed results of the whole dataset.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        self.classes = self.dataset_meta['classes']

        # different from kitti, waymo do not need to convert the ann file
        # handle the mv_image_based load_mode
        if self.load_type == 'mv_image_based':
            assert len(results) % 5 == 0, 'The multi-view image-based results'
            ' must be 5 times as large as the original frame-based results.'
            frame_results = [
                results[i:i + 5] for i in range(0, len(results), 5)
            ]
            results = self.merge_multi_view_boxes(frame_results)

        if self.result_prefix is None:
            eval_tmp_dir = tempfile.TemporaryDirectory()
            result_prefix = osp.join(eval_tmp_dir.name, 'results')
        else:
            eval_tmp_dir = None
            result_prefix = self.result_prefix

        self.format_results(results, result_prefix=result_prefix)

        metric_dict = {}

        if self.format_only:
            logger.info('results are saved in '
                        f'{osp.dirname(self.result_prefix)}')
            return metric_dict

        for metric in self.metrics:
            ap_dict = self.waymo_evaluate(
                result_prefix, metric=metric, logger=logger)
            metric_dict.update(ap_dict)
        if eval_tmp_dir is not None:
            eval_tmp_dir.cleanup()

        return metric_dict

    def waymo_evaluate(self,
                       result_prefix: str,
                       metric: Optional[str] = None,
                       logger: Optional[MMLogger] = None) -> Dict[str, float]:
        """Evaluation in Waymo protocol.

        Args:
            result_prefix (str): The location that stored the prediction
                results.
            metric (str, optional): Metric to be evaluated. Defaults to None.
            logger (MMLogger, optional): Logger used for printing related
                information during evaluation. Defaults to None.

        Returns:
            Dict[str, float]: Results of each evaluation metric.
        """

        import subprocess

        if metric == 'mAP':
            eval_str = 'mmdet3d/evaluation/functional/waymo_utils/' + \
                f'compute_detection_metrics_main {result_prefix}.bin ' + \
                f'{self.waymo_bin_file}'
            print(eval_str)
            ret_bytes = subprocess.check_output(eval_str, shell=True)
            ret_texts = ret_bytes.decode('utf-8')
            print_log(ret_texts, logger=logger)

            ap_dict = {
                'Vehicle/L1 mAP': 0,
                'Vehicle/L1 mAPH': 0,
                'Vehicle/L2 mAP': 0,
                'Vehicle/L2 mAPH': 0,
                'Pedestrian/L1 mAP': 0,
                'Pedestrian/L1 mAPH': 0,
                'Pedestrian/L2 mAP': 0,
                'Pedestrian/L2 mAPH': 0,
                'Sign/L1 mAP': 0,
                'Sign/L1 mAPH': 0,
                'Sign/L2 mAP': 0,
                'Sign/L2 mAPH': 0,
                'Cyclist/L1 mAP': 0,
                'Cyclist/L1 mAPH': 0,
                'Cyclist/L2 mAP': 0,
                'Cyclist/L2 mAPH': 0,
                'Overall/L1 mAP': 0,
                'Overall/L1 mAPH': 0,
                'Overall/L2 mAP': 0,
                'Overall/L2 mAPH': 0
            }
            mAP_splits = ret_texts.split('mAP ')
            mAPH_splits = ret_texts.split('mAPH ')
            for idx, key in enumerate(ap_dict.keys()):
                split_idx = int(idx / 2) + 1
                if idx % 2 == 0:  # mAP
                    ap_dict[key] = float(mAP_splits[split_idx].split(']')[0])
                else:  # mAPH
                    ap_dict[key] = float(mAPH_splits[split_idx].split(']')[0])
            ap_dict['Overall/L1 mAP'] = \
                (ap_dict['Vehicle/L1 mAP'] + ap_dict['Pedestrian/L1 mAP'] +
                    ap_dict['Cyclist/L1 mAP']) / 3
            ap_dict['Overall/L1 mAPH'] = \
                (ap_dict['Vehicle/L1 mAPH'] + ap_dict['Pedestrian/L1 mAPH'] +
                    ap_dict['Cyclist/L1 mAPH']) / 3
            ap_dict['Overall/L2 mAP'] = \
                (ap_dict['Vehicle/L2 mAP'] + ap_dict['Pedestrian/L2 mAP'] +
                    ap_dict['Cyclist/L2 mAP']) / 3
            ap_dict['Overall/L2 mAPH'] = \
                (ap_dict['Vehicle/L2 mAPH'] + ap_dict['Pedestrian/L2 mAPH'] +
                    ap_dict['Cyclist/L2 mAPH']) / 3
        elif metric == 'LET_mAP':
            eval_str = 'mmdet3d/evaluation/functional/waymo_utils/' + \
                f'compute_detection_let_metrics_main {result_prefix}.bin ' + \
                f'{self.waymo_bin_file}'

            print(eval_str)
            ret_bytes = subprocess.check_output(eval_str, shell=True)
            ret_texts = ret_bytes.decode('utf-8')

            print_log(ret_texts, logger=logger)
            ap_dict = {
                'Vehicle mAPL': 0,
                'Vehicle mAP': 0,
                'Vehicle mAPH': 0,
                'Pedestrian mAPL': 0,
                'Pedestrian mAP': 0,
                'Pedestrian mAPH': 0,
                'Sign mAPL': 0,
                'Sign mAP': 0,
                'Sign mAPH': 0,
                'Cyclist mAPL': 0,
                'Cyclist mAP': 0,
                'Cyclist mAPH': 0,
                'Overall mAPL': 0,
                'Overall mAP': 0,
                'Overall mAPH': 0
            }
            mAPL_splits = ret_texts.split('mAPL ')
            mAP_splits = ret_texts.split('mAP ')
            mAPH_splits = ret_texts.split('mAPH ')
            for idx, key in enumerate(ap_dict.keys()):
                split_idx = int(idx / 3) + 1
                if idx % 3 == 0:  # mAPL
                    ap_dict[key] = float(mAPL_splits[split_idx].split(']')[0])
                elif idx % 3 == 1:  # mAP
                    ap_dict[key] = float(mAP_splits[split_idx].split(']')[0])
                else:  # mAPH
                    ap_dict[key] = float(mAPH_splits[split_idx].split(']')[0])
            ap_dict['Overall mAPL'] = \
                (ap_dict['Vehicle mAPL'] + ap_dict['Pedestrian mAPL'] +
                    ap_dict['Cyclist mAPL']) / 3
            ap_dict['Overall mAP'] = \
                (ap_dict['Vehicle mAP'] + ap_dict['Pedestrian mAP'] +
                    ap_dict['Cyclist mAP']) / 3
            ap_dict['Overall mAPH'] = \
                (ap_dict['Vehicle mAPH'] + ap_dict['Pedestrian mAPH'] +
                    ap_dict['Cyclist mAPH']) / 3
        return ap_dict

    def format_results(
        self,
        results: List[dict],
        result_prefix: Optional[str] = None
    ) -> Tuple[dict, Union[tempfile.TemporaryDirectory, None]]:
        """Format the results to bin file.

        Args:
            results (List[dict]): Testing results of the dataset.
            result_prefix (str, optional): The prefix of result file. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Defaults to None.
        """
        waymo_results_final_path = f'{result_prefix}.bin'

        from ..functional.waymo_utils.prediction_to_waymo import \
            Prediction2Waymo
        converter = Prediction2Waymo(results, waymo_results_final_path,
                                     self.classes)
        converter.convert()

    def merge_multi_view_boxes(self, frame_results: List[dict]) -> dict:
        """Merge bounding boxes predicted from multi-view images.

        Args:
            box_dict_per_frame (List[dict]): The results of prediction for each
                camera.
            cam0_info (dict): Store the sample idx for the given frame.

        Returns:
            Dict: Merged results.
        """
        merged_results = []
        for frame_result in frame_results:
            merged_result = dict()
            merged_result['sample_idx'] = frame_result[0]['sample_idx'] // 5
            merged_result['context_name'] = frame_result[0]['context_name']
            merged_result['timestamp'] = frame_result[0]['timestamp']
            bboxes_3d, scores_3d, labels_3d = [], [], []
            for result in frame_result:
                assert result['timestamp'] == merged_result['timestamp']
                bboxes_3d.append(result['bboxes_3d'])
                scores_3d.append(result['scores_3d'])
                labels_3d.append(result['labels_3d'])

            bboxes_3d = np.concatenate(bboxes_3d)
            scores_3d = np.concatenate(scores_3d)
            labels_3d = np.concatenate(labels_3d)

            device = get_device()
            lidar_boxes3d = LiDARInstance3DBoxes(
                torch.from_numpy(bboxes_3d).to(device))
            scores = torch.from_numpy(scores_3d).to(device)
            labels = torch.from_numpy(labels_3d).long().to(device)
            nms_scores = scores.new_zeros(scores.shape[0],
                                          len(self.classes) + 1)
            indices = labels.new_tensor(list(range(scores.shape[0])))
            nms_scores[indices, labels] = scores
            lidar_boxes3d_for_nms = xywhr2xyxyr(lidar_boxes3d.bev)
            boxes3d = lidar_boxes3d.tensor
            bboxes_3d, scores_3d, labels_3d = box3d_multiclass_nms(
                boxes3d, lidar_boxes3d_for_nms, nms_scores,
                self.nms_cfg.score_thr, self.nms_cfg.max_per_frame,
                self.nms_cfg)

            merged_result['bboxes_3d'] = bboxes_3d.cpu().numpy()
            merged_result['scores_3d'] = scores_3d.cpu().numpy()
            merged_result['labels_3d'] = labels_3d.cpu().numpy()
            merged_results.append(merged_result)
        return merged_results
