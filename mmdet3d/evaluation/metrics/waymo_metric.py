# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from os import path as osp
from typing import Dict, List, Optional, Union

import mmengine
import numpy as np
import torch
from mmengine import Config, load
from mmengine.logging import MMLogger, print_log

from mmdet3d.models.layers import box3d_multiclass_nms
from mmdet3d.registry import METRICS
from mmdet3d.structures import (Box3DMode, LiDARInstance3DBoxes, bbox3d2result,
                                xywhr2xyxyr)
from .kitti_metric import KittiMetric


@METRICS.register_module()
class WaymoMetric(KittiMetric):
    """Waymo evaluation metric.

    Args:
        ann_file (str): The path of the annotation file in kitti format.
        waymo_bin_file (str): The path of the annotation file in waymo format.
        data_root (str): Path of dataset root.
                         Used for storing waymo evaluation programs.
        split (str): The split of the evaluation set.
        metric (str | list[str]): Metrics to be evaluated.
            Default to 'bbox'.
        pcd_limit_range (list): The range of point cloud used to
            filter invalid predicted boxes.
            Default to [0, -40, -3, 70.4, 40, 0.0].
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
        pklfile_prefix (str, optional): The prefix of pkl files, including
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Default: None.
        submission_prefix (str, optional): The prefix of submission data.
            If not specified, the submission data will not be generated.
            Default: None.
        task: (str, optional): task for 3D detection, if cam, would filter
            the points that outside the image.
        default_cam_key (str, optional): The default camera for lidar to
            camear conversion. By default, KITTI: CAM2, Waymo: CAM_FRONT
        use_pred_sample_idx (bool, optional): In formating results, use the
            sample index from the prediction or from the load annoataitons.
            By default, KITTI: True, Waymo: False, Waymo has a conversion
            process, which needs to use the sample id from load annotation.
        collect_device (str): Device name used for collecting results
            from different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        file_client_args (dict): file client for reading gt in waymo format.
    """

    def __init__(self,
                 ann_file: str,
                 waymo_bin_file: str,
                 data_root: str,
                 split: str = 'training',
                 metric: Union[str, List[str]] = 'bbox',
                 pcd_limit_range: List[float] = [-85, -85, -5, 85, 85, 5],
                 prefix: Optional[str] = None,
                 pklfile_prefix: str = None,
                 submission_prefix: str = None,
                 task='lidar',
                 default_cam_key: str = 'CAM_FRONT',
                 use_pred_sample_idx: bool = False,
                 collect_device: str = 'cpu',
                 file_client_args: dict = dict(backend='disk')):

        self.waymo_bin_file = waymo_bin_file
        self.data_root = data_root
        self.split = split
        self.task = task
        self.use_pred_sample_idx = use_pred_sample_idx
        super().__init__(
            ann_file=ann_file,
            metric=metric,
            pcd_limit_range=pcd_limit_range,
            prefix=prefix,
            pklfile_prefix=pklfile_prefix,
            submission_prefix=submission_prefix,
            default_cam_key=default_cam_key,
            collect_device=collect_device,
            file_client_args=file_client_args)
        self.default_prefix = 'Waymo metric'

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        self.classes = self.dataset_meta['CLASSES']

        # load annotations
        self.data_infos = load(self.ann_file)['data_list']
        # different from kitti, waymo do not need to convert the ann file

        if self.pklfile_prefix is None:
            eval_tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(eval_tmp_dir.name, 'results')
        else:
            eval_tmp_dir = None
            pklfile_prefix = self.pklfile_prefix

        # load annotations

        result_dict, tmp_dir = self.format_results(
            results,
            pklfile_prefix=pklfile_prefix,
            submission_prefix=self.submission_prefix,
            classes=self.classes)

        import subprocess
        eval_str = 'mmdet3d/evaluation/functional/waymo_utils/' + \
            f'compute_detection_metrics_main {pklfile_prefix}.bin ' + \
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
        if eval_tmp_dir is not None:
            eval_tmp_dir.cleanup()

        if tmp_dir is not None:
            tmp_dir.cleanup()

        return ap_dict

    def format_results(self,
                       results: List[dict],
                       pklfile_prefix: str = None,
                       submission_prefix: str = None,
                       classes: List[str] = None):
        """Format the results to pkl file.

        Args:
            results (list[dict]): Testing results of the
                dataset.
            pklfile_prefix (str, optional): The prefix of pkl files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.
            submission_prefix (str, optional): The prefix of submitted files.
                It includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.
            classes (list[String], optional): A list of class name. Defaults
                to None.

        Returns:
            tuple: (result_dict, tmp_dir), result_dict is a dict containing
                the formatted result, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        result_files, tmp_dir = super().format_results(results, pklfile_prefix,
                                                       submission_prefix,
                                                       classes)

        waymo_root = self.data_root
        if self.split == 'training':
            waymo_tfrecords_dir = osp.join(waymo_root, 'validation')
            prefix = '1'
        elif self.split == 'testing':
            waymo_tfrecords_dir = osp.join(waymo_root, 'testing')
            prefix = '2'
        else:
            raise ValueError('Not supported split value.')
        waymo_save_tmp_dir = tempfile.TemporaryDirectory()
        waymo_results_save_dir = waymo_save_tmp_dir.name
        waymo_results_final_path = f'{pklfile_prefix}.bin'
        from ..functional.waymo_utils.prediction_kitti_to_waymo import \
            KITTI2Waymo
        converter = KITTI2Waymo(
            result_files['pred_instances_3d'],
            waymo_tfrecords_dir,
            waymo_results_save_dir,
            waymo_results_final_path,
            prefix,
            file_client_args=self.file_client_args)
        converter.convert()
        waymo_save_tmp_dir.cleanup()
        return result_files, waymo_save_tmp_dir

    def merge_multi_view_boxes(self, box_dict_per_frame: List[dict],
                               cam0_info: dict):
        """Merge bounding boxes predicted from multi-view images.
        Args:
            box_dict_per_frame (list[dict]): The results of prediction
                for each camera.
            cam2_info (dict): store the sample id for the given frame.

        Returns:
            merged_box_dict (dict), store the merge results
        """
        box_dict = dict()
        # convert list[dict] to dict[list]
        for key in box_dict_per_frame[0].keys():
            box_dict[key] = list()
            for cam_idx in range(self.num_cams):
                box_dict[key].append(box_dict_per_frame[cam_idx][key])
        # merge each elements
        box_dict['sample_id'] = cam0_info['image_id']
        for key in ['bbox', 'box3d_lidar', 'scores', 'label_preds']:
            box_dict[key] = np.concatenate(box_dict[key])

        # apply nms to box3d_lidar (box3d_camera are in different systems)
        # TODO: move this global setting into config
        nms_cfg = dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=500,
            nms_thr=0.05,
            score_thr=0.001,
            min_bbox_size=0,
            max_per_frame=100)
        nms_cfg = Config(nms_cfg)
        lidar_boxes3d = LiDARInstance3DBoxes(
            torch.from_numpy(box_dict['box3d_lidar']).cuda())
        scores = torch.from_numpy(box_dict['scores']).cuda()
        labels = torch.from_numpy(box_dict['label_preds']).long().cuda()
        nms_scores = scores.new_zeros(scores.shape[0], len(self.CLASSES) + 1)
        indices = labels.new_tensor(list(range(scores.shape[0])))
        nms_scores[indices, labels] = scores
        lidar_boxes3d_for_nms = xywhr2xyxyr(lidar_boxes3d.bev)
        boxes3d = lidar_boxes3d.tensor
        # generate attr scores from attr labels
        boxes3d, scores, labels = box3d_multiclass_nms(
            boxes3d, lidar_boxes3d_for_nms, nms_scores, nms_cfg.score_thr,
            nms_cfg.max_per_frame, nms_cfg)
        lidar_boxes3d = LiDARInstance3DBoxes(boxes3d)
        det = bbox3d2result(lidar_boxes3d, scores, labels)
        box_preds_lidar = det['boxes_3d']
        scores = det['scores_3d']
        labels = det['labels_3d']
        # box_preds_camera is in the cam0 system
        rect = cam0_info['calib']['R0_rect'].astype(np.float32)
        Trv2c = cam0_info['calib']['Tr_velo_to_cam'].astype(np.float32)
        box_preds_camera = box_preds_lidar.convert_to(
            Box3DMode.CAM, rect @ Trv2c, correct_yaw=True)
        # Note: bbox is meaningless in final evaluation, set to 0
        merged_box_dict = dict(
            bbox=np.zeros([box_preds_lidar.tensor.shape[0], 4]),
            box3d_camera=box_preds_camera.tensor.numpy(),
            box3d_lidar=box_preds_lidar.tensor.numpy(),
            scores=scores.numpy(),
            label_preds=labels.numpy(),
            sample_idx=box_dict['sample_idx'],
        )
        return merged_box_dict

    def bbox2result_kitti(self,
                          net_outputs: list,
                          sample_id_list: list,
                          class_names: list,
                          pklfile_prefix: str = None,
                          submission_prefix: str = None):
        """Convert 3D detection results to kitti format for evaluation and test
        submission.

        Args:
            net_outputs (list[dict]): List of array storing the
                inferenced bounding boxes and scores.
            sample_id_list (list[int]): List of input sample id.
            class_names (list[String]): A list of class names.
            pklfile_prefix (str, optional): The prefix of pkl file.
                Defaults to None.
            submission_prefix (str, optional): The prefix of submission file.
                Defaults to None.

        Returns:
            list[dict]: A list of dictionaries with the kitti format.
        """
        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'
        if submission_prefix is not None:
            mmengine.mkdir_or_exist(submission_prefix)

        det_annos = []
        print('\nConverting prediction to KITTI format')
        for idx, pred_dicts in enumerate(
                mmengine.track_iter_progress(net_outputs)):
            annos = []
            sample_idx = sample_id_list[idx]
            info = self.data_infos[sample_idx]
            # Here default used 'CAM2' to compute metric. If you want to
            # use another camera, please modify it.
            image_shape = (info['images'][self.default_cam_key]['height'],
                           info['images'][self.default_cam_key]['width'])

            if self.task == 'mono3d':
                if idx % self.num_cams == 0:
                    box_dict_per_frame = []
                    cam0_idx = idx
            box_dict = self.convert_valid_bboxes(pred_dicts, info)

            if self.task == 'mono3d':
                box_dict_per_frame.append(box_dict)
                if (idx + 1) % self.num_cams != 0:
                    continue
                box_dict = self.merge_multi_view_boxes(
                    box_dict_per_frame, self.data_infos[cam0_idx])
            anno = {
                'name': [],
                'truncated': [],
                'occluded': [],
                'alpha': [],
                'bbox': [],
                'dimensions': [],
                'location': [],
                'rotation_y': [],
                'score': []
            }
            if len(box_dict['bbox']) > 0:
                box_2d_preds = box_dict['bbox']
                box_preds = box_dict['box3d_camera']
                scores = box_dict['scores']
                box_preds_lidar = box_dict['box3d_lidar']
                label_preds = box_dict['label_preds']

                for box, box_lidar, bbox, score, label in zip(
                        box_preds, box_preds_lidar, box_2d_preds, scores,
                        label_preds):
                    bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                    bbox[:2] = np.maximum(bbox[:2], [0, 0])
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    anno['alpha'].append(
                        -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6])
                    anno['bbox'].append(bbox)
                    anno['dimensions'].append(box[3:6])
                    anno['location'].append(box[:3])
                    anno['rotation_y'].append(box[6])
                    anno['score'].append(score)

                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)
            else:
                anno = {
                    'name': np.array([]),
                    'truncated': np.array([]),
                    'occluded': np.array([]),
                    'alpha': np.array([]),
                    'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.array([]),
                    'score': np.array([]),
                }
                annos.append(anno)

            if submission_prefix is not None:
                curr_file = f'{submission_prefix}/{sample_idx:06d}.txt'
                with open(curr_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print(
                            '{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
                                anno['name'][idx], anno['alpha'][idx],
                                bbox[idx][0], bbox[idx][1], bbox[idx][2],
                                bbox[idx][3], dims[idx][1], dims[idx][2],
                                dims[idx][0], loc[idx][0], loc[idx][1],
                                loc[idx][2], anno['rotation_y'][idx],
                                anno['score'][idx]),
                            file=f)
            if self.use_pred_sample_idx:
                save_sample_idx = sample_idx
            else:
                # use the sample idx in the info file
                # In waymo validation sample_idx in prediction is 000xxx
                # but in info file it is 1000xxx
                save_sample_idx = box_dict['sample_idx']
            annos[-1]['sample_id'] = np.array(
                [save_sample_idx] * len(annos[-1]['score']), dtype=np.int64)

            det_annos += annos

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            else:
                out = pklfile_prefix
            mmengine.dump(det_annos, out)
            print(f'Result is saved to {out}.')

        return det_annos
