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
from mmdet3d.structures import (Box3DMode, CameraInstance3DBoxes,
                                LiDARInstance3DBoxes, bbox3d2result,
                                points_cam2img, xywhr2xyxyr)
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
            Default to 'mAP'.
        pcd_limit_range (list): The range of point cloud used to
            filter invalid predicted boxes.
            Default to [0, -40, -3, 70.4, 40, 0.0].
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
        convert_kitti_format (bool, optional): Whether convert the reuslts to
            kitti format. Now, in order to be compatible with camera-based
            methods, defaults to True.
        pklfile_prefix (str, optional): The prefix of pkl files, including
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Default: None.
        submission_prefix (str, optional): The prefix of submission data.
            If not specified, the submission data will not be generated.
            Default: None.
        load_type (str, optional): Type of loading mode during training.

            - 'frame_based': Load all of the instances in the frame.
            - 'mv_image_based': Load all of the instances in the frame and need
                to convert to the FOV-based data type to support image-based
                detector.
            - 'fov_image_base': Only load the instances inside the default cam,
                and need to convert to the FOV-based data type to support
                image-based detector.
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
            Defaults to ``dict(backend='disk')``.
        idx2metainfo (Optional[str], optional): The file path of the metainfo
            in waymmo. It stores the mapping from sample_idx to metainfo.
            The metainfo must contain the keys: 'idx2contextname' and
            'idx2timestamp'. Defaults to None.
    """
    num_cams = 5

    def __init__(self,
                 ann_file: str,
                 waymo_bin_file: str,
                 data_root: str,
                 split: str = 'training',
                 metric: Union[str, List[str]] = 'mAP',
                 pcd_limit_range: List[float] = [-85, -85, -5, 85, 85, 5],
                 convert_kitti_format: bool = True,
                 prefix: Optional[str] = None,
                 pklfile_prefix: str = None,
                 submission_prefix: str = None,
                 load_type: str = 'frame_based',
                 default_cam_key: str = 'CAM_FRONT',
                 use_pred_sample_idx: bool = False,
                 collect_device: str = 'cpu',
                 file_client_args: dict = dict(backend='disk'),
                 idx2metainfo: Optional[str] = None):
        self.waymo_bin_file = waymo_bin_file
        self.data_root = data_root
        self.split = split
        self.load_type = load_type
        self.use_pred_sample_idx = use_pred_sample_idx
        self.convert_kitti_format = convert_kitti_format

        if idx2metainfo is not None:
            self.idx2metainfo = mmengine.load(idx2metainfo)
        else:
            self.idx2metainfo = None

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
            results (list): The processed results of the whole dataset.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        self.classes = self.dataset_meta['classes']

        # load annotations
        self.data_infos = load(self.ann_file)['data_list']
        assert len(results) == len(self.data_infos), \
            'invalid list length of network outputs'
        # different from kitti, waymo do not need to convert the ann file
        # handle the mv_image_based load_mode
        if self.load_type == 'mv_image_based':
            new_data_infos = []
            for info in self.data_infos:
                height = info['images'][self.default_cam_key]['height']
                width = info['images'][self.default_cam_key]['width']
                for (cam_key, img_info) in info['images'].items():
                    camera_info = dict()
                    camera_info['images'] = dict()
                    camera_info['images'][cam_key] = img_info
                    # TODO remove the check by updating the data info;
                    if 'height' not in img_info:
                        img_info['height'] = height
                        img_info['width'] = width
                    if 'cam_instances' in info \
                            and cam_key in info['cam_instances']:
                        camera_info['instances'] = info['cam_instances'][
                            cam_key]
                    else:
                        camera_info['instances'] = []
                    camera_info['ego2global'] = info['ego2global']
                    if 'image_sweeps' in info:
                        camera_info['image_sweeps'] = info['image_sweeps']

                    # TODO check if need to modify the sample id
                    # TODO check when will use it except for evaluation.
                    camera_info['sample_idx'] = info['sample_idx']
                    new_data_infos.append(camera_info)
            self.data_infos = new_data_infos

        if self.pklfile_prefix is None:
            eval_tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(eval_tmp_dir.name, 'results')
        else:
            eval_tmp_dir = None
            pklfile_prefix = self.pklfile_prefix

        result_dict, tmp_dir = self.format_results(
            results,
            pklfile_prefix=pklfile_prefix,
            submission_prefix=self.submission_prefix,
            classes=self.classes)

        metric_dict = {}
        for metric in self.metrics:
            ap_dict = self.waymo_evaluate(
                pklfile_prefix, metric=metric, logger=logger)
            metric_dict[metric] = ap_dict
        if eval_tmp_dir is not None:
            eval_tmp_dir.cleanup()

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return metric_dict

    def waymo_evaluate(self,
                       pklfile_prefix: str,
                       metric: str = None,
                       logger: MMLogger = None) -> dict:
        """Evaluation in Waymo protocol.

        Args:
            pklfile_prefix (str): The location that stored the prediction
                results.
            metric (str): Metric to be evaluated. Defaults to None.
            logger (MMLogger, optional): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """

        import subprocess

        if metric == 'mAP':
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
        elif metric == 'LET_mAP':
            eval_str = 'mmdet3d/evaluation/functional/waymo_utils/' + \
                f'compute_detection_let_metrics_main {pklfile_prefix}.bin ' + \
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

    def format_results(self,
                       results: List[dict],
                       pklfile_prefix: str = None,
                       submission_prefix: str = None,
                       classes: List[str] = None):
        """Format the results to bin file.

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
        waymo_save_tmp_dir = tempfile.TemporaryDirectory()
        waymo_results_save_dir = waymo_save_tmp_dir.name
        waymo_results_final_path = f'{pklfile_prefix}.bin'

        if self.convert_kitti_format:
            results_kitti_format, tmp_dir = super().format_results(
                results, pklfile_prefix, submission_prefix, classes)
            final_results = results_kitti_format['pred_instances_3d']
        else:
            final_results = results
            for i, res in enumerate(final_results):
                # Actually, `sample_idx` here is the filename without suffix.
                # It's for identitying the sample in formating.
                res['sample_idx'] = self.data_infos[i]['sample_idx']
                res['pred_instances_3d']['bboxes_3d'].limit_yaw(
                    offset=0.5, period=np.pi * 2)

        waymo_root = self.data_root
        if self.split == 'training':
            waymo_tfrecords_dir = osp.join(waymo_root, 'validation')
            prefix = '1'
        elif self.split == 'testing':
            waymo_tfrecords_dir = osp.join(waymo_root, 'testing')
            prefix = '2'
        else:
            raise ValueError('Not supported split value.')

        from ..functional.waymo_utils.prediction_to_waymo import \
            Prediction2Waymo
        converter = Prediction2Waymo(
            final_results,
            waymo_tfrecords_dir,
            waymo_results_save_dir,
            waymo_results_final_path,
            prefix,
            classes,
            file_client_args=self.file_client_args,
            from_kitti_format=self.convert_kitti_format,
            idx2metainfo=self.idx2metainfo)
        converter.convert()
        waymo_save_tmp_dir.cleanup()

        return final_results, waymo_save_tmp_dir

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
        box_dict['sample_idx'] = cam0_info['image_id']
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
        nms_scores = scores.new_zeros(scores.shape[0], len(self.classes) + 1)
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
        box_preds_lidar = det['bboxes_3d']
        scores = det['scores_3d']
        labels = det['labels_3d']
        # box_preds_camera is in the cam0 system
        lidar2cam = cam0_info['images'][self.default_cam_key]['lidar2img']
        lidar2cam = np.array(lidar2cam).astype(np.float32)
        box_preds_camera = box_preds_lidar.convert_to(
            Box3DMode.CAM, lidar2cam, correct_yaw=True)
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
        if submission_prefix is not None:
            mmengine.mkdir_or_exist(submission_prefix)

        det_annos = []
        print('\nConverting prediction to KITTI format')
        for idx, pred_dicts in enumerate(
                mmengine.track_iter_progress(net_outputs)):
            annos = []
            sample_idx = sample_id_list[idx]
            info = self.data_infos[sample_idx]

            if self.load_type == 'mv_image_based':
                if idx % self.num_cams == 0:
                    box_dict_per_frame = []
                    cam0_key = list(info['images'].keys())[0]
                    cam0_info = info
                    # Here in mono3d, we use the 'CAM_FRONT' "the first
                    # index in the camera" as the default image shape.
                    # If you want to another camera, please modify it.
                    image_shape = (info['images'][cam0_key]['height'],
                                   info['images'][cam0_key]['width'])
                box_dict = self.convert_valid_bboxes(pred_dicts, info)
            else:
                box_dict = self.convert_valid_bboxes(pred_dicts, info)
                # Here default used 'CAM_FRONT' to compute metric.
                # If you want to use another camera, please modify it.
                image_shape = (info['images'][self.default_cam_key]['height'],
                               info['images'][self.default_cam_key]['width'])
            if self.load_type == 'mv_image_based':
                box_dict_per_frame.append(box_dict)
                if (idx + 1) % self.num_cams != 0:
                    continue
                box_dict = self.merge_multi_view_boxes(box_dict_per_frame,
                                                       cam0_info)

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
            annos[-1]['sample_idx'] = np.array(
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

    def convert_valid_bboxes(self, box_dict: dict, info: dict):
        """Convert the predicted boxes into valid ones. Should handle the
        load_model (frame_based, mv_image_based, fov_image_based), separately.

        Args:
            box_dict (dict): Box dictionaries to be converted.

                - bboxes_3d (:obj:`LiDARInstance3DBoxes`): 3D bounding boxes.
                - scores_3d (torch.Tensor): Scores of boxes.
                - labels_3d (torch.Tensor): Class labels of boxes.
            info (dict): Data info.

        Returns:
            dict: Valid predicted boxes.

                - bbox (np.ndarray): 2D bounding boxes.
                - box3d_camera (np.ndarray): 3D bounding boxes in
                    camera coordinate.
                - box3d_lidar (np.ndarray): 3D bounding boxes in
                    LiDAR coordinate.
                - scores (np.ndarray): Scores of boxes.
                - label_preds (np.ndarray): Class label predictions.
                - sample_idx (int): Sample index.
        """
        # TODO: refactor this function
        box_preds = box_dict['bboxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        sample_idx = info['sample_idx']
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)

        if len(box_preds) == 0:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)
        # Here default used 'CAM_FRONT' to compute metric. If you want to
        # use another camera, please modify it.
        if self.load_type in ['frame_based', 'fov_image_based']:
            cam_key = self.default_cam_key
        elif self.load_type == 'mv_image_based':
            cam_key = list(info['images'].keys())[0]
        else:
            raise NotImplementedError

        lidar2cam = np.array(info['images'][cam_key]['lidar2cam']).astype(
            np.float32)
        P2 = np.array(info['images'][cam_key]['cam2img']).astype(np.float32)
        img_shape = (info['images'][cam_key]['height'],
                     info['images'][cam_key]['width'])
        P2 = box_preds.tensor.new_tensor(P2)

        if isinstance(box_preds, LiDARInstance3DBoxes):
            box_preds_camera = box_preds.convert_to(Box3DMode.CAM, lidar2cam)
            box_preds_lidar = box_preds
        elif isinstance(box_preds, CameraInstance3DBoxes):
            box_preds_camera = box_preds
            box_preds_lidar = box_preds.convert_to(Box3DMode.LIDAR,
                                                   np.linalg.inv(lidar2cam))

        box_corners = box_preds_camera.corners
        box_corners_in_image = points_cam2img(box_corners, P2)
        # box_corners_in_image: [N, 8, 2]
        minxy = torch.min(box_corners_in_image, dim=1)[0]
        maxxy = torch.max(box_corners_in_image, dim=1)[0]
        box_2d_preds = torch.cat([minxy, maxxy], dim=1)
        # Post-processing
        # check box_preds_camera
        image_shape = box_preds.tensor.new_tensor(img_shape)
        valid_cam_inds = ((box_2d_preds[:, 0] < image_shape[1]) &
                          (box_2d_preds[:, 1] < image_shape[0]) &
                          (box_2d_preds[:, 2] > 0) & (box_2d_preds[:, 3] > 0))
        # check box_preds_lidar
        if self.load_type in ['frame_based']:
            limit_range = box_preds.tensor.new_tensor(self.pcd_limit_range)
            valid_pcd_inds = ((box_preds_lidar.center > limit_range[:3]) &
                              (box_preds_lidar.center < limit_range[3:]))
            valid_inds = valid_pcd_inds.all(-1)
        if self.load_type in ['mv_image_based', 'fov_image_based']:
            valid_inds = valid_cam_inds

        if valid_inds.sum() > 0:
            return dict(
                bbox=box_2d_preds[valid_inds, :].numpy(),
                pred_box_type_3d=type(box_preds),
                box3d_camera=box_preds_camera[valid_inds].tensor.numpy(),
                box3d_lidar=box_preds_lidar[valid_inds].tensor.numpy(),
                scores=scores[valid_inds].numpy(),
                label_preds=labels[valid_inds].numpy(),
                sample_idx=sample_idx)
        else:
            return dict(
                bbox=np.zeros([0, 4]),
                pred_box_type_3d=type(box_preds),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0]),
                sample_idx=sample_idx)
