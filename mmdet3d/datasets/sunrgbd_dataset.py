import numpy as np
from os import path as osp

from mmdet3d.core import show_result
from mmdet3d.core.bbox import DepthInstance3DBoxes
from mmdet.datasets import DATASETS
from .custom_3d import Custom3DDataset


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
                 modality=None,
                 box_type_3d='Depth',
                 filter_empty_gt=True,
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
        if self.modality is None:
            self.modality = dict(
                use_camera=True,
                use_lidar=True,
            )
        assert self.modality['use_camera'] or self.modality['use_lidar']

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str, optional): Filename of point clouds.
                - file_name (str, optional): Filename of point clouds.
                - img_prefix (str | None, optional): Prefix of image files.
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
            img_filename = osp.join(self.data_root,
                                    info['image']['image_path'])
            input_dict['img_prefix'] = None
            input_dict['img_info'] = dict(filename=img_filename)
            calib = info['calib']
            input_dict['calib'] = calib

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

                - gt_bboxes_3d (:obj:`DepthInstance3DBoxes`): \
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
            gt_labels_3d = info['annos']['class'].astype(np.long)
        else:
            gt_bboxes_3d = np.zeros((0, 7), dtype=np.float32)
            gt_labels_3d = np.zeros((0, ), dtype=np.long)

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

    def show(self, results, out_dir):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        for i, result in enumerate(results):
            data_info = self.data_infos[i]
            pts_path = data_info['pts_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = np.fromfile(
                osp.join(self.data_root, pts_path),
                dtype=np.float32).reshape(-1, 6)
            points[:, 3:] *= 255
            if data_info['annos']['gt_num'] > 0:
                gt_bboxes = data_info['annos']['gt_boxes_upright_depth']
            else:
                gt_bboxes = np.zeros((0, 7))
            pred_bboxes = result['boxes_3d'].tensor.numpy()
            pred_bboxes[..., 2] += pred_bboxes[..., 5] / 2
            show_result(points, gt_bboxes, pred_bboxes, out_dir, file_name)

    def evaluate(self,
                 results,
                 metric=None,
                 iou_thr=(0.25, 0.5),
                 logger=None,
                 show=False,
                 out_dir=None):

        if isinstance(results[0], dict):
            return super().evaluate(results, metric, iou_thr, logger, show,
                                    out_dir)

        # iou_thrs = (0.5)
        # jsonfile_prefix = None,
        # classwise = False,
        # proposal_nums = (100, 300, 1000),
        # metric_items = None

        # result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        # eval_results = {}
        # cocoGt = self.coco
        # for metric in metrics:
        #     msg = f'Evaluating {metric}...'
        #     if logger is None:
        #         msg = '\n' + msg
        #     print_log(msg, logger=logger)

        #     if metric == 'proposal_fast':
        #         ar = self.fast_eval_recall(
        #             results, proposal_nums, iou_thrs, logger='silent')
        #         log_msg = []
        #         for i, num in enumerate(proposal_nums):
        #             eval_results[f'AR@{num}'] = ar[i]
        #             log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
        #         log_msg = ''.join(log_msg)
        #         print_log(log_msg, logger=logger)
        #         continue

        #     if metric not in result_files:
        #         raise KeyError(f'{metric} is not in results')
        #     try:
        #         cocoDt = cocoGt.loadRes(result_files[metric])
        #     except IndexError:
        #         print_log(
        #             'The testing results of the whole dataset is empty.',
        #             logger=logger,
        #             level=logging.ERROR)
        #         break

        #     iou_type = 'bbox' if metric == 'proposal' else metric
        #     cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
        #     cocoEval.params.catIds = self.cat_ids
        #     cocoEval.params.imgIds = self.img_ids
        #     cocoEval.params.maxDets = list(proposal_nums)
        #     cocoEval.params.iouThrs = iou_thrs
        #     # mapping of cocoEval.stats
        #     coco_metric_names = {
        #         'mAP': 0,
        #         'mAP_50': 1,
        #         'mAP_75': 2,
        #         'mAP_s': 3,
        #         'mAP_m': 4,
        #         'mAP_l': 5,
        #         'AR@100': 6,
        #         'AR@300': 7,
        #         'AR@1000': 8,
        #         'AR_s@1000': 9,
        #         'AR_m@1000': 10,
        #         'AR_l@1000': 11
        #     }
        #     if metric_items is not None:
        #         for metric_item in metric_items:
        #             if metric_item not in coco_metric_names:
        #                 raise KeyError(
        #                     f'metric item {metric_item} is not supported')

        #     if metric == 'proposal':
        #         cocoEval.params.useCats = 0
        #         cocoEval.evaluate()
        #         cocoEval.accumulate()
        #         cocoEval.summarize()
        #         if metric_items is None:
        #             metric_items = [
        #                 'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
        #                 'AR_m@1000', 'AR_l@1000'
        #             ]

        #         for item in metric_items:
        #             val = float(
        #                 f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
        #             eval_results[item] = val
        #     else:
        #         cocoEval.evaluate()
        #         cocoEval.accumulate()
        #         cocoEval.summarize()
        #         if classwise:  # Compute per-category AP
        #             # Compute per-category AP
        #             # from https://github.com/facebookresearch/detectron2/
        #             precisions = cocoEval.eval['precision']
        #             # precision: (iou, recall, cls, area range, max dets)
        #             assert len(self.cat_ids) == precisions.shape[2]

        #             results_per_category = []
        #             for idx, catId in enumerate(self.cat_ids):
        #                 # area range index 0: all area ranges
        #                 # max dets index -1: typically 100 per image
        #                 nm = self.coco.loadCats(catId)[0]
        #                 precision = precisions[:, :, idx, 0, -1]
        #                 precision = precision[precision > -1]
        #                 if precision.size:
        #                     ap = np.mean(precision)
        #                 else:
        #                     ap = float('nan')
        #                 results_per_category.append(
        #                     (f'{nm["name"]}', f'{float(ap):0.3f}'))

        #             num_columns = min(6, len(results_per_category) * 2)
        #             results_flatten = list(
        #                 itertools.chain(*results_per_category))
        #             headers = ['category', 'AP'] * (num_columns // 2)
        #             results_2d = itertools.zip_longest(*[
        #                 results_flatten[i::num_columns]
        #                 for i in range(num_columns)
        #             ])
        #             table_data = [headers]
        #             table_data += [result for result in results_2d]
        #             table = AsciiTable(table_data)
        #             print_log('\n' + table.table, logger=logger)

        #         if metric_items is None:
        #             metric_items = [
        #                 'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
        #             ]

        #         for metric_item in metric_items:
        #             key = f'{metric}_{metric_item}'
        #             val = float(
        #                 f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
        #             )
        #             eval_results[key] = val
        #         ap = cocoEval.stats[:6]
        #         eval_results[f'{metric}_mAP_copypaste'] = (
        #             f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
        #             f'{ap[4]:.3f} {ap[5]:.3f}')
        # if tmp_dir is not None:
        #     tmp_dir.cleanup()
        # return eval_results
