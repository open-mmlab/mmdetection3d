# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp
import numpy as np

import mmcv
from mmcv.utils import print_log

from mmdet.datasets import DATASETS
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

        self.ignore = np.array(ignore, dtype=np.int64)
        self.n_classes = len(self.CLASSES)

        self.include = np.array([n for n in range(self.n_classes) if n not in self.ignore], dtype=np.int64)
        self.offset = offset
        self.eps = 1e-15
        self.min_points = min_points
        self.reset()
        self.global_cfg = mmcv.load(osp.join(self.data_root, 'semantic-kitti.yaml'))

    def reset(self):
        # general things
        # iou stuff
        self.px_iou_conf_matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)
        # panoptic stuff
        self.pan_tp = np.zeros(self.n_classes, dtype=np.int64)
        self.pan_iou = np.zeros(self.n_classes, dtype=np.double)
        self.pan_fp = np.zeros(self.n_classes, dtype=np.int64)
        self.pan_fn = np.zeros(self.n_classes, dtype=np.int64)

        self.evaluated_fnames = []

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
        class_remap = self.global_cfg["learning_map"]
        gt_annos = []
        for i, seq in enumerate(filename):

            gt = np.fromfile(osp.join(self.data_root, seq), dtype=np.int32).reshape(-1, 1)
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

                ap_result_str, ap_dict_ = self.semantic_kitti_eval(
                    gt_annos,
                    result_files_,
                    self.CLASSES)
                for ap_type, ap in ap_dict_.items():
                    ap_dict[f'{name}/{ap_type}'] = float('{:.4f}'.format(ap))
                ap_dict_list.append(ap_dict)
                print_log(
                    f'Results of {name}:\n' + ap_result_str, logger=logger)

        else:
            ap_result_str, ap_dict_list = self.semantic_kitti_eval(gt_annos, result_files)
            print_log('\n' + ap_result_str, logger=logger)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)
        return ap_dict_list

    def semantic_kitti_eval(self, gt_annos, result_files):

        assert len(gt_annos) == len(result_files)
        result_str = ''
        result_dict_list = []
        for f in range(len(gt_annos)):

            pred_sem = result_files[f]['pred_sem']
            pred_inst = result_files[f]['pred_inst']
            gt_sem = gt_annos[f]['gt_sem']
            gt_inst = gt_annos[f]['gt_inst']

            self.batch_sem_IoU(pred_sem, gt_sem)
            self.batch_panoptic(pred_sem, gt_sem, pred_inst, gt_inst)

            class_PQ, class_SQ, class_RQ, class_all_PQ, class_all_SQ, class_all_RQ = self.getPQ()
            class_IoU, class_all_IoU = self.getSemIoU()

            # make python variables
            class_PQ = class_PQ.item()
            class_SQ = class_SQ.item()
            class_RQ = class_RQ.item()
            class_all_PQ = class_all_PQ.flatten().tolist()
            class_all_SQ = class_all_SQ.flatten().tolist()
            class_all_RQ = class_all_RQ.flatten().tolist()
            class_IoU = class_IoU.item()
            class_all_IoU = class_all_IoU.flatten().tolist()

            # # prepare results for print
            result_str = f'\n----------- frame: {f} ------------\n\n'
            result_str += '\n----------- TOTALS ------------\n\n'
            result_str += f'PQ: {class_PQ}\nSQ: {class_SQ}\nRQ: {class_RQ}\nIoU: {class_IoU}\n\n'
            # result_str += ('PQ: {}\n'.format(class_PQ))
            # result_str += ('SQ: {}\n'.format(class_SQ))
            # result_str += ('RQ: {}\n'.format(class_RQ))
            # result_str += ('IoU: {}\n'.format(class_IoU))

            for i, (pq, sq, rq, iou) in enumerate(zip(class_all_PQ, class_all_SQ, class_all_RQ, class_all_IoU)):
                result_str += f'Class {self.CLASSES[i]}\t PQ: {pq}\t SQ: {sq}\t RQ: {rq}\t IoU:{iou}\n'

            print(result_str)

            # prepare results for logger
            result_dict = {}

            result_dict["all"] = {}
            result_dict["all"]["PQ"] = class_PQ
            result_dict["all"]["SQ"] = class_SQ
            result_dict["all"]["RQ"] = class_RQ
            result_dict["all"]["IoU"] = class_IoU

            for idx, (pq, rq, sq, iou) in enumerate(zip(class_all_PQ, class_all_RQ, class_all_SQ, class_all_IoU)):
                class_str = self.CLASSES[idx]
                result_dict[class_str] = {}
                result_dict[class_str]["PQ"] = pq
                result_dict[class_str]["SQ"] = sq
                result_dict[class_str]["RQ"] = rq
                result_dict[class_str]["IoU"] = iou

            result_dict_list.append(result_dict)

        return result_str, result_dict_list

    def getPQ(self):
        # get PQ and first calculate for all classes
        sq_all = self.pan_iou.astype(np.double) / np.maximum(self.pan_tp.astype(np.double), self.eps)
        rq_all = self.pan_tp.astype(np.double) / np.maximum(
            self.pan_tp.astype(np.double) + 0.5 * self.pan_fp.astype(np.double) + 0.5 * self.pan_fn.astype(np.double),
            self.eps)
        pq_all = sq_all * rq_all

        # then do the REAL mean (no ignored classes)
        SQ = sq_all[self.include].mean()
        RQ = rq_all[self.include].mean()
        PQ = pq_all[self.include].mean()

        return PQ, SQ, RQ, pq_all, sq_all, rq_all

    def getSemIoU(self):
        tp, fp, fn = self.getSemIoUStats()
        # print(f"tp={tp}")
        # print(f"fp={fp}")
        # print(f"fn={fn}")
        intersection = tp
        union = tp + fp + fn
        union = np.maximum(union, self.eps)
        iou = intersection.astype(np.double) / union.astype(np.double)
        iou_mean = (intersection[self.include].astype(np.double) / union[self.include].astype(np.double)).mean()

        return iou_mean, iou

    def getSemIoUStats(self):
        # clone to avoid modifying the real deal
        conf = self.px_iou_conf_matrix.copy().astype(np.double)
        # remove fp from confusion on the ignore classes predictions
        # points that were predicted of another class, but were ignore
        # (corresponds to zeroing the cols of those classes, since the predictions
        # go on the rows)
        conf[:, self.ignore] = 0

        # get the clean stats
        tp = conf.diagonal()
        fp = conf.sum(axis=1) - tp
        fn = conf.sum(axis=0) - tp
        return tp, fp, fn

    def getSemAcc(self):
        tp, fp, fn = self.getSemIoUStats()
        total_tp = tp.sum()
        total = tp[self.include].sum() + fp[self.include].sum()
        total = np.maximum(total, self.eps)
        acc_mean = total_tp.astype(np.double) / total.astype(np.double)

        return acc_mean

    def batch_sem_IoU(self, pred_sem, gt_sem):

        idxs = np.stack([pred_sem, gt_sem], axis=0)

        # make confusion matrix (cols = gt, rows = pred)
        np.add.at(self.px_iou_conf_matrix, tuple(idxs), 1)

    def batch_panoptic(self, pred_sem, gt_sem, pred_inst, gt_inst):

        pred_inst = pred_inst + 1
        gt_inst = gt_inst + 1

        # only interested in points that are outside the void area (not in excluded classes)
        for cl in self.ignore:
        # make a mask for this class
            gt_not_in_excl_mask = gt_sem != cl
            # remove all other points
            pred_sem = pred_sem[gt_not_in_excl_mask]
            gt_sem = gt_sem[gt_not_in_excl_mask]
            pred_inst = pred_inst[gt_not_in_excl_mask]
            gt_inst = gt_inst[gt_not_in_excl_mask]

        # first step is to count intersections > 0.5 IoU for each class (except the ignored ones)
        for cl in self.include:
            # print("*"*80)
            # print("CLASS", cl.item())
            # get a class mask
            x_inst_in_cl_mask = pred_sem == cl
            y_inst_in_cl_mask = gt_sem == cl

            # get instance points in class (makes outside stuff 0)
            x_inst_in_cl = pred_inst * x_inst_in_cl_mask.astype(np.int64)
            y_inst_in_cl = gt_inst * y_inst_in_cl_mask.astype(np.int64)

            # generate the areas for each unique instance prediction
            unique_pred, counts_pred = np.unique(x_inst_in_cl[x_inst_in_cl > 0], return_counts=True)
            id2idx_pred = {id: idx for idx, id in enumerate(unique_pred)}
            matched_pred = np.array([False] * unique_pred.shape[0])
            # print("Unique predictions:", unique_pred)

            # generate the areas for each unique instance gt_np
            unique_gt, counts_gt = np.unique(y_inst_in_cl[y_inst_in_cl > 0], return_counts=True)
            id2idx_gt = {id: idx for idx, id in enumerate(unique_gt)}
            matched_gt = np.array([False] * unique_gt.shape[0])
            # print("Unique ground truth:", unique_gt)

            # generate intersection using offset
            valid_combos = np.logical_and(x_inst_in_cl > 0, y_inst_in_cl > 0)
            offset_combo = x_inst_in_cl[valid_combos] + self.offset * y_inst_in_cl[valid_combos]
            unique_combo, counts_combo = np.unique(offset_combo, return_counts=True)

            # generate an intersection map
            # count the intersections with over 0.5 IoU as TP
            gt_labels = unique_combo // self.offset
            pred_labels = unique_combo % self.offset
            gt_areas = np.array([counts_gt[id2idx_gt[id]] for id in gt_labels])
            pred_areas = np.array([counts_pred[id2idx_pred[id]] for id in pred_labels])
            intersections = counts_combo
            unions = gt_areas + pred_areas - intersections
            ious = intersections.astype(np.float) / unions.astype(np.float)


            tp_indexes = ious > 0.5
            self.pan_tp[cl] += np.sum(tp_indexes)
            self.pan_iou[cl] += np.sum(ious[tp_indexes])

            matched_gt[[id2idx_gt[id] for id in gt_labels[tp_indexes]]] = True
            matched_pred[[id2idx_pred[id] for id in pred_labels[tp_indexes]]] = True

            # count the FN
            self.pan_fn[cl] += np.sum(np.logical_and(counts_gt >= self.min_points, matched_gt == False))

            # count the FP
            self.pan_fp[cl] += np.sum(np.logical_and(counts_pred >= self.min_points, matched_pred == False))
