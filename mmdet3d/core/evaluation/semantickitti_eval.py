# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np


class SemanticKITTIEval(object):

    def __init__(self, classes=None, offset=2**32, min_points=30, ignore=[0]):
        self.classes = classes
        self.ignore = np.array(ignore, dtype=np.int64)
        self.n_classes = len(classes)

        self.include = np.array(
            [n for n in range(self.n_classes) if n not in self.ignore],
            dtype=np.int64)
        self.offset = offset
        self.eps = 1e-15
        self.min_points = min_points
        self.reset()

    def reset(self):
        # general things
        # iou stuff
        self.px_iou_conf_matrix = np.zeros((self.n_classes, self.n_classes),
                                           dtype=np.int64)
        # panoptic stuff
        self.pan_tp = np.zeros(self.n_classes, dtype=np.int64)
        self.pan_iou = np.zeros(self.n_classes, dtype=np.double)
        self.pan_fp = np.zeros(self.n_classes, dtype=np.int64)
        self.pan_fn = np.zeros(self.n_classes, dtype=np.int64)

        self.evaluated_fnames = []

    def evaluate(self, gt_annos, result_files):
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

            PQ, SQ, RQ, all_PQ, all_SQ, all_RQ = self.getPQ()
            IoU, all_IoU = self.getSemIoU()

            # make python variables
            PQ = PQ.item()
            SQ = SQ.item()
            RQ = RQ.item()
            all_PQ = all_PQ.flatten().tolist()
            all_SQ = all_SQ.flatten().tolist()
            all_RQ = all_RQ.flatten().tolist()
            IoU = IoU.item()
            all_IoU = all_IoU.flatten().tolist()

            # # prepare results for print
            result_str = f'\n----------- frame: {f} ------------\n\n'
            result_str += '\n----------- TOTALS ------------\n\n'
            result_str += f'PQ: {PQ}\nSQ: {SQ}\nRQ: {RQ}\nIoU: {IoU}\n\n'

            for i, (pq, sq, rq,
                    iou) in enumerate(zip(all_PQ, all_SQ, all_RQ, all_IoU)):
                result_str += f'Class {self.classes[i]}\t\
                    PQ: {pq}\t SQ: {sq}\t RQ: {rq}\t IoU:{iou}\n'

            print(result_str)

            # prepare results for logger
            result_dict = {}

            result_dict['all'] = {}
            result_dict['all']['PQ'] = PQ
            result_dict['all']['SQ'] = SQ
            result_dict['all']['RQ'] = RQ
            result_dict['all']['IoU'] = IoU

            for idx, (pq, rq, sq,
                      iou) in enumerate(zip(all_PQ, all_RQ, all_SQ, all_IoU)):
                class_str = self.classes[idx]
                result_dict[class_str] = {}
                result_dict[class_str]['PQ'] = pq
                result_dict[class_str]['SQ'] = sq
                result_dict[class_str]['RQ'] = rq
                result_dict[class_str]['IoU'] = iou

            result_dict_list.append(result_dict)

        return result_str, result_dict_list

    def getPQ(self):
        # get PQ and first calculate for all classes
        sq_all = self.pan_iou.astype(np.double) / np.maximum(
            self.pan_tp.astype(np.double), self.eps)
        rq_all = self.pan_tp.astype(np.double) / np.maximum(
            self.pan_tp.astype(np.double) + 0.5 * self.pan_fp.astype(np.double)
            + 0.5 * self.pan_fn.astype(np.double), self.eps)
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
        iou_mean = (intersection[self.include].astype(np.double) /
                    union[self.include].astype(np.double)).mean()

        return iou_mean, iou

    def getSemIoUStats(self):
        # clone to avoid modifying the real deal
        conf = self.px_iou_conf_matrix.copy().astype(np.double)
        # remove fp from confusion on the ignore classes predictions
        # points that were predicted of another class, but were ignore
        # (corresponds to zeroing the cols of those classes,
        # since the predictions go on the rows)
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

        # only interested in points that are
        # outside the void area (not in excluded classes)
        for cl in self.ignore:
            # make a mask for this class
            gt_not_in_excl_mask = gt_sem != cl
            # remove all other points
            pred_sem = pred_sem[gt_not_in_excl_mask]
            gt_sem = gt_sem[gt_not_in_excl_mask]
            pred_inst = pred_inst[gt_not_in_excl_mask]
            gt_inst = gt_inst[gt_not_in_excl_mask]

        # first step is to count intersections > 0.5 IoU
        # for each class (except the ignored ones)
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
            unique_pred, counts_pred = np.unique(
                x_inst_in_cl[x_inst_in_cl > 0], return_counts=True)
            id2idx_pred = {id: idx for idx, id in enumerate(unique_pred)}
            matched_pred = np.array([False] * unique_pred.shape[0])
            # print("Unique predictions:", unique_pred)

            # generate the areas for each unique instance gt_np
            unique_gt, counts_gt = np.unique(
                y_inst_in_cl[y_inst_in_cl > 0], return_counts=True)
            id2idx_gt = {id: idx for idx, id in enumerate(unique_gt)}
            matched_gt = np.array([False] * unique_gt.shape[0])
            # print("Unique ground truth:", unique_gt)

            # generate intersection using offset
            valid_combos = np.logical_and(x_inst_in_cl > 0, y_inst_in_cl > 0)
            offset_combo = x_inst_in_cl[
                valid_combos] + self.offset * y_inst_in_cl[valid_combos]
            unique_combo, counts_combo = np.unique(
                offset_combo, return_counts=True)

            # generate an intersection map
            # count the intersections with over 0.5 IoU as TP
            gt_labels = unique_combo // self.offset
            pred_labels = unique_combo % self.offset
            gt_areas = np.array([counts_gt[id2idx_gt[id]] for id in gt_labels])
            pred_areas = np.array(
                [counts_pred[id2idx_pred[id]] for id in pred_labels])
            intersections = counts_combo
            unions = gt_areas + pred_areas - intersections
            ious = intersections.astype(np.float) / unions.astype(np.float)

            tp_indexes = ious > 0.5
            self.pan_tp[cl] += np.sum(tp_indexes)
            self.pan_iou[cl] += np.sum(ious[tp_indexes])

            matched_gt[[id2idx_gt[id] for id in gt_labels[tp_indexes]]] = True
            matched_pred[[id2idx_pred[id]
                          for id in pred_labels[tp_indexes]]] = True

            # count the FN
            self.pan_fn[cl] += np.sum(
                np.logical_and(counts_gt >= self.min_points,
                               matched_gt is False))

            # count the FP
            self.pan_fp[cl] += np.sum(
                np.logical_and(counts_pred >= self.min_points,
                               matched_pred is False))
