# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmengine.logging import print_log


class EvalPanoptic:
    r"""Evaluate panoptic results for Semantickitti and NuScenes.
    Please refer to the `semantic kitti api
    <https://github.com/PRBonn/semantic-kitti-api/>`_ for more details
    Args:
        classes (list): Classes used in the dataset.
        things (list): Things classes used in the dataset.
        stuff (list): Stuff classes used in the dataset.
        min_points (int): Minimum point number of object to be
            counted as ground truth in evaluation.
        offset (int): Offset for instance ids to concat with
            semantic labels.
        label2cat (tuple): Map from label to category.
        ignore_index (list[int]): Ignored classes in evaluation.
        logger (logging.Logger | str, optional): Logger used for printing.
            Default: None.
    """

    def __init__(self,
                 classes,
                 things,
                 stuff,
                 min_points,
                 offset,
                 label2cat,
                 ignore_index,
                 logger=None):
        self.classes = classes
        self.things = things
        self.stuff = stuff
        self.ignore_index = np.array(ignore_index, dtype=int)
        self.n_classes = len(classes)
        self.label2cat = label2cat
        self.logger = logger
        self.include = np.array(
            [n for n in range(self.n_classes) if n not in self.ignore_index],
            dtype=int)
        self.offset = offset
        self.eps = 1e-15
        self.min_points = min_points
        self.reset()

    def reset(self):
        """Reset class variables."""
        # general things
        # iou stuff
        self.px_iou_conf_matrix = np.zeros((self.n_classes, self.n_classes),
                                           dtype=int)
        # panoptic stuff
        self.pan_tp = np.zeros(self.n_classes, dtype=int)
        self.pan_iou = np.zeros(self.n_classes, dtype=np.double)
        self.pan_fp = np.zeros(self.n_classes, dtype=int)
        self.pan_fn = np.zeros(self.n_classes, dtype=int)

        self.evaluated_fnames = []

    def evaluate(self, gt_labels, seg_preds):
        """Evaluate the predictions.

        Args:
            gt_labels (list[torch.Tensor]): Ground Truth.
            seg_preds (list[torch.Tensor]): Predictions.
        Returns:
            tuple[dict]: Results of strings and evaluation metrics.
        """
        assert len(seg_preds) == len(gt_labels)
        for f in range(len(seg_preds)):
            gt_semantic_seg = gt_labels[f]['pts_semantic_mask'].astype(int)
            gt_instance_seg = gt_labels[f]['pts_instance_mask'].astype(int)
            pred_semantic_seg = seg_preds[f]['pts_semantic_mask'].astype(int)
            pred_instance_seg = seg_preds[f]['pts_instance_mask'].astype(int)

            self.add_semantic_sample(pred_semantic_seg, gt_semantic_seg)
            self.add_panoptic_sample(pred_semantic_seg, gt_semantic_seg,
                                     pred_instance_seg, gt_instance_seg)

        result_dicts = self.print_results(self.logger)

        return result_dicts

    def print_results(self, logger=None):
        """Print results.

        Args:
            logger (logging.Logger | str, optional): The way to print the
                summary. See `mmengine.logging.print_log()` for details.
                Default: None.

        Returns:
            dict[str, float]: Dict of results.
        """
        pq, sq, rq, all_pq, all_sq, all_rq = self.get_pq()
        miou, iou = self.get_iou()

        # now make a nice dictionary
        output_dict = {}

        # make python variables
        pq = pq.item()
        sq = sq.item()
        rq = rq.item()
        all_pq = all_pq.flatten().tolist()
        all_sq = all_sq.flatten().tolist()
        all_rq = all_rq.flatten().tolist()
        miou = miou.item()
        iou = iou.flatten().tolist()

        output_dict['all'] = {}
        output_dict['all']['pq'] = pq
        output_dict['all']['sq'] = sq
        output_dict['all']['rq'] = rq
        output_dict['all']['miou'] = miou
        for idx, (_pq, _sq, _rq,
                  _iou) in enumerate(zip(all_pq, all_sq, all_rq, iou)):
            class_str = self.classes[idx]
            output_dict[class_str] = {}
            output_dict[class_str]['pq'] = _pq
            output_dict[class_str]['sq'] = _sq
            output_dict[class_str]['rq'] = _rq
            output_dict[class_str]['miou'] = _iou

        pq_dagger = np.mean(
            [float(output_dict[c]['pq']) for c in self.things] +
            [float(output_dict[c]['miou']) for c in self.stuff])

        pq_things = np.mean([float(output_dict[c]['pq']) for c in self.things])
        rq_things = np.mean([float(output_dict[c]['rq']) for c in self.things])
        sq_things = np.mean([float(output_dict[c]['sq']) for c in self.things])

        pq_stuff = np.mean([float(output_dict[c]['pq']) for c in self.stuff])
        rq_stuff = np.mean([float(output_dict[c]['rq']) for c in self.stuff])
        sq_stuff = np.mean([float(output_dict[c]['sq']) for c in self.stuff])

        result_dicts = {}
        result_dicts['pq'] = float(pq)
        result_dicts['pq_dagger'] = float(pq_dagger)
        result_dicts['sq_mean'] = float(sq)
        result_dicts['rq_mean'] = float(rq)
        result_dicts['miou'] = float(miou)
        result_dicts['pq_stuff'] = float(pq_stuff)
        result_dicts['rq_stuff'] = float(rq_stuff)
        result_dicts['sq_stuff'] = float(sq_stuff)
        result_dicts['pq_things'] = float(pq_things)
        result_dicts['rq_things'] = float(rq_things)
        result_dicts['sq_things'] = float(sq_things)

        if logger is not None:
            print_log('|        |   IoU   |   PQ   |   RQ   |  SQ   |', logger)
            for k, v in output_dict.items():
                print_log(
                    '|{}| {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(
                        k.ljust(8)[-8:], v['miou'], v['pq'], v['rq'], v['sq']),
                    logger)
            print_log('True Positive: ', logger)
            print_log('\t|\t'.join([str(x) for x in self.pan_tp]), logger)
            print_log('False Positive: ')
            print_log('\t|\t'.join([str(x) for x in self.pan_fp]), logger)
            print_log('False Negative: ')
            print_log('\t|\t'.join([str(x) for x in self.pan_fn]), logger)

        else:
            print('|        |   IoU   |   PQ   |   RQ   |  SQ   |')
            for k, v in output_dict.items():
                print('|{}| {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(
                    k.ljust(8)[-8:], v['miou'], v['pq'], v['rq'], v['sq']))
            print('True Positive: ')
            print('\t|\t'.join([str(x) for x in self.pan_tp]))
            print('False Positive: ')
            print('\t|\t'.join([str(x) for x in self.pan_fp]))
            print('False Negative: ')
            print('\t|\t'.join([str(x) for x in self.pan_fn]))

        return result_dicts

    def get_pq(self):
        """Get results of PQ metric.

        Returns:
            tuple(numpy.array): PQ, SQ, RQ of each class and all class.
        """
        # get PQ and first calculate for all classes
        sq_all = self.pan_iou.astype(np.double) / np.maximum(
            self.pan_tp.astype(np.double), self.eps)
        rq_all = self.pan_tp.astype(np.double) / np.maximum(
            self.pan_tp.astype(np.double) + 0.5 * self.pan_fp.astype(np.double)
            + 0.5 * self.pan_fn.astype(np.double), self.eps)
        pq_all = sq_all * rq_all

        # then do the REAL mean (no ignored classes)
        sq = sq_all[self.include].mean()
        rq = rq_all[self.include].mean()
        pq = pq_all[self.include].mean()

        return (pq, sq, rq, pq_all, sq_all, rq_all)

    def get_iou(self):
        """Get results of IOU metric.

        Returns:
            tuple(numpy.array): iou of all class and each class.
        """
        tp, fp, fn = self.get_iou_stats()
        intersection = tp
        union = tp + fp + fn
        union = np.maximum(union, self.eps)
        iou = intersection.astype(np.double) / union.astype(np.double)
        iou_mean = (intersection[self.include].astype(np.double) /
                    union[self.include].astype(np.double)).mean()

        return iou_mean, iou

    def get_iou_stats(self):
        """Get IOU statistics of TP, FP and FN."""
        # copy to avoid modifying the real deal
        conf = self.px_iou_conf_matrix.copy().astype(np.double)
        # remove fp from confusion on the ignore classes predictions
        # points that were predicted of another class, but were ignore
        # (corresponds to zeroing the cols of those classes,
        # since the predictions go on the rows)
        conf[:, self.ignore_index] = 0

        # get the clean stats
        tp = conf.diagonal()
        fp = conf.sum(axis=1) - tp
        fn = conf.sum(axis=0) - tp
        return tp, fp, fn

    def add_semantic_sample(self, semantic_preds, gt_semantics):
        """Add one batch of semantic predictions and ground truths.

        Args:
            semantic_preds (torch.Tensor): Semantic predictions.
            gt_semantics (torch.Tensor): Semantic ground truths.
        """
        idxs = np.stack([semantic_preds, gt_semantics], axis=0)
        # make confusion matrix (cols = gt, rows = pred)
        np.add.at(self.px_iou_conf_matrix, tuple(idxs), 1)

    def add_panoptic_sample(self, semantic_preds, gt_semantics, instance_preds,
                            gt_instances):
        """Add one sample of panoptic predictions and ground truths for
        evaluation.

        Args:
            semantic_preds (torch.Tensor): Semantic predictions.
            gt_semantics (torch.Tensor): Semantic ground truths.
            instance_preds (torch.Tensor): Instance predictions.
            gt_instances (torch.Tensor): Instance ground truths.
        """
        # avoid zero (ignored label)
        instance_preds = instance_preds + 1
        gt_instances = gt_instances + 1

        # only interested in points that are
        # outside the void area (not in excluded classes)
        for cl in self.ignore_index:
            # make a mask for this class
            gt_not_in_excl_mask = gt_semantics != cl
            # remove all other points
            semantic_preds = semantic_preds[gt_not_in_excl_mask]
            gt_semantics = gt_semantics[gt_not_in_excl_mask]
            instance_preds = instance_preds[gt_not_in_excl_mask]
            gt_instances = gt_instances[gt_not_in_excl_mask]

        # first step is to count intersections > 0.5 IoU
        # for each class (except the ignored ones)
        for cl in self.include:
            # get a class mask
            pred_inst_in_cl_mask = semantic_preds == cl
            gt_inst_in_cl_mask = gt_semantics == cl

            # get instance points in class (makes outside stuff 0)
            pred_inst_in_cl = instance_preds * pred_inst_in_cl_mask.astype(int)
            gt_inst_in_cl = gt_instances * gt_inst_in_cl_mask.astype(int)

            # generate the areas for each unique instance prediction
            unique_pred, counts_pred = np.unique(
                pred_inst_in_cl[pred_inst_in_cl > 0], return_counts=True)
            id2idx_pred = {id: idx for idx, id in enumerate(unique_pred)}
            matched_pred = np.array([False] * unique_pred.shape[0])

            # generate the areas for each unique instance gt_np
            unique_gt, counts_gt = np.unique(
                gt_inst_in_cl[gt_inst_in_cl > 0], return_counts=True)
            id2idx_gt = {id: idx for idx, id in enumerate(unique_gt)}
            matched_gt = np.array([False] * unique_gt.shape[0])

            # generate intersection using offset
            valid_combos = np.logical_and(pred_inst_in_cl > 0,
                                          gt_inst_in_cl > 0)
            offset_combo = pred_inst_in_cl[
                valid_combos] + self.offset * gt_inst_in_cl[valid_combos]
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
            ious = intersections.astype(float) / unions.astype(float)

            tp_indexes = ious > 0.5
            self.pan_tp[cl] += np.sum(tp_indexes)
            self.pan_iou[cl] += np.sum(ious[tp_indexes])

            matched_gt[[id2idx_gt[id] for id in gt_labels[tp_indexes]]] = True
            matched_pred[[id2idx_pred[id]
                          for id in pred_labels[tp_indexes]]] = True

            # count the FN
            if len(counts_gt) > 0:
                self.pan_fn[cl] += np.sum(
                    np.logical_and(counts_gt >= self.min_points, ~matched_gt))

            # count the FP
            if len(matched_pred) > 0:
                self.pan_fp[cl] += np.sum(
                    np.logical_and(counts_pred >= self.min_points,
                                   ~matched_pred))
