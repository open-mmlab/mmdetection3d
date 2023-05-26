# Copyright (c) OpenMMLab. All rights reserved.
# adapted from https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py # noqa
from copy import deepcopy

import numpy as np

from . import util_3d


def evaluate_matches(matches, class_labels, options):
    """Evaluate instance segmentation from matched gt and predicted instances
    for all scenes.

    Args:
        matches (dict): Contains gt2pred and pred2gt infos for every scene.
        class_labels (tuple[str]): Class names.
        options (dict): ScanNet evaluator options. See get_options.

    Returns:
        np.array: Average precision scores for all thresholds and categories.
    """
    overlaps = options['overlaps']
    min_region_sizes = [options['min_region_sizes'][0]]
    dist_threshes = [options['distance_threshes'][0]]
    dist_confs = [options['distance_confs'][0]]

    # results: class x overlap
    ap = np.zeros((len(dist_threshes), len(class_labels), len(overlaps)))
    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
            zip(min_region_sizes, dist_threshes, dist_confs)):
        for oi, overlap_th in enumerate(overlaps):
            pred_visited = {}
            for m in matches:
                for label_name in class_labels:
                    for p in matches[m]['pred'][label_name]:
                        if 'filename' in p:
                            pred_visited[p['filename']] = False
            for li, label_name in enumerate(class_labels):
                y_true = np.empty(0)
                y_score = np.empty(0)
                hard_false_negatives = 0
                has_gt = False
                has_pred = False
                for m in matches:
                    pred_instances = matches[m]['pred'][label_name]
                    gt_instances = matches[m]['gt'][label_name]
                    # filter groups in ground truth
                    gt_instances = [
                        gt for gt in gt_instances
                        if gt['instance_id'] >= 1000 and gt['vert_count'] >=
                        min_region_size and gt['med_dist'] <= distance_thresh
                        and gt['dist_conf'] >= distance_conf
                    ]
                    if gt_instances:
                        has_gt = True
                    if pred_instances:
                        has_pred = True

                    cur_true = np.ones(len(gt_instances))
                    cur_score = np.ones(len(gt_instances)) * (-float('inf'))
                    cur_match = np.zeros(len(gt_instances), dtype=bool)
                    # collect matches
                    for (gti, gt) in enumerate(gt_instances):
                        found_match = False
                        for pred in gt['matched_pred']:
                            # greedy assignments
                            if pred_visited[pred['filename']]:
                                continue
                            overlap = float(pred['intersection']) / (
                                gt['vert_count'] + pred['vert_count'] -
                                pred['intersection'])
                            if overlap > overlap_th:
                                confidence = pred['confidence']
                                # if already have a prediction for this gt,
                                # the prediction with the lower score is automatically a false positive # noqa
                                if cur_match[gti]:
                                    max_score = max(cur_score[gti], confidence)
                                    min_score = min(cur_score[gti], confidence)
                                    cur_score[gti] = max_score
                                    # append false positive
                                    cur_true = np.append(cur_true, 0)
                                    cur_score = np.append(cur_score, min_score)
                                    cur_match = np.append(cur_match, True)
                                # otherwise set score
                                else:
                                    found_match = True
                                    cur_match[gti] = True
                                    cur_score[gti] = confidence
                                    pred_visited[pred['filename']] = True
                        if not found_match:
                            hard_false_negatives += 1
                    # remove non-matched ground truth instances
                    cur_true = cur_true[cur_match]
                    cur_score = cur_score[cur_match]

                    # collect non-matched predictions as false positive
                    for pred in pred_instances:
                        found_gt = False
                        for gt in pred['matched_gt']:
                            overlap = float(gt['intersection']) / (
                                gt['vert_count'] + pred['vert_count'] -
                                gt['intersection'])
                            if overlap > overlap_th:
                                found_gt = True
                                break
                        if not found_gt:
                            num_ignore = pred['void_intersection']
                            for gt in pred['matched_gt']:
                                # group?
                                if gt['instance_id'] < 1000:
                                    num_ignore += gt['intersection']
                                # small ground truth instances
                                if gt['vert_count'] < min_region_size or gt[
                                        'med_dist'] > distance_thresh or gt[
                                            'dist_conf'] < distance_conf:
                                    num_ignore += gt['intersection']
                            proportion_ignore = float(
                                num_ignore) / pred['vert_count']
                            # if not ignored append false positive
                            if proportion_ignore <= overlap_th:
                                cur_true = np.append(cur_true, 0)
                                confidence = pred['confidence']
                                cur_score = np.append(cur_score, confidence)

                    # append to overall results
                    y_true = np.append(y_true, cur_true)
                    y_score = np.append(y_score, cur_score)

                # compute average precision
                if has_gt and has_pred:
                    # compute precision recall curve first

                    # sorting and cumsum
                    score_arg_sort = np.argsort(y_score)
                    y_score_sorted = y_score[score_arg_sort]
                    y_true_sorted = y_true[score_arg_sort]
                    y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                    # unique thresholds
                    (thresholds, unique_indices) = np.unique(
                        y_score_sorted, return_index=True)
                    num_prec_recall = len(unique_indices) + 1

                    # prepare precision recall
                    num_examples = len(y_score_sorted)
                    # follow https://github.com/ScanNet/ScanNet/pull/26 ? # noqa
                    num_true_examples = y_true_sorted_cumsum[-1] if len(
                        y_true_sorted_cumsum) > 0 else 0
                    precision = np.zeros(num_prec_recall)
                    recall = np.zeros(num_prec_recall)

                    # deal with the first point
                    y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                    # deal with remaining
                    for idx_res, idx_scores in enumerate(unique_indices):
                        cumsum = y_true_sorted_cumsum[idx_scores - 1]
                        tp = num_true_examples - cumsum
                        fp = num_examples - idx_scores - tp
                        fn = cumsum + hard_false_negatives
                        p = float(tp) / (tp + fp)
                        r = float(tp) / (tp + fn)
                        precision[idx_res] = p
                        recall[idx_res] = r

                    # first point in curve is artificial
                    precision[-1] = 1.
                    recall[-1] = 0.

                    # compute average of precision-recall curve
                    recall_for_conv = np.copy(recall)
                    recall_for_conv = np.append(recall_for_conv[0],
                                                recall_for_conv)
                    recall_for_conv = np.append(recall_for_conv, 0.)

                    stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5],
                                             'valid')
                    # integrate is now simply a dot product
                    ap_current = np.dot(precision, stepWidths)

                elif has_gt:
                    ap_current = 0.0
                else:
                    ap_current = float('nan')
                ap[di, li, oi] = ap_current
    return ap


def compute_averages(aps, options, class_labels):
    """Averages AP scores for all categories.

    Args:
        aps (np.array): AP scores for all thresholds and categories.
        options (dict): ScanNet evaluator options. See get_options.
        class_labels (tuple[str]): Class names.

    Returns:
        dict: Overall and per-category AP scores.
    """
    d_inf = 0
    o50 = np.where(np.isclose(options['overlaps'], 0.5))
    o25 = np.where(np.isclose(options['overlaps'], 0.25))
    o_all_but25 = np.where(
        np.logical_not(np.isclose(options['overlaps'], 0.25)))
    avg_dict = {}
    avg_dict['all_ap'] = np.nanmean(aps[d_inf, :, o_all_but25])
    avg_dict['all_ap_50%'] = np.nanmean(aps[d_inf, :, o50])
    avg_dict['all_ap_25%'] = np.nanmean(aps[d_inf, :, o25])
    avg_dict['classes'] = {}
    for (li, label_name) in enumerate(class_labels):
        avg_dict['classes'][label_name] = {}
        avg_dict['classes'][label_name]['ap'] = np.average(aps[d_inf, li,
                                                               o_all_but25])
        avg_dict['classes'][label_name]['ap50%'] = np.average(aps[d_inf, li,
                                                                  o50])
        avg_dict['classes'][label_name]['ap25%'] = np.average(aps[d_inf, li,
                                                                  o25])
    return avg_dict


def assign_instances_for_scan(pred_info, gt_ids, options, valid_class_ids,
                              class_labels, id_to_label):
    """Assign gt and predicted instances for a single scene.

    Args:
        pred_info (dict): Predicted masks, labels and scores.
        gt_ids (np.array): Ground truth instance masks.
        options (dict): ScanNet evaluator options. See get_options.
        valid_class_ids (tuple[int]): Ids of valid categories.
        class_labels (tuple[str]): Class names.
        id_to_label (dict[int, str]): Mapping of valid class id to class label.

    Returns:
        dict: Per class assigned gt to predicted instances.
        dict: Per class assigned predicted to gt instances.
    """
    # get gt instances
    gt_instances = util_3d.get_instances(gt_ids, valid_class_ids, class_labels,
                                         id_to_label)
    # associate
    gt2pred = deepcopy(gt_instances)
    for label in gt2pred:
        for gt in gt2pred[label]:
            gt['matched_pred'] = []
    pred2gt = {}
    for label in class_labels:
        pred2gt[label] = []
    num_pred_instances = 0
    # mask of void labels in the ground truth
    bool_void = np.logical_not(np.in1d(gt_ids // 1000, valid_class_ids))
    # go through all prediction masks
    for pred_mask_file in pred_info:
        label_id = int(pred_info[pred_mask_file]['label_id'])
        conf = pred_info[pred_mask_file]['conf']
        if not label_id in id_to_label:  # noqa E713
            continue
        label_name = id_to_label[label_id]
        # read the mask
        pred_mask = pred_info[pred_mask_file]['mask']
        if len(pred_mask) != len(gt_ids):
            raise ValueError('len(pred_mask) != len(gt_ids)')
        # convert to binary
        pred_mask = np.not_equal(pred_mask, 0)
        num = np.count_nonzero(pred_mask)
        if num < options['min_region_sizes'][0]:
            continue  # skip if empty

        pred_instance = {}
        pred_instance['filename'] = pred_mask_file
        pred_instance['pred_id'] = num_pred_instances
        pred_instance['label_id'] = label_id
        pred_instance['vert_count'] = num
        pred_instance['confidence'] = conf
        pred_instance['void_intersection'] = np.count_nonzero(
            np.logical_and(bool_void, pred_mask))

        # matched gt instances
        matched_gt = []
        # go through all gt instances with matching label
        for (gt_num, gt_inst) in enumerate(gt2pred[label_name]):
            intersection = np.count_nonzero(
                np.logical_and(gt_ids == gt_inst['instance_id'], pred_mask))
            if intersection > 0:
                gt_copy = gt_inst.copy()
                pred_copy = pred_instance.copy()
                gt_copy['intersection'] = intersection
                pred_copy['intersection'] = intersection
                matched_gt.append(gt_copy)
                gt2pred[label_name][gt_num]['matched_pred'].append(pred_copy)
        pred_instance['matched_gt'] = matched_gt
        num_pred_instances += 1
        pred2gt[label_name].append(pred_instance)

    return gt2pred, pred2gt


def scannet_eval(preds, gts, options, valid_class_ids, class_labels,
                 id_to_label):
    """Evaluate instance segmentation in ScanNet protocol.

    Args:
        preds (list[dict]): Per scene predictions of mask, label and
            confidence.
        gts (list[np.array]): Per scene ground truth instance masks.
        options (dict): ScanNet evaluator options. See get_options.
        valid_class_ids (tuple[int]): Ids of valid categories.
        class_labels (tuple[str]): Class names.
        id_to_label (dict[int, str]): Mapping of valid class id to class label.

    Returns:
        dict: Overall and per-category AP scores.
    """
    options = get_options(options)
    matches = {}
    for i, (pred, gt) in enumerate(zip(preds, gts)):
        matches_key = i
        # assign gt to predictions
        gt2pred, pred2gt = assign_instances_for_scan(pred, gt, options,
                                                     valid_class_ids,
                                                     class_labels, id_to_label)
        matches[matches_key] = {}
        matches[matches_key]['gt'] = gt2pred
        matches[matches_key]['pred'] = pred2gt

    ap_scores = evaluate_matches(matches, class_labels, options)
    avgs = compute_averages(ap_scores, options, class_labels)
    return avgs


def get_options(options=None):
    """Set ScanNet evaluator options.

    Args:
        options (dict, optional): Not default options. Default: None.

    Returns:
        dict: Updated options with all 4 keys.
    """
    assert options is None or isinstance(options, dict)
    _options = dict(
        overlaps=np.append(np.arange(0.5, 0.95, 0.05), 0.25),
        min_region_sizes=np.array([100]),
        distance_threshes=np.array([float('inf')]),
        distance_confs=np.array([-float('inf')]))
    if options is not None:
        _options.update(options)
    return _options
