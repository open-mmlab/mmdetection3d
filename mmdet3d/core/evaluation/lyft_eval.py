# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp

import mmcv
import numpy as np
from lyft_dataset_sdk.eval.detection.mAP_evaluation import (Box3D, get_ap,
                                                            get_class_names,
                                                            get_ious,
                                                            group_by_key,
                                                            wrap_in_box)
from mmcv.utils import print_log
from terminaltables import AsciiTable


def load_lyft_gts(lyft, data_root, eval_split, logger=None):
    """Loads ground truth boxes from database.

    Args:
        lyft (:obj:`LyftDataset`): Lyft class in the sdk.
        data_root (str): Root of data for reading splits.
        eval_split (str): Name of the split for evaluation.
        logger (logging.Logger | str, optional): Logger used for printing
        related information during evaluation. Default: None.

    Returns:
        list[dict]: List of annotation dictionaries.
    """
    split_scenes = mmcv.list_from_file(
        osp.join(data_root, f'{eval_split}.txt'))

    # Read out all sample_tokens in DB.
    sample_tokens_all = [s['token'] for s in lyft.sample]
    assert len(sample_tokens_all) > 0, 'Error: Database has no samples!'

    if eval_split == 'test':
        # Check that you aren't trying to cheat :)
        assert len(lyft.sample_annotation) > 0, \
            'Error: You are trying to evaluate on the test set \
             but you do not have the annotations!'

    sample_tokens = []
    for sample_token in sample_tokens_all:
        scene_token = lyft.get('sample', sample_token)['scene_token']
        scene_record = lyft.get('scene', scene_token)
        if scene_record['name'] in split_scenes:
            sample_tokens.append(sample_token)

    all_annotations = []

    print_log('Loading ground truth annotations...', logger=logger)
    # Load annotations and filter predictions and annotations.
    for sample_token in mmcv.track_iter_progress(sample_tokens):
        sample = lyft.get('sample', sample_token)
        sample_annotation_tokens = sample['anns']
        for sample_annotation_token in sample_annotation_tokens:
            # Get label name in detection task and filter unused labels.
            sample_annotation = \
                lyft.get('sample_annotation', sample_annotation_token)
            detection_name = sample_annotation['category_name']
            if detection_name is None:
                continue
            annotation = {
                'sample_token': sample_token,
                'translation': sample_annotation['translation'],
                'size': sample_annotation['size'],
                'rotation': sample_annotation['rotation'],
                'name': detection_name,
            }
            all_annotations.append(annotation)

    return all_annotations


def load_lyft_predictions(res_path):
    """Load Lyft predictions from json file.

    Args:
        res_path (str): Path of result json file recording detections.

    Returns:
        list[dict]: List of prediction dictionaries.
    """
    predictions = mmcv.load(res_path)
    predictions = predictions['results']
    all_preds = []
    for sample_token in predictions.keys():
        all_preds.extend(predictions[sample_token])
    return all_preds


def lyft_eval(lyft, data_root, res_path, eval_set, output_dir, logger=None):
    """Evaluation API for Lyft dataset.

    Args:
        lyft (:obj:`LyftDataset`): Lyft class in the sdk.
        data_root (str): Root of data for reading splits.
        res_path (str): Path of result json file recording detections.
        eval_set (str): Name of the split for evaluation.
        output_dir (str): Output directory for output json files.
        logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.

    Returns:
        dict[str, float]: The evaluation results.
    """
    # evaluate by lyft metrics
    gts = load_lyft_gts(lyft, data_root, eval_set, logger)
    predictions = load_lyft_predictions(res_path)

    class_names = get_class_names(gts)
    print('Calculating mAP@0.5:0.95...')

    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    metrics = {}
    average_precisions = \
        get_classwise_aps(gts, predictions, class_names, iou_thresholds)
    APs_data = [['IOU', 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]]

    mAPs = np.mean(average_precisions, axis=0)
    mAPs_cate = np.mean(average_precisions, axis=1)
    final_mAP = np.mean(mAPs)

    metrics['average_precisions'] = average_precisions.tolist()
    metrics['mAPs'] = mAPs.tolist()
    metrics['Final mAP'] = float(final_mAP)
    metrics['class_names'] = class_names
    metrics['mAPs_cate'] = mAPs_cate.tolist()

    APs_data = [['class', 'mAP@0.5:0.95']]
    for i in range(len(class_names)):
        row = [class_names[i], round(mAPs_cate[i], 3)]
        APs_data.append(row)
    APs_data.append(['Overall', round(final_mAP, 3)])
    APs_table = AsciiTable(APs_data, title='mAPs@0.5:0.95')
    APs_table.inner_footing_row_border = True
    print_log(APs_table.table, logger=logger)

    res_path = osp.join(output_dir, 'lyft_metrics.json')
    mmcv.dump(metrics, res_path)
    return metrics


def get_classwise_aps(gt, predictions, class_names, iou_thresholds):
    """Returns an array with an average precision per class.

    Note: Ground truth and predictions should have the following format.

    .. code-block::

    gt = [{
        'sample_token': '0f0e3ce89d2324d8b45aa55a7b4f8207
                         fbb039a550991a5149214f98cec136ac',
        'translation': [974.2811881299899, 1714.6815014457964,
                        -23.689857123368846],
        'size': [1.796, 4.488, 1.664],
        'rotation': [0.14882026466054782, 0, 0, 0.9888642620837121],
        'name': 'car'
    }]

    predictions = [{
        'sample_token': '0f0e3ce89d2324d8b45aa55a7b4f8207
                         fbb039a550991a5149214f98cec136ac',
        'translation': [971.8343488872263, 1713.6816097857359,
                        -25.82534357061308],
        'size': [2.519726579986132, 7.810161372666739, 3.483438286096803],
        'rotation': [0.10913582721095375, 0.04099572636992043,
                     0.01927712319721745, 1.029328402625659],
        'name': 'car',
        'score': 0.3077029437237213
    }]

    Args:
        gt (list[dict]): list of dictionaries in the format described below.
        predictions (list[dict]): list of dictionaries in the format
            described below.
        class_names (list[str]): list of the class names.
        iou_thresholds (list[float]): IOU thresholds used to calculate
            TP / FN

    Returns:
        np.ndarray: an array with an average precision per class.
    """
    assert all([0 <= iou_th <= 1 for iou_th in iou_thresholds])

    gt_by_class_name = group_by_key(gt, 'name')
    pred_by_class_name = group_by_key(predictions, 'name')

    average_precisions = np.zeros((len(class_names), len(iou_thresholds)))

    for class_id, class_name in enumerate(class_names):
        if class_name in pred_by_class_name:
            recalls, precisions, average_precision = get_single_class_aps(
                gt_by_class_name[class_name], pred_by_class_name[class_name],
                iou_thresholds)
            average_precisions[class_id, :] = average_precision

    return average_precisions


def get_single_class_aps(gt, predictions, iou_thresholds):
    """Compute recall and precision for all iou thresholds. Adapted from
    LyftDatasetDevkit.

    Args:
        gt (list[dict]): list of dictionaries in the format described above.
        predictions (list[dict]): list of dictionaries in the format
            described below.
        iou_thresholds (list[float]): IOU thresholds used to calculate
            TP / FN

    Returns:
        tuple[np.ndarray]: Returns (recalls, precisions, average precisions)
            for each class.
    """
    num_gts = len(gt)
    image_gts = group_by_key(gt, 'sample_token')
    image_gts = wrap_in_box(image_gts)

    sample_gt_checked = {
        sample_token: np.zeros((len(boxes), len(iou_thresholds)))
        for sample_token, boxes in image_gts.items()
    }

    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)

    # go down dets and mark TPs and FPs
    num_predictions = len(predictions)
    tps = np.zeros((num_predictions, len(iou_thresholds)))
    fps = np.zeros((num_predictions, len(iou_thresholds)))

    for prediction_index, prediction in enumerate(predictions):
        predicted_box = Box3D(**prediction)

        sample_token = prediction['sample_token']

        max_overlap = -np.inf
        jmax = -1

        if sample_token in image_gts:
            gt_boxes = image_gts[sample_token]
            # gt_boxes per sample
            gt_checked = sample_gt_checked[sample_token]
            # gt flags per sample
        else:
            gt_boxes = []
            gt_checked = None

        if len(gt_boxes) > 0:
            overlaps = get_ious(gt_boxes, predicted_box)

            max_overlap = np.max(overlaps)

            jmax = np.argmax(overlaps)

        for i, iou_threshold in enumerate(iou_thresholds):
            if max_overlap > iou_threshold:
                if gt_checked[jmax, i] == 0:
                    tps[prediction_index, i] = 1.0
                    gt_checked[jmax, i] = 1
                else:
                    fps[prediction_index, i] = 1.0
            else:
                fps[prediction_index, i] = 1.0

    # compute precision recall
    fps = np.cumsum(fps, axis=0)
    tps = np.cumsum(tps, axis=0)

    recalls = tps / float(num_gts)
    # avoid divide by zero in case the first detection
    # matches a difficult ground truth
    precisions = tps / np.maximum(tps + fps, np.finfo(np.float64).eps)

    aps = []
    for i in range(len(iou_thresholds)):
        recall = recalls[:, i]
        precision = precisions[:, i]
        assert np.all(0 <= recall) & np.all(recall <= 1)
        assert np.all(0 <= precision) & np.all(precision <= 1)
        ap = get_ap(recall, precision)
        aps.append(ap)

    aps = np.array(aps)

    return recalls, precisions, aps
