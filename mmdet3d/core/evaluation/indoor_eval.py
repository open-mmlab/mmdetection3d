import numpy as np
import torch

from mmdet3d.core.bbox.iou_calculators.iou3d_calculator import bbox_overlaps_3d


def boxes3d_depth_to_lidar(boxes3d, mid_to_bottom=True):
    """ Boxes3d Depth to Lidar.

    Flip X-right,Y-forward,Z-up to X-forward,Y-left,Z-up.

    Args:
        boxes3d (ndarray): (N, 7) [x, y, z, w, l, h, r] in depth coords.

    Return:
        boxes3d_lidar (ndarray): (N, 7) [x, y, z, l, h, w, r] in LiDAR coords.
    """
    boxes3d_lidar = boxes3d.copy()
    boxes3d_lidar[..., [0, 1, 2, 3, 4, 5]] = boxes3d_lidar[...,
                                                           [1, 0, 2, 4, 3, 5]]
    boxes3d_lidar[..., 1] *= -1
    if mid_to_bottom:
        boxes3d_lidar[..., 2] -= boxes3d_lidar[..., 5] / 2
    return boxes3d_lidar


def get_iou_gpu(bb1, bb2):
    """Get IoU.

    Compute IoU of two bounding boxes.

    Args:
        bb1 (ndarray): [x, y, z, w, l, h, ry] in LiDAR.
        bb2 (ndarray): [x, y, z, h, w, l, ry] in LiDAR.

    Returns:
        ans_iou (tensor): The answer of IoU.
    """

    bb1 = torch.from_numpy(bb1).float().cuda()
    bb2 = torch.from_numpy(bb2).float().cuda()
    iou3d = bbox_overlaps_3d(bb1, bb2, mode='iou', coordinate='lidar')
    return iou3d.cpu().numpy()


def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls (ndarray): shape (num_scales, num_dets) or (num_dets, )
        precisions (ndarray): shape (num_scales, num_dets) or (num_dets, )
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or ndarray: calculated average precision
    """
    if recalls.ndim == 1:
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
            ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    return ap


def eval_det_cls(pred, gt, ovthresh=None):
    """Generic functions to compute precision/recall for object detection
        for a single class.

    Args:
        pred (dict): map of {img_id: [(bbox, score)]} where bbox is numpy array
        gt (dict): map of {img_id: [bbox]}
        ovthresh (List[float]): a list, iou threshold

    Return:
        ndarray: numpy array of length nd
        ndarray: numpy array of length nd
        float: scalar, average precision
    """

    # construct gt objects
    class_recs = {}  # {img_id: {'bbox': bbox list, 'det': matched list}}
    npos = 0
    for img_id in gt.keys():
        bbox = np.array(gt[img_id])
        det = [[False] * len(bbox) for i in ovthresh]
        npos += len(bbox)
        class_recs[img_id] = {'bbox': bbox, 'det': det}
    # pad empty list to all other imgids
    for img_id in pred.keys():
        if img_id not in gt:
            class_recs[img_id] = {'bbox': np.array([]), 'det': []}

    # construct dets
    image_ids = []
    confidence = []
    BB = []
    ious = []
    for img_id in pred.keys():
        cur_num = len(pred[img_id])
        if cur_num == 0:
            continue
        BB_cur = np.zeros((cur_num, 7))  # hard code
        box_idx = 0
        for box, score in pred[img_id]:
            image_ids.append(img_id)
            confidence.append(score)
            BB.append(box)
            BB_cur[box_idx] = box
            box_idx += 1
        gt_cur = class_recs[img_id]['bbox'].astype(float)
        if len(gt_cur) > 0:
            # calculate iou in each image
            iou_cur = get_iou_gpu(BB_cur, gt_cur)
            for i in range(cur_num):
                ious.append(iou_cur[i])
        else:
            for i in range(cur_num):
                ious.append(np.zeros(1))

    confidence = np.array(confidence)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    image_ids = [image_ids[x] for x in sorted_ind]
    ious = [ious[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp_thresh = [np.zeros(nd) for i in ovthresh]
    fp_thresh = [np.zeros(nd) for i in ovthresh]
    for d in range(nd):
        R = class_recs[image_ids[d]]
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)
        cur_iou = ious[d]

        if BBGT.size > 0:
            # compute overlaps
            for j in range(BBGT.shape[0]):
                # iou = get_iou_main(get_iou_func, (bb, BBGT[j,...]))
                iou = cur_iou[j]
                if iou > ovmax:
                    ovmax = iou
                    jmax = j

        for iou_idx, thresh in enumerate(ovthresh):
            if ovmax > thresh:
                if not R['det'][iou_idx][jmax]:
                    tp_thresh[iou_idx][d] = 1.
                    R['det'][iou_idx][jmax] = 1
                else:
                    fp_thresh[iou_idx][d] = 1.
            else:
                fp_thresh[iou_idx][d] = 1.

    ret = []
    for iou_idx, thresh in enumerate(ovthresh):
        # compute precision recall
        fp = np.cumsum(fp_thresh[iou_idx])
        tp = np.cumsum(tp_thresh[iou_idx])
        recall = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = average_precision(recall, precision)
        ret.append((recall, precision, ap))

    return ret


def eval_map_recall(det_infos, gt_infos, ovthresh=None):
    """Evaluate mAP and Recall.

    Generic functions to compute precision/recall for object detection
        for multiple classes.

    Args:
        det_infos (List): Label, bbox and score of the detection result.
        gt_infos (List): Label, bbox of the groundtruth.
        ovthresh (List[float]): iou threshold.
            Default: None.

    Return:
        dict: {classname: rec}.
        dict: {classname: prec_all}.
        dict: {classname: scalar}.
    """
    pred_all = {}
    scan_cnt = 0
    for batch_pred_map_cls in det_infos:
        for i in range(len(batch_pred_map_cls)):
            pred_all[scan_cnt] = batch_pred_map_cls[i]
            scan_cnt += 1

    pred = {}  # map {classname: pred}
    gt = {}  # map {classname: gt}
    for img_id in pred_all.keys():
        for label, bbox, score in pred_all[img_id]:
            if label not in pred:
                pred[int(label)] = {}
            if img_id not in pred[label]:
                pred[int(label)][img_id] = []
            if label not in gt:
                gt[int(label)] = {}
            if img_id not in gt[label]:
                gt[int(label)][img_id] = []
            pred[int(label)][img_id].append((bbox, score))

    for img_id in range(len(gt_infos)):
        for label, bbox in gt_infos[img_id]:
            if label not in gt:
                gt[label] = {}
            if img_id not in gt[label]:
                gt[label][img_id] = []
            gt[label][img_id].append(bbox)

    ret_values = []
    for classname in gt.keys():
        if classname in pred:
            ret_values.append(
                eval_det_cls(pred[classname], gt[classname], ovthresh))
    recall = [{} for i in ovthresh]
    precision = [{} for i in ovthresh]
    ap = [{} for i in ovthresh]

    for i, label in enumerate(gt.keys()):
        for iou_idx, thresh in enumerate(ovthresh):
            if label in pred:
                recall[iou_idx][label], precision[iou_idx][label], ap[iou_idx][
                    label] = ret_values[i][iou_idx]
            else:
                recall[iou_idx][label] = 0
                precision[iou_idx][label] = 0
                ap[iou_idx][label] = 0

    return recall, precision, ap


def indoor_eval(gt_annos, dt_annos, metric, label2cat):
    """Scannet Evaluation.

    Evaluate the result of the detection.

    Args:
        gt_annos (List): GT annotations.
        dt_annos (List): Detection annotations.
        metric (List[float]): AP IoU thresholds.
        label2cat (dict): {label: cat}.

    Return:
        dict: Dict of results.
    """
    gt_infos = []
    for gt_anno in gt_annos:
        if gt_anno['gt_num'] != 0:
            # convert to lidar coor for evaluation
            bbox_lidar_bottom = boxes3d_depth_to_lidar(
                gt_anno['gt_boxes_upright_depth'], mid_to_bottom=True)
            if bbox_lidar_bottom.shape[-1] == 6:
                bbox_lidar_bottom = np.pad(bbox_lidar_bottom, ((0, 0), (0, 1)),
                                           'constant')
            gt_info_temp = []
            for i in range(gt_anno['gt_num']):
                gt_info_temp.append(
                    [gt_anno['class'][i], bbox_lidar_bottom[i]])
            gt_infos.append(gt_info_temp)

    result_str = str()
    result_str += 'mAP'
    rec, prec, ap = eval_map_recall(dt_annos, gt_infos, metric)
    ret_dict = {}
    for i, iou_thresh in enumerate(metric):
        for label in ap[i].keys():
            ret_dict[f'{label2cat[label]}_AP_{int(iou_thresh * 100)}'] = ap[i][
                label]
        ret_dict[f'mAP_{int(iou_thresh * 100)}'] = sum(ap[i].values()) / len(
            ap[i])
        for label in rec[i].keys():
            ret_dict[f'{label2cat[label]}_rec_{int(iou_thresh * 100)}'] = rec[
                i][label]
        ret_dict[f'mAR_{int(iou_thresh * 100)}'] = sum(rec[i].values()) / len(
            rec[i])
    return ret_dict
