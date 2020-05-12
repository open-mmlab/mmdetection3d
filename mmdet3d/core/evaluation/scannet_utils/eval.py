import numpy as np
import torch

from mmdet3d.core.bbox.iou_calculators.iou3d_calculator import bbox_overlaps_3d


def voc_ap(rec, prec, use_07_metric=False):
    """ Voc AP

    Compute VOC AP given precision and recall.

    Args:
        rec (ndarray): Recall.
        prec (ndarray): Precision.
        use_07_metric (bool): Whether to use 07 metric.
            Default: False.

    Returns:
        ap (float): VOC AP.
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def boxes3d_to_bevboxes_lidar_torch(boxes3d):
    """Boxes3d to Bevboxes Lidar.

    Transform 3d boxes to bev boxes.

    Args:
        boxes3d (tensor): [x, y, z, w, l, h, ry] in LiDAR coords.

    Returns:
        boxes_bev (tensor): [x1, y1, x2, y2, ry].
    """
    boxes_bev = boxes3d.new(torch.Size((boxes3d.shape[0], 5)))

    cu, cv = boxes3d[:, 0], boxes3d[:, 1]
    half_l, half_w = boxes3d[:, 4] / 2, boxes3d[:, 3] / 2
    boxes_bev[:, 0], boxes_bev[:, 1] = cu - half_w, cv - half_l
    boxes_bev[:, 2], boxes_bev[:, 3] = cu + half_w, cv + half_l
    boxes_bev[:, 4] = boxes3d[:, 6]
    return boxes_bev


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


def eval_det_cls(pred, gt, ovthresh=None, use_07_metric=False):
    """ Generic functions to compute precision/recall for object detection
        for a single class.
        Input:
            pred: map of {img_id: [(bbox, score)]} where bbox is numpy array
            gt: map of {img_id: [bbox]}
            ovthresh: a list, iou threshold
            use_07_metric: bool, if True use VOC07 11 point method
        Output:
            rec: numpy array of length nd
            prec: numpy array of length nd
            ap: scalar, average precision
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
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
        ret.append((rec, prec, ap))

    return ret


def eval_det_cls_wrapper(arguments):
    pred, gt, ovthresh, use_07_metric = arguments
    ret = eval_det_cls(pred, gt, ovthresh, use_07_metric)
    return ret


def eval_det_multiprocessing(pred_all,
                             gt_all,
                             ovthresh=None,
                             use_07_metric=False):
    """ Evaluate Detection Multiprocessing.

    Generic functions to compute precision/recall for object detection
        for multiple classes.

    Args:
        pred_all (dict): map of {img_id: [(classname, bbox, score)]}.
        gt_all (dict): map of {img_id: [(classname, bbox)]}.
        ovthresh (List[float]): iou threshold.
            Default: None.
        use_07_metric (bool): if true use VOC07 11 point method.
            Default: False.
        get_iou_func (func): The function to get iou.
            Default: get_iou_gpu.

    Return:
        rec (dict): {classname: rec}.
        prec (dict): {classname: prec_all}.
        ap (dict): {classname: scalar}.
    """
    pred = {}  # map {classname: pred}
    gt = {}  # map {classname: gt}
    for img_id in pred_all.keys():
        for classname, bbox, score in pred_all[img_id]:
            if classname not in pred:
                pred[classname] = {}
            if img_id not in pred[classname]:
                pred[classname][img_id] = []
            if classname not in gt:
                gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            pred[classname][img_id].append((bbox, score))
    for img_id in gt_all.keys():
        for classname, bbox in gt_all[img_id]:
            if classname not in gt:
                gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            gt[classname][img_id].append(bbox)

    ret_values = []
    args = [(pred[classname], gt[classname], ovthresh, use_07_metric)
            for classname in gt.keys() if classname in pred]
    rec = [{} for i in ovthresh]
    prec = [{} for i in ovthresh]
    ap = [{} for i in ovthresh]
    for arg in args:
        ret_values.append(eval_det_cls_wrapper(arg))

    for i, classname in enumerate(gt.keys()):
        for iou_idx, thresh in enumerate(ovthresh):
            if classname in pred:
                rec[iou_idx][classname], prec[iou_idx][classname], ap[iou_idx][
                    classname] = ret_values[i][iou_idx]
            else:
                rec[iou_idx][classname] = 0
                prec[iou_idx][classname] = 0
                ap[iou_idx][classname] = 0
        # print(classname, ap[classname])

    return rec, prec, ap


class APCalculator(object):
    """AP Calculator.

    Calculating Average Precision.

    Args:
        ap_iou_thresh (List[float]): a list,
            which contains float between 0 and 1.0
            IoU threshold to judge whether a prediction is positive.
        class2type_map (dict): {class_int:class_name}.
    """

    def __init__(self, ap_iou_thresh=None, class2type_map=None):

        self.ap_iou_thresh = ap_iou_thresh
        self.class2type_map = class2type_map
        self.reset()

    def step(self, det_infos, gt_infos):
        """ Step.

        Accumulate one batch of prediction and groundtruth.

        Args:
            batch_pred_map_cls (List[List]): a list of lists
                [[(pred_cls, pred_box_params, score),...],...].
            batch_gt_map_cls (List[List]): a list of lists
                [[(gt_cls, gt_box_params),...],...].
        """
        # cache pred infos
        for batch_pred_map_cls in det_infos:
            for i in range(len(batch_pred_map_cls)):
                self.pred_map_cls[self.scan_cnt] = batch_pred_map_cls[i]
                self.scan_cnt += 1

        # cacge gt infos
        self.scan_cnt = 0
        for gt_info in gt_infos:
            cur_gt = list()
            for n in range(gt_info['gt_num']):
                cur_gt.append((gt_info['class'][n],
                               gt_info['gt_boxes_upright_depth'][n]))
            self.gt_map_cls[self.scan_cnt] = cur_gt
            self.scan_cnt += 1

    def compute_metrics(self):

        recs, precs, aps = eval_det_multiprocessing(
            self.pred_map_cls, self.gt_map_cls, ovthresh=self.ap_iou_thresh)
        ret = []
        for i, iou_thresh in enumerate(self.ap_iou_thresh):
            ret_dict = {}
            rec, _, ap = recs[i], precs[i], aps[i]
            for key in sorted(ap.keys()):
                clsname = self.class2type_map[
                    key] if self.class2type_map else str(key)
                ret_dict['%s Average Precision %d' %
                         (clsname, iou_thresh * 100)] = ap[key]
            ret_dict['mAP%d' % (iou_thresh * 100)] = np.mean(list(ap.values()))
            rec_list = []
            for key in sorted(ap.keys()):
                clsname = self.class2type_map[
                    key] if self.class2type_map else str(key)
                try:
                    ret_dict['%s Recall %d' %
                             (clsname, iou_thresh * 100)] = rec[key][-1]
                    rec_list.append(rec[key][-1])
                except TypeError:
                    ret_dict['%s Recall %d' % (clsname, iou_thresh * 100)] = 0
                    rec_list.append(0)
            ret_dict['AR%d' % (iou_thresh * 100)] = np.mean(rec_list)
            ret.append(ret_dict)
        return ret

    def reset(self):
        self.gt_map_cls = {}  # {scan_id: [(classname, bbox)]}
        self.pred_map_cls = {}  # {scan_id: [(classname, bbox, score)]}
        self.scan_cnt = 0


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


def scannet_eval(gt_annos, dt_annos, metric, class2type):
    """Scannet Evaluation.

    Evaluate the result of the detection.

    Args:
        gt_annos (List): GT annotations.
        dt_annos (List): Detection annotations.
        metric (dict): AP IoU thresholds.
        class2type (dict): {class: type}.

    Return:
        result_str (str): Result string.
        metrics_dict (dict): Result.
    """

    for gt_anno in gt_annos:
        if gt_anno['gt_num'] != 0:
            # convert to lidar coor for evaluation
            bbox_lidar_bottom = boxes3d_depth_to_lidar(
                gt_anno['gt_boxes_upright_depth'], mid_to_bottom=True)
            gt_anno['gt_boxes_upright_depth'] = np.pad(bbox_lidar_bottom,
                                                       ((0, 0), (0, 1)),
                                                       'constant')
    ap_iou_thresholds = metric['AP_IOU_THRESHHOLDS']
    ap_calculator = APCalculator(ap_iou_thresholds, class2type)
    ap_calculator.step(dt_annos, gt_annos)
    result_str = str()
    result_str += 'mAP'
    metrics_dict = {}
    metrics = ap_calculator.compute_metrics()
    for i, iou_threshold in enumerate(ap_iou_thresholds):
        metrics_tmp = metrics[i]
        metrics_dict.update(metrics_tmp)
        result_str += '(%.2f):%s   ' % (iou_threshold,
                                        metrics_dict['mAP%d' %
                                                     (iou_threshold * 100)])
    return result_str, metrics_dict
