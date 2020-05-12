import numpy as np
import torch

from mmdet3d.ops.iou3d import iou3d_cuda


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
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
    """
    :param boxes3d: (N, 7) [x, y, z, w, l, h, ry] in LiDAR coords
    :return:
        boxes_bev: (N, 5) [x1, y1, x2, y2, ry]
    """
    boxes_bev = boxes3d.new(torch.Size((boxes3d.shape[0], 5)))

    cu, cv = boxes3d[:, 0], boxes3d[:, 1]
    half_l, half_w = boxes3d[:, 4] / 2, boxes3d[:, 3] / 2
    boxes_bev[:, 0], boxes_bev[:, 1] = cu - half_w, cv - half_l
    boxes_bev[:, 2], boxes_bev[:, 3] = cu + half_w, cv + half_l
    boxes_bev[:, 4] = boxes3d[:, 6]
    return boxes_bev


def boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    :param boxes_a: (N, 7) [x, y, z, w, l, h, ry]  in LiDAR
    :param boxes_b: (M, 7) [x, y, z, h, w, l, ry]
    :return:
        ans_iou: (M, N)
    """
    boxes_a_bev = boxes3d_to_bevboxes_lidar_torch(boxes_a)
    boxes_b_bev = boxes3d_to_bevboxes_lidar_torch(boxes_b)
    # height overlap
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5]).view(-1, 1)
    boxes_a_height_min = boxes_a[:, 2].view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5]).view(1, -1)
    boxes_b_height_min = boxes_b[:, 2].view(1, -1)

    # bev overlap
    overlaps_bev = boxes_a.new_zeros(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0])))  # (N, M)
    iou3d_cuda.boxes_overlap_bev_gpu(boxes_a_bev.contiguous(),
                                     boxes_b_bev.contiguous(), overlaps_bev)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)

    return iou3d


def get_iou_gpu(bb1, bb2):
    """ Compute IoU of two bounding boxes.
        ** Define your bod IoU function HERE **
    """

    bb1 = torch.from_numpy(bb1).float().cuda()
    bb2 = torch.from_numpy(bb2).float().cuda()

    iou3d = boxes_iou3d_gpu(bb1, bb2)
    return iou3d.cpu().numpy()


def eval_det_cls(pred,
                 gt,
                 ovthresh=None,
                 use_07_metric=False,
                 get_iou_func=get_iou_gpu):
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
    BB = np.array(BB)  # (nd,4 or 8,3 or 6)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, ...]
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
    pred, gt, ovthresh, use_07_metric, get_iou_func = arguments
    ret = eval_det_cls(pred, gt, ovthresh, use_07_metric, get_iou_func)
    return ret


def eval_det_multiprocessing(pred_all,
                             gt_all,
                             ovthresh=None,
                             use_07_metric=False,
                             get_iou_func=get_iou_gpu):
    """ Generic functions to compute precision/recall for object detection
        for multiple classes.
        Input:
            pred_all: map of {img_id: [(classname, bbox, score)]}
            gt_all: map of {img_id: [(classname, bbox)]}
            ovthresh: a list, iou threshold
            use_07_metric: bool, if true use VOC07 11 point method
        Output:
            rec: {classname: rec}
            prec: {classname: prec_all}
            ap: {classname: scalar}
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
    args = [(pred[classname], gt[classname], ovthresh, use_07_metric,
             get_iou_func) for classname in gt.keys() if classname in pred]
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
    ''' Calculating Average Precision '''

    def __init__(self, ap_iou_thresh=None, class2type_map=None):
        """
        Args:
            ap_iou_thresh: a list, which contains float between 0 and 1.0
                IoU threshold to judge whether a prediction is positive.
            class2type_map: [optional] dict {class_int:class_name}
        """
        self.ap_iou_thresh = ap_iou_thresh
        self.class2type_map = class2type_map
        self.reset()

    def step(self, det_infos, gt_infos):
        """ Accumulate one batch of prediction and groundtruth.

        Args:
            batch_pred_map_cls: a list of lists
                [[(pred_cls, pred_box_params, score),...],...]
            batch_gt_map_cls: a list of lists
                [[(gt_cls, gt_box_params),...],...]
                should have the same length with batch_pred_map_cls
                (batch_size)
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
        """ Use accumulated predictions and groundtruths to compute Average Precision.
        """
        recs, precs, aps = eval_det_multiprocessing(
            self.pred_map_cls,
            self.gt_map_cls,
            ovthresh=self.ap_iou_thresh,
            get_iou_func=get_iou_gpu)
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
                except KeyError:
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
    """ Flip X-right,Y-forward,Z-up to X-forward,Y-left,Z-up
    :param boxes3d_depth: (N, 7) [x, y, z, w, l, h, r] in depth coords
    :return:
        boxes3d_lidar: (N, 7) [x, y, z, l, h, w, r] in LiDAR coords
    """
    boxes3d_lidar = boxes3d.copy()
    boxes3d_lidar[..., [0, 1, 2, 3, 4, 5]] = boxes3d_lidar[...,
                                                           [1, 0, 2, 4, 3, 5]]
    boxes3d_lidar[..., 1] *= -1
    if mid_to_bottom:
        boxes3d_lidar[..., 2] -= boxes3d_lidar[..., 5] / 2
    return boxes3d_lidar


def scannet_eval(gt_annos, dt_annos, metric, class2type):
    for gt_anno in gt_annos:
        if gt_anno['gt_num'] != 0:
            # convert to lidar coor for evaluation
            bbox_lidar_bottom = boxes3d_depth_to_lidar(
                gt_anno['gt_boxes_upright_depth'], mid_to_bottom=True)
            gt_anno['gt_boxes_upright_depth'] = np.pad(bbox_lidar_bottom,
                                                       ((0, 0), (0, 1)),
                                                       'constant')
    ap_iou_thresholds = metric.AP_IOU_THRESHHOLDS
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
