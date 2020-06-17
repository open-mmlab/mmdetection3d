import torch

from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu, nms_normal_gpu


def box3d_multiclass_nms(mlvl_bboxes,
                         mlvl_bboxes_for_nms,
                         mlvl_scores,
                         score_thr,
                         max_num,
                         cfg,
                         mlvl_dir_scores=None):
    # do multi class nms
    # the fg class id range: [0, num_classes-1]
    num_classes = mlvl_scores.shape[1] - 1
    bboxes = []
    scores = []
    labels = []
    dir_scores = []
    for i in range(0, num_classes):
        # get bboxes and scores of this class
        cls_inds = mlvl_scores[:, i] > score_thr
        if not cls_inds.any():
            continue

        _scores = mlvl_scores[cls_inds, i]
        _bboxes_for_nms = mlvl_bboxes_for_nms[cls_inds, :]

        if cfg.use_rotate_nms:
            nms_func = nms_gpu
        else:
            nms_func = nms_normal_gpu

        selected = nms_func(_bboxes_for_nms, _scores, cfg.nms_thr)
        _mlvl_bboxes = mlvl_bboxes[cls_inds, :]
        bboxes.append(_mlvl_bboxes[selected])
        scores.append(_scores[selected])
        cls_label = mlvl_bboxes.new_full((len(selected), ),
                                         i,
                                         dtype=torch.long)
        labels.append(cls_label)

        if mlvl_dir_scores is not None:
            _mlvl_dir_scores = mlvl_dir_scores[cls_inds]
            dir_scores.append(_mlvl_dir_scores[selected])

    if bboxes:
        bboxes = torch.cat(bboxes, dim=0)
        scores = torch.cat(scores, dim=0)
        labels = torch.cat(labels, dim=0)
        if mlvl_dir_scores is not None:
            dir_scores = torch.cat(dir_scores, dim=0)
        if bboxes.shape[0] > max_num:
            _, inds = scores.sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            scores = scores[inds]
            if mlvl_dir_scores is not None:
                dir_scores = dir_scores[inds]
    else:
        bboxes = mlvl_scores.new_zeros((0, mlvl_bboxes.size(-1)))
        scores = mlvl_scores.new_zeros((0, ))
        labels = mlvl_scores.new_zeros((0, ), dtype=torch.long)
        dir_scores = mlvl_scores.new_zeros((0, ))
    return bboxes, scores, labels, dir_scores


def aligned_3d_nms(boxes, scores, classes, thresh):
    """3d nms for aligned boxes.

    Args:
        boxes (Tensor): Aligned box with shape [n, 6].
        scores (Tensor): Scores of each box.
        classes (Tensor): Class of each box.
        thresh (float): Iou threshold for nms.

    Returns:
        Tensor: Indices of selected boxes.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    z1 = boxes[:, 2]
    x2 = boxes[:, 3]
    y2 = boxes[:, 4]
    z2 = boxes[:, 5]
    area = (x2 - x1) * (y2 - y1) * (z2 - z1)
    zero = boxes.new_zeros(1, )

    score_sorted = torch.argsort(scores)
    pick = []
    while (score_sorted.shape[0] != 0):
        last = score_sorted.shape[0]
        i = score_sorted[-1]
        pick.append(i)

        xx1 = torch.max(x1[i], x1[score_sorted[:last - 1]])
        yy1 = torch.max(y1[i], y1[score_sorted[:last - 1]])
        zz1 = torch.max(z1[i], z1[score_sorted[:last - 1]])
        xx2 = torch.min(x2[i], x2[score_sorted[:last - 1]])
        yy2 = torch.min(y2[i], y2[score_sorted[:last - 1]])
        zz2 = torch.min(z2[i], z2[score_sorted[:last - 1]])
        classes1 = classes[i]
        classes2 = classes[score_sorted[:last - 1]]
        inter_l = torch.max(zero, xx2 - xx1)
        inter_w = torch.max(zero, yy2 - yy1)
        inter_h = torch.max(zero, zz2 - zz1)

        inter = inter_l * inter_w * inter_h
        iou = inter / (area[i] + area[score_sorted[:last - 1]] - inter)
        iou = iou * (classes1 == classes2).float()
        score_sorted = score_sorted[torch.nonzero(iou <= thresh).flatten()]

    indices = boxes.new_tensor(pick, dtype=torch.long)
    return indices
