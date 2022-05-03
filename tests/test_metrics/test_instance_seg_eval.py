# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmdet3d.core import instance_seg_eval


def test_instance_seg_eval():
    valid_class_ids = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34,
                       36, 39)
    class_labels = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                    'window', 'bookshelf', 'picture', 'counter', 'desk',
                    'curtain', 'refrigerator', 'showercurtrain', 'toilet',
                    'sink', 'bathtub', 'garbagebin')
    n_points_list = [3300, 3000]
    gt_labels_list = [[0, 0, 0, 0, 0, 0, 14, 14, 2, 1],
                      [13, 13, 2, 1, 3, 3, 0, 0, 0]]
    gt_instance_masks = []
    gt_semantic_masks = []
    pred_instance_masks = []
    pred_instance_labels = []
    pred_instance_scores = []
    for n_points, gt_labels in zip(n_points_list, gt_labels_list):
        gt_instance_mask = np.ones(n_points, dtype=np.int) * -1
        gt_semantic_mask = np.ones(n_points, dtype=np.int) * -1
        pred_instance_mask = np.ones(n_points, dtype=np.int) * -1
        labels = []
        scores = []
        for i, gt_label in enumerate(gt_labels):
            begin = i * 300
            end = begin + 300
            gt_instance_mask[begin:end] = i
            gt_semantic_mask[begin:end] = gt_label
            pred_instance_mask[begin:end] = i
            labels.append(gt_label)
            scores.append(.99)
        gt_instance_masks.append(torch.tensor(gt_instance_mask))
        gt_semantic_masks.append(torch.tensor(gt_semantic_mask))
        pred_instance_masks.append(torch.tensor(pred_instance_mask))
        pred_instance_labels.append(torch.tensor(labels))
        pred_instance_scores.append(torch.tensor(scores))

    ret_value = instance_seg_eval(
        gt_semantic_masks=gt_semantic_masks,
        gt_instance_masks=gt_instance_masks,
        pred_instance_masks=pred_instance_masks,
        pred_instance_labels=pred_instance_labels,
        pred_instance_scores=pred_instance_scores,
        valid_class_ids=valid_class_ids,
        class_labels=class_labels)
    for label in [
            'cabinet', 'bed', 'chair', 'sofa', 'showercurtrain', 'toilet'
    ]:
        metrics = ret_value['classes'][label]
        assert metrics['ap'] == 1.0
        assert metrics['ap50%'] == 1.0
        assert metrics['ap25%'] == 1.0

    pred_instance_masks[1][2240:2700] = -1
    pred_instance_masks[0][2700:3000] = 8
    pred_instance_labels[0][9] = 2
    ret_value = instance_seg_eval(
        gt_semantic_masks=gt_semantic_masks,
        gt_instance_masks=gt_instance_masks,
        pred_instance_masks=pred_instance_masks,
        pred_instance_labels=pred_instance_labels,
        pred_instance_scores=pred_instance_scores,
        valid_class_ids=valid_class_ids,
        class_labels=class_labels)
    assert abs(ret_value['classes']['cabinet']['ap50%'] - 0.72916) < 0.01
    assert abs(ret_value['classes']['cabinet']['ap25%'] - 0.88888) < 0.01
    assert abs(ret_value['classes']['bed']['ap50%'] - 0.5) < 0.01
    assert abs(ret_value['classes']['bed']['ap25%'] - 0.5) < 0.01
    assert abs(ret_value['classes']['chair']['ap50%'] - 0.375) < 0.01
    assert abs(ret_value['classes']['chair']['ap25%'] - 1.0) < 0.01
