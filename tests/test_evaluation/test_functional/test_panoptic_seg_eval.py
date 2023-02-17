# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmdet3d.evaluation.functional.panoptic_seg_eval import panoptic_seg_eval


def test_panoptic_seg_eval():
    if not torch.cuda.is_available():
        pytest.skip()

    classes = ['unlabeled', 'person', 'dog', 'grass', 'sky']
    label2cat = {
        0: 'unlabeled',
        1: 'person',
        2: 'dog',
        3: 'grass',
        4: 'sky',
    }

    thing_classes = ['person', 'dog']
    stuff_classes = ['grass', 'sky']
    ignore_index = [0]  # only ignore ignore class
    min_points = 1  # for this example we care about all points
    offset = 2**16

    # generate ground truth and prediction
    semantic_preds = []
    instance_preds = []
    gt_semantic = []
    gt_instance = []

    # some ignore stuff
    num_ignore = 50
    semantic_preds.extend([0 for i in range(num_ignore)])
    instance_preds.extend([0 for i in range(num_ignore)])
    gt_semantic.extend([0 for i in range(num_ignore)])
    gt_instance.extend([0 for i in range(num_ignore)])

    # grass segment
    num_grass = 50
    num_grass_pred = 40  # rest is sky
    semantic_preds.extend([1 for i in range(num_grass_pred)])  # grass
    semantic_preds.extend([2
                           for i in range(num_grass - num_grass_pred)])  # sky
    instance_preds.extend([0 for i in range(num_grass)])
    gt_semantic.extend([1 for i in range(num_grass)])  # grass
    gt_instance.extend([0 for i in range(num_grass)])

    # sky segment
    num_sky = 50
    num_sky_pred = 40  # rest is grass
    semantic_preds.extend([2 for i in range(num_sky_pred)])  # sky
    semantic_preds.extend([1 for i in range(num_sky - num_sky_pred)])  # grass
    instance_preds.extend([0 for i in range(num_sky)])  # first instance
    gt_semantic.extend([2 for i in range(num_sky)])  # sky
    gt_instance.extend([0 for i in range(num_sky)])  # first instance

    # wrong dog as person prediction
    num_dog = 50
    num_person = num_dog
    semantic_preds.extend([3 for i in range(num_person)])
    instance_preds.extend([35 for i in range(num_person)])
    gt_semantic.extend([4 for i in range(num_dog)])
    gt_instance.extend([22 for i in range(num_dog)])

    # two persons in prediction, but three in gt
    num_person = 50
    semantic_preds.extend([3 for i in range(6 * num_person)])
    instance_preds.extend([8 for i in range(4 * num_person)])
    instance_preds.extend([95 for i in range(2 * num_person)])
    gt_semantic.extend([3 for i in range(6 * num_person)])
    gt_instance.extend([33 for i in range(3 * num_person)])
    gt_instance.extend([42 for i in range(num_person)])
    gt_instance.extend([11 for i in range(2 * num_person)])

    # gt and pred to numpy
    semantic_preds = np.array(semantic_preds, dtype=int).reshape(1, -1)
    instance_preds = np.array(instance_preds, dtype=int).reshape(1, -1)
    gt_semantic = np.array(gt_semantic, dtype=int).reshape(1, -1)
    gt_instance = np.array(gt_instance, dtype=int).reshape(1, -1)

    gt_labels = [{
        'pts_semantic_mask': gt_semantic,
        'pts_instance_mask': gt_instance
    }]

    seg_preds = [{
        'pts_semantic_mask': semantic_preds,
        'pts_instance_mask': instance_preds
    }]

    ret_value = panoptic_seg_eval(gt_labels, seg_preds, classes, thing_classes,
                                  stuff_classes, min_points, offset, label2cat,
                                  ignore_index)

    assert np.isclose(ret_value['pq'], 0.47916666666666663)
    assert np.isclose(ret_value['rq_mean'], 0.6666666666666666)
    assert np.isclose(ret_value['sq_mean'], 0.5520833333333333)
    assert np.isclose(ret_value['miou'], 0.5476190476190476)
