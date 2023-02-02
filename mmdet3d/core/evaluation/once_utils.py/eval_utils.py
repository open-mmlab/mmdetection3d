# Copyright (c) OpenMMLab. All rights reserved.
r"""Adapted from `openpcdet/once_dataset
    <https://github.com/open-mmlab/openpcdet>`_.
"""

import numpy as np

def compute_split_parts(num_samples, num_parts):
    part_samples = num_samples // num_parts
    remain_samples = num_samples % num_parts
    if part_samples == 0:
        return [num_samples]
    if remain_samples == 0:
        return [part_samples] * num_parts
    else:
        return [part_samples] * num_parts + [remain_samples]

def overall_filter(boxes):
    ignore = np.zeros(boxes.shape[0], dtype=np.bool) # all false
    return ignore

def distance_filter(boxes, level):
    ignore = np.ones(boxes.shape[0], dtype=np.bool) # all true
    dist = np.sqrt(np.sum(boxes[:, 0:3] * boxes[:, 0:3], axis=1))

    if level == 0: # 0-30m
        flag = dist < 30
    elif level == 1: # 30-50m
        flag = (dist >= 30) & (dist < 50)
    elif level == 2: # 50m-inf
        flag = dist >= 50
    else:
        assert False, 'level < 3 for distance metric, found level %s' % (str(level))

    ignore[flag] = False
    return ignore

def overall_distance_filter(boxes, level):
    ignore = np.ones(boxes.shape[0], dtype=np.bool) # all true
    dist = np.sqrt(np.sum(boxes[:, 0:3] * boxes[:, 0:3], axis=1))

    if level == 0:
        flag = np.ones(boxes.shape[0], dtype=np.bool)
    elif level == 1: # 0-30m
        flag = dist < 30
    elif level == 2: # 30-50m
        flag = (dist >= 30) & (dist < 50)
    elif level == 3: # 50m-inf
        flag = dist >= 50
    else:
        assert False, 'level < 4 for overall & distance metric, found level %s' % (str(level))

    ignore[flag] = False
    return ignore