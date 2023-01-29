# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmdet3d.structures.bbox_3d.utils import limit_period


def normalize_bbox(bboxes, pc_range):

    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    length = bboxes[..., 3:4].log()
    width = bboxes[..., 4:5].log()
    height = bboxes[..., 5:6].log()

    rot = -bboxes[..., 6:7] - np.pi / 2
    rot = limit_period(rot, period=np.pi * 2)
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, length, width, cz, height, rot.sin(), rot.cos(), vx, vy),
            dim=-1)
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, length, width, cz, height, rot.sin(), rot.cos()), dim=-1)
    return normalized_bboxes


def denormalize_bbox(normalized_bboxes, pc_range):
    # rotation
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)
    rot = -rot - np.pi / 2
    rot = limit_period(rot, period=np.pi * 2)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    # size
    length = normalized_bboxes[..., 2:3]
    width = normalized_bboxes[..., 3:4]
    height = normalized_bboxes[..., 5:6]

    width = width.exp()
    length = length.exp()
    height = height.exp()
    if normalized_bboxes.size(-1) > 8:
        # velocity
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat(
            [cx, cy, cz, length, width, height, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat(
            [cx, cy, cz, length, width, height, rot], dim=-1)

    return denormalized_bboxes
