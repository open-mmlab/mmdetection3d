# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

import mmdet3d
from mmdet3d.structures.bbox_3d.utils import limit_period


def normalize_bbox(bboxes, pc_range):

    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    length = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()

    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, w, length, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1)
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, w, length, cz, h, rot.sin(), rot.cos()), dim=-1)
    return normalized_bboxes


def denormalize_bbox(normalized_bboxes, pc_range):
    # rotation
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    # size
    w = normalized_bboxes[..., 2:3]
    length = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]

    w = w.exp()
    length = length.exp()
    h = h.exp()
    if normalized_bboxes.size(-1) > 8:
        # velocity
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat(
            [cx, cy, cz, w, length, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, length, h, rot],
                                        dim=-1)

    if int(mmdet3d.__version__[0]) >= 1:
        denormalized_bboxes_clone = denormalized_bboxes.clone()
        denormalized_bboxes[:, 3] = denormalized_bboxes_clone[:, 4]
        denormalized_bboxes[:, 4] = denormalized_bboxes_clone[:, 3]
        # change yaw
        denormalized_bboxes[:,
                            6] = -denormalized_bboxes_clone[:, 6] - np.pi / 2
        denormalized_bboxes[:, 6] = limit_period(
            denormalized_bboxes[:, 6], period=np.pi * 2)
    return denormalized_bboxes
