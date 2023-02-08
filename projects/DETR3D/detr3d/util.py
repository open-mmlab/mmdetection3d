from typing import List

import torch
from torch import Tensor


def normalize_bbox(bboxes: Tensor, pc_range: List) -> Tensor:
    """ normalize bboxes
        Args:
            bboxes (Tensor): boxes with unnormalized
                coordinates (cx,cy,cz,L,W,H,φ,v_x,v_y). Shape [num_gt, 9].
            pc_range (List): Perception range of the detector
        Returns:
            normalized_bboxes (Tensor): boxes with normalized coordinate
                (cx,cy,L,W,cz,H,sin(φ),cos(φ),v_x,v_y).
                All in range [0, 1] and shape [num_query, 10].
    """

    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    L = bboxes[..., 3:4].log()
    W = bboxes[..., 4:5].log()
    H = bboxes[..., 5:6].log()

    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, L, W, cz, H, rot.sin(), rot.cos(), vx, vy), dim=-1)
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, L, W, cz, H, rot.sin(), rot.cos()), dim=-1)
    return normalized_bboxes


def denormalize_bbox(normalized_bboxes, pc_range):
    """ denormalize bboxes
        Args:
            normalized_bboxes (Tensor): boxes with normalized coordinate
                (cx,cy,L,W,cz,H,sin(φ),cos(φ),v_x,v_y).
                All in range [0, 1] and shape [num_query, 10].
            pc_range (List): Perception range of the detector
        Returns:
            denormalized_bboxes (Tensor): boxes with unnormalized
                coordinates (cx,cy,cz,L,W,H,φ,v_x,v_y). Shape [num_gt, 9].
    """
    # rotation
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    # size, the meaning of L,W may alter in different version of mmdet3d
    L = normalized_bboxes[..., 2:3]
    W = normalized_bboxes[..., 3:4]
    H = normalized_bboxes[..., 5:6]

    L = L.exp()
    W = W.exp()
    H = H.exp()
    if normalized_bboxes.size(-1) > 8:
        # velocity
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, L, W, H, rot, vx, vy],
                                        dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, L, W, H, rot], dim=-1)
    return denormalized_bboxes
