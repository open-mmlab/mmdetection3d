# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
from torch import Tensor

from mmdet3d.structures import CameraInstance3DBoxes


def filter_outside_objs(gt_bboxes_list: List[Tensor],
                        gt_labels_list: List[Tensor],
                        gt_bboxes_3d_list: List[CameraInstance3DBoxes],
                        gt_labels_3d_list: List[Tensor],
                        centers2d_list: List[Tensor],
                        img_metas: List[dict]) -> None:
    """Function to filter the objects label outside the image.

    Args:
        gt_bboxes_list (List[Tensor]): Ground truth bboxes of each image,
            each has shape (num_gt, 4).
        gt_labels_list (List[Tensor]): Ground truth labels of each box,
            each has shape (num_gt,).
        gt_bboxes_3d_list (List[:obj:`CameraInstance3DBoxes`]): 3D Ground
            truth bboxes of each image, each has shape
            (num_gt, bbox_code_size).
        gt_labels_3d_list (List[Tensor]): 3D Ground truth labels of each
            box, each has shape (num_gt,).
        centers2d_list (List[Tensor]): Projected 3D centers onto 2D image,
            each has shape (num_gt, 2).
        img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
    """
    bs = len(centers2d_list)

    for i in range(bs):
        centers2d = centers2d_list[i].clone()
        img_shape = img_metas[i]['img_shape']
        keep_inds = (centers2d[:, 0] > 0) & \
                    (centers2d[:, 0] < img_shape[1]) & \
                    (centers2d[:, 1] > 0) & \
                    (centers2d[:, 1] < img_shape[0])
        centers2d_list[i] = centers2d[keep_inds]
        gt_labels_list[i] = gt_labels_list[i][keep_inds]
        gt_bboxes_list[i] = gt_bboxes_list[i][keep_inds]
        gt_bboxes_3d_list[i].tensor = gt_bboxes_3d_list[i].tensor[keep_inds]
        gt_labels_3d_list[i] = gt_labels_3d_list[i][keep_inds]


def get_centers2d_target(centers2d: Tensor, centers: Tensor,
                         img_shape: tuple) -> Tensor:
    """Function to get target centers2d.

    Args:
        centers2d (Tensor): Projected 3D centers onto 2D images.
        centers (Tensor): Centers of 2d gt bboxes.
        img_shape (tuple): Resized image shape.

    Returns:
        torch.Tensor: Projected 3D centers (centers2D) target.
    """
    N = centers2d.shape[0]
    h, w = img_shape[:2]
    valid_intersects = centers2d.new_zeros((N, 2))
    a = (centers[:, 1] - centers2d[:, 1]) / (centers[:, 0] - centers2d[:, 0])
    b = centers[:, 1] - a * centers[:, 0]
    left_y = b
    right_y = (w - 1) * a + b
    top_x = -b / a
    bottom_x = (h - 1 - b) / a

    left_coors = torch.stack((left_y.new_zeros(N, ), left_y), dim=1)
    right_coors = torch.stack((right_y.new_full((N, ), w - 1), right_y), dim=1)
    top_coors = torch.stack((top_x, top_x.new_zeros(N, )), dim=1)
    bottom_coors = torch.stack((bottom_x, bottom_x.new_full((N, ), h - 1)),
                               dim=1)

    intersects = torch.stack(
        [left_coors, right_coors, top_coors, bottom_coors], dim=1)
    intersects_x = intersects[:, :, 0]
    intersects_y = intersects[:, :, 1]
    inds = (intersects_x >= 0) & (intersects_x <=
                                  w - 1) & (intersects_y >= 0) & (
                                      intersects_y <= h - 1)
    valid_intersects = intersects[inds].reshape(N, 2, 2)
    dist = torch.norm(valid_intersects - centers2d.unsqueeze(1), dim=2)
    min_idx = torch.argmin(dist, dim=1)

    min_idx = min_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2)
    centers2d_target = valid_intersects.gather(dim=1, index=min_idx).squeeze(1)

    return centers2d_target


def handle_proj_objs(
        centers2d_list: List[Tensor], gt_bboxes_list: List[Tensor],
        img_metas: List[dict]
) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
    """Function to handle projected object centers2d, generate target
    centers2d.

    Args:
        gt_bboxes_list (List[Tensor]): Ground truth bboxes of each image,
            shape (num_gt, 4).
        centers2d_list (List[Tensor]): Projected 3D centers onto 2D image,
            shape (num_gt, 2).
        img_metas (List[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.

    Returns:
        Tuple[List[Tensor], List[Tensor], List[Tensor]]: It contains three
        elements. The first is the target centers2d after handling the
        truncated objects. The second is the offsets between target centers2d
        and round int dtype centers2d,and the last is the truncation mask
        for each object in batch data.
    """
    bs = len(centers2d_list)
    centers2d_target_list = []
    trunc_mask_list = []
    offsets2d_list = []
    # for now, only pad mode that img is padded by right and
    # bottom side is supported.
    for i in range(bs):
        centers2d = centers2d_list[i]
        gt_bbox = gt_bboxes_list[i]
        img_shape = img_metas[i]['img_shape']
        centers2d_target = centers2d.clone()
        inside_inds = (centers2d[:, 0] > 0) & \
                      (centers2d[:, 0] < img_shape[1]) & \
                      (centers2d[:, 1] > 0) & \
                      (centers2d[:, 1] < img_shape[0])
        outside_inds = ~inside_inds

        # if there are outside objects
        if outside_inds.any():
            centers = (gt_bbox[:, :2] + gt_bbox[:, 2:]) / 2
            outside_centers2d = centers2d[outside_inds]
            match_centers = centers[outside_inds]
            target_outside_centers2d = get_centers2d_target(
                outside_centers2d, match_centers, img_shape)
            centers2d_target[outside_inds] = target_outside_centers2d

        offsets2d = centers2d - centers2d_target.round().int()
        trunc_mask = outside_inds

        centers2d_target_list.append(centers2d_target)
        trunc_mask_list.append(trunc_mask)
        offsets2d_list.append(offsets2d)

    return (centers2d_target_list, offsets2d_list, trunc_mask_list)
