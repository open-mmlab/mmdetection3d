# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmdet3d.core.utils import array_converter


def filter_outside_objs(gt_bboxes_list, gt_labels_list, gt_bboxes_3d_list,
                        gt_labels_3d_list, centers2d_list, img_metas):
    """Function to filter the objects label outside the image.

    Args:
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
            each has shape (num_gt, 4).
        gt_labels_list (list[Tensor]): Ground truth labels of each box,
            each has shape (num_gt,).
        gt_bboxes_3d_list (list[Tensor]): 3D Ground truth bboxes of each
            image, each has shape (num_gt, bbox_code_size).
        gt_labels_3d_list (list[Tensor]): 3D Ground truth labels of each
            box, each has shape (num_gt,).
        centers2d_list (list[Tensor]): Projected 3D centers onto 2D image,
            each has shape (num_gt, 2).
        img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.

    Returns:
        None
    """
    bs = len(centers2d_list)

    for i in range(bs):
        centers2d = centers2d_list[i].copy()
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

    return


@array_converter(
    to_torch=False,
    apply_to=('centers2d', 'centers'),
    template_arg_name_='centers2d')
def get_target_centers2d(centers2d, centers, img_shape):
    """
        Args:
            centers2d (Tensor): Projected 3D centers onto 2D images.
            centers (Tensor): Centers of 2d gt bboxes.
            img_shape (tuple): Resized image shape.

        Returns:
            torch.Tensor: Target centers2d for real centers2d.
        """
    h, w = img_shape[:2]
    target_centers2d_list = []
    for i in range(centers2d.shape[0]):
        # y = ax + b
        # get a line
        center2d = centers2d[i]
        center = centers[i]
        a, b = np.polyfit([center2d[0], center[0]], [center2d[1], center[1]],
                          1)
        valid_intersects = []
        left_y = b
        if (0 <= left_y <= h - 1):
            valid_intersects.append(np.array([0, left_y]))

        right_y = (w - 1) * a + b
        if (0 <= right_y <= h - 1):
            valid_intersects.append(np.array([w - 1, right_y]))

        top_x = -b / a
        if (0 <= top_x <= w - 1):
            valid_intersects.append(np.array([top_x, 0]))

        bottom_x = (h - 1 - b) / a
        if (0 <= bottom_x <= w - 1):
            valid_intersects.append(np.array([bottom_x, h - 1]))

        valid_intersects = np.stack(valid_intersects)
        min_idx = np.argmin(
            np.linalg.norm(valid_intersects - center2d.reshape(1, 2), axis=1))
        target_centers2d_list.append(valid_intersects[min_idx])
    target_centers2d = np.stack(target_centers2d_list)

    return target_centers2d


def handle_trunc_objs(centers2d_list, gt_bboxes_list, img_metas):
    """Function to handle truncated projected object centers2d, generate target
    centers2d.

    Args:
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
            shape (num_gt, 4).
        centers2d_list (list[Tensor]): Projected 3D centers onto 2D image,
            shape (num_gt, 2).
        img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.

    Returns:
       tuple:
            target_centers2d_list(list[Tensor]): Target centers2d after
                handling the truncated objects.
            offsets2d_list(list[Tensor]): The offsets between target
                centers2d and real centers2d.
            trunc_mask_list(list[Tensor]): The truncation mask for each
                object.
    """
    # h, w, 3
    bs = len(centers2d_list)
    target_centers2d_list = []
    trunc_mask_list = []
    offsets2d_list = []
    # for now, only pad mode that img is padded by right
    # and bottom side is supported.
    for i in range(bs):
        centers2d = centers2d_list[i]
        gt_bbox = gt_bboxes_list[i]
        img_shape = img_metas[i]['img_shape']
        target_centers2d = centers2d.copy()
        inside_inds = (centers2d[:, 0] > 0) & \
            (centers2d[:, 0] < img_shape[1]) & \
            (centers2d[:, 1] > 0) & \
            (centers2d[:, 1] < img_shape[0])

        outside_inds = ~inside_inds
        centers = (gt_bbox[:, :2] + gt_bbox[:, 2:]) / 2  # (N, 2)
        outside_centers2d = centers2d[outside_inds]
        match_centers = centers[outside_inds]

        target_outside_centers2d = get_target_centers2d(
            outside_centers2d, match_centers, img_shape)

        target_centers2d[outside_inds] = target_outside_centers2d
        offsets2d = centers2d - target_centers2d.round().int()
        trunc_mask = outside_inds

        target_centers2d.append(target_centers2d)
        trunc_mask_list.append(trunc_mask)
        offsets2d_list.append(offsets2d)

    return (target_centers2d_list, offsets2d_list, trunc_mask_list)
