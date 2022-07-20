# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet3d.structures import points_cam2img


def get_keypoints(gt_bboxes_3d_list,
                  centers2d_list,
                  img_metas,
                  use_local_coords=True):
    """Function to filter the objects label outside the image.

    Args:
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
            shape (num_gt, 4).
        centers2d_list (list[Tensor]): Projected 3D centers onto 2D image,
            shape (num_gt, 2).
        img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        use_local_coords (bool, optional): Wheher to use local coordinates
            for keypoints. Default: True.

    Returns:
        tuple[list[Tensor]]: It contains two elements, the first is the
        keypoints for each projected 2D bbox in batch data. The second is
        the visible mask of depth calculated by keypoints.
    """

    assert len(gt_bboxes_3d_list) == len(centers2d_list)
    bs = len(gt_bboxes_3d_list)
    keypoints2d_list = []
    keypoints_depth_mask_list = []

    for i in range(bs):
        gt_bboxes_3d = gt_bboxes_3d_list[i]
        centers2d = centers2d_list[i]
        img_shape = img_metas[i]['img_shape']
        cam2img = img_metas[i]['cam2img']
        h, w = img_shape[:2]
        # (N, 8, 3)
        corners3d = gt_bboxes_3d.corners
        top_centers3d = torch.mean(corners3d[:, [0, 1, 4, 5], :], dim=1)
        bot_centers3d = torch.mean(corners3d[:, [2, 3, 6, 7], :], dim=1)
        # (N, 2, 3)
        top_bot_centers3d = torch.stack((top_centers3d, bot_centers3d), dim=1)
        keypoints3d = torch.cat((corners3d, top_bot_centers3d), dim=1)
        # (N, 10, 2)
        keypoints2d = points_cam2img(keypoints3d, cam2img)

        # keypoints mask: keypoints must be inside
        # the image and in front of the camera
        keypoints_x_visible = (keypoints2d[..., 0] >= 0) & (
            keypoints2d[..., 0] <= w - 1)
        keypoints_y_visible = (keypoints2d[..., 1] >= 0) & (
            keypoints2d[..., 1] <= h - 1)
        keypoints_z_visible = (keypoints3d[..., -1] > 0)

        # (N, 1O)
        keypoints_visible = keypoints_x_visible & \
            keypoints_y_visible & keypoints_z_visible
        # center, diag-02, diag-13
        keypoints_depth_valid = torch.stack(
            (keypoints_visible[:, [8, 9]].all(dim=1),
             keypoints_visible[:, [0, 3, 5, 6]].all(dim=1),
             keypoints_visible[:, [1, 2, 4, 7]].all(dim=1)),
            dim=1)
        keypoints_visible = keypoints_visible.float()

        if use_local_coords:
            keypoints2d = torch.cat((keypoints2d - centers2d.unsqueeze(1),
                                     keypoints_visible.unsqueeze(-1)),
                                    dim=2)
        else:
            keypoints2d = torch.cat(
                (keypoints2d, keypoints_visible.unsqueeze(-1)), dim=2)

        keypoints2d_list.append(keypoints2d)
        keypoints_depth_mask_list.append(keypoints_depth_valid)

    return (keypoints2d_list, keypoints_depth_mask_list)
