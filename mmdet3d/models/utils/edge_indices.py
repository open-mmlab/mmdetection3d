# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np


def get_edge_indices(img_metas, pad_mode='default'):
    """Function to filter the objects label outside the image.

    Args:
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
            shape (num_gt, 4).
        centers2d_list (list[Tensor]): Projected 3D centers onto 2D image,
            shape (num_gt, 2).
        img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.

    Returns:
        tuple:
            edge_indices(np.ndarray): Edge indices for each image in batch
                data.
            edge_len(int): The length of edge indices.
    """
    for i in range(len(img_metas)):
        img_shape = img_metas[i]['img_shape']
        h, w = img_shape[:2]

        if pad_mode == 'default':
            x_min = 0
            y_min = 0
            x_max, y_max = w - 1, h - 1
        step = 1

        # boundary idxs
        edge_indices = []

        # left
        y = np.arange(y_min, y_max, step)
        x = np.ones(len(y)) * x_min

        edge_indices_edge = np.stack((x, y), axis=1)
        edge_indices_edge[:, 0] = np.clip(edge_indices_edge[:, 0], x_min,
                                          x_max)
        edge_indices_edge[:, 1] = np.clip(edge_indices_edge[:, 1], y_min,
                                          y_max)
        edge_indices.append(edge_indices_edge)

        # bottom
        x = np.arange(x_min, x_max, step)
        y = np.ones(len(x)) * y_max

        edge_indices_edge = np.stack((x, y), axis=1)
        edge_indices_edge[:, 0] = np.clip(edge_indices_edge[:, 0], x_min,
                                          x_max)
        edge_indices_edge[:, 1] = np.clip(edge_indices_edge[:, 1], y_min,
                                          y_max)
        edge_indices.append(edge_indices_edge)

        # right
        y = np.arange(y_max, y_min, -step)
        x = np.ones(len(y)) * x_max

        edge_indices_edge = np.stack((x, y), axis=1)
        edge_indices_edge[:, 0] = np.clip(edge_indices_edge[:, 0], x_min,
                                          x_max)
        edge_indices_edge[:, 1] = np.clip(edge_indices_edge[:, 1], y_min,
                                          y_max)
        edge_indices.append(edge_indices_edge)

        # top
        x = np.arange(x_max, x_min - 1, -step)
        y = np.ones(len(x)) * y_min

        edge_indices_edge = np.stack((x, y), axis=1)
        edge_indices_edge[:, 0] = np.clip(edge_indices_edge[:, 0], x_min,
                                          x_max)
        edge_indices_edge[:, 1] = np.clip(edge_indices_edge[:, 1], y_min,
                                          y_max)
        edge_indices.append(edge_indices_edge)

        edge_indices = np.concatenate([index for index in edge_indices],
                                      axis=0)
        edge_len = edge_indices.shape[0] - 1

        return (edge_indices, edge_len)
