# Copyright (c) OpenMMLab. All rights reserved.
import numba
import numpy as np
import torch


def get_edge_indices(img_metas,
                     step=1,
                     pad_mode='default',
                     dtype=np.float32,
                     device='cpu'):
    """Function to filter the objects label outside the image.
    The edge_indices are generated using numpy on cpu, not tensor
    on CUDA, considering the running time. When batch size = 8,
    running this function 100 times on numpy with numba acceleration
    consumes 0.08s while 0.32s on pure numpy, 0.98s on CUDA tensor.

    Args:
        img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        step (int, optional): Step size used for generateing
            edge indices. Default: 1.
        pad_mode (str, optional): Padding mode during data pipeline.
            Default: 'default'.
        dtype (torch.dtype, optional): Dtype of edge indices tensor.
            Default: np.float32.
        device (str, optional): Device of edge indices tensor.
            Default: 'cpu'.

    Returns:
        list[Tensor]: Edge indices for each image in batch data.
    """
    edge_indices_list = []
    for i in range(len(img_metas)):
        img_shape = img_metas[i]['img_shape']
        h, w = img_shape[:2]
        edge_indices = []

        if pad_mode == 'default':
            x_min = 0
            y_min = 0
            x_max, y_max = w - 1, h - 1
        else:
            raise NotImplementedError

        # left
        y = np.arange(y_min, y_max, step, dtype=dtype)
        x = np.ones(len(y)) * x_min

        edge_indices_edge = get_one_edge(x, y, x_min, x_max, y_min, y_max)
        edge_indices.append(edge_indices_edge)

        # bottom
        x = np.arange(x_min, x_max, step, dtype=dtype)
        y = np.ones(len(x)) * y_max

        edge_indices_edge = get_one_edge(x, y, x_min, x_max, y_min, y_max)
        edge_indices.append(edge_indices_edge)

        # right
        y = np.arange(y_max, y_min, -step, dtype=dtype)
        x = np.ones(len(y)) * x_max

        edge_indices_edge = get_one_edge(x, y, x_min, x_max, y_min, y_max)
        edge_indices.append(edge_indices_edge)

        # top
        x = np.arange(x_max, x_min, -step, dtype=dtype)
        y = np.ones(len(x)) * y_min

        edge_indices_edge = get_one_edge(x, y, x_min, x_max, y_min, y_max)
        edge_indices.append(edge_indices_edge)

        edge_indices = \
            np.concatenate([index for index in edge_indices], axis=0)
        edge_indices = torch.from_numpy(edge_indices).to(device).long()
        edge_indices_list.append(edge_indices)

    return edge_indices_list


@numba.njit
def get_one_edge(x, y, x_min, x_max, y_min, y_max):
    """Function to generate one edge indices.

    Args:
        x (float): One edge x coordinates.
        y (float): One edge y coordinates.
        x_min (float|int): Minimum x coordinates of image.
        x_max (float|int): Maximum x coordinates of image.
        y_min (float|int): Minimum y coordinates of image.
        y_max (float|int): Maximum y coordinates of image.

    Returns:
        np.array: Edge indices of one side for input image.
    """
    edge_indices_edge = np.stack((x, y), axis=1)
    edge_indices_edge[:, 0] = np.clip(edge_indices_edge[:, 0], x_min, x_max)
    edge_indices_edge[:, 1] = np.clip(edge_indices_edge[:, 1], y_min, y_max)
    return edge_indices_edge
