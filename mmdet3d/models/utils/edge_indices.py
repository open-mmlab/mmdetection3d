# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import numpy as np
import torch
from torch import Tensor


def get_edge_indices(img_metas: List[dict],
                     downsample_ratio: int,
                     step: int = 1,
                     pad_mode: str = 'default',
                     dtype: type = np.float32,
                     device: str = 'cpu') -> List[Tensor]:
    """Function to filter the objects label outside the image.
    The edge_indices are generated using numpy on cpu rather
    than on CUDA due to the latency issue. When batch size = 8,
    this function with numpy array is ~8 times faster than that
    with CUDA tensor (0.09s and 0.72s in 100 runs).

    Args:
        img_metas (List[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        downsample_ratio (int): Downsample ratio of output feature,
        step (int): Step size used for generateing
            edge indices. Defaults to 1.
        pad_mode (str): Padding mode during data pipeline.
            Defaults to 'default'.
        dtype (type): Dtype of edge indices tensor.
            Defaults to np.float32.
        device (str): Device of edge indices tensor.
            Defaults to 'cpu'.

    Returns:
        List[Tensor]: Edge indices for each image in batch data.
    """
    edge_indices_list = []
    for i in range(len(img_metas)):
        img_shape = img_metas[i]['img_shape']
        pad_shape = img_metas[i]['pad_shape']
        h, w = img_shape[:2]
        pad_h, pad_w = pad_shape
        edge_indices = []

        if pad_mode == 'default':
            x_min = 0
            y_min = 0
            x_max = (w - 1) // downsample_ratio
            y_max = (h - 1) // downsample_ratio
        elif pad_mode == 'center':
            x_min = np.ceil((pad_w - w) / 2 * downsample_ratio)
            y_min = np.ceil((pad_h - h) / 2 * downsample_ratio)
            x_max = x_min + w // downsample_ratio
            y_max = y_min + h // downsample_ratio
        else:
            raise NotImplementedError

        # left
        y = np.arange(y_min, y_max, step, dtype=dtype)
        x = np.ones(len(y)) * x_min

        edge_indices_edge = np.stack((x, y), axis=1)
        edge_indices.append(edge_indices_edge)

        # bottom
        x = np.arange(x_min, x_max, step, dtype=dtype)
        y = np.ones(len(x)) * y_max

        edge_indices_edge = np.stack((x, y), axis=1)
        edge_indices.append(edge_indices_edge)

        # right
        y = np.arange(y_max, y_min, -step, dtype=dtype)
        x = np.ones(len(y)) * x_max

        edge_indices_edge = np.stack((x, y), axis=1)
        edge_indices.append(edge_indices_edge)

        # top
        x = np.arange(x_max, x_min, -step, dtype=dtype)
        y = np.ones(len(x)) * y_min

        edge_indices_edge = np.stack((x, y), axis=1)
        edge_indices.append(edge_indices_edge)

        edge_indices = \
            np.concatenate([index for index in edge_indices], axis=0)
        edge_indices = torch.from_numpy(edge_indices).to(device).long()
        edge_indices_list.append(edge_indices)

    return edge_indices_list
