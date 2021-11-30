# Copyright (c) OpenMMLab. All rights reserved.
import torch


def get_edge_indices(img_metas,
                     step=1,
                     pad_mode='default',
                     dtype=torch.float32,
                     device='cpu'):
    """Function to filter the objects label outside the image.

    Args:
        img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        step (int): Step size used for generateing edge indices.
            Default: 1.
        pad_mode (str): Padding mode during data pipeline.
            Default: 'default'.
        dtype (torch.dtype): Dtype of edge indices tensor.
            Default: torch.float32.
        device (str): Device of edge indices tensor. Default: 'cpu'.

    Returns:
        tuple:
            edge_indices_list(list[Tensor]): Edge indices for each
                image in batch data.
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
        y = torch.arange(y_min, y_max, step, dtype=dtype, device=device)
        x = torch.ones(len(y)) * x_min

        edge_indices_edge = torch.stack((x, y), dim=1)
        edge_indices_edge[:, 0] = torch.clamp(edge_indices_edge[:, 0], x_min)
        edge_indices_edge[:, 1] = torch.clamp(edge_indices_edge[:, 1], y_min)
        edge_indices_edge = torch.unique(edge_indices_edge, dim=0)
        edge_indices.append(edge_indices_edge)

        # bottom
        x = torch.arange(x_min, x_max, step, dtype=dtype, device=device)
        y = torch.ones(len(x)) * y_max

        edge_indices_edge = torch.stack((x, y), dim=1)
        edge_indices_edge[:, 0] = torch.clamp(edge_indices_edge[:, 0], x_min)
        edge_indices_edge[:, 1] = torch.clamp(edge_indices_edge[:, 1], y_min)
        edge_indices_edge = torch.unique(edge_indices_edge, dim=0)
        edge_indices.append(edge_indices_edge)

        # right
        y = torch.arange(y_max, y_min, -step, dtype=dtype, device=device)
        x = torch.ones(len(y)) * x_max

        edge_indices_edge = torch.stack((x, y), dim=1)
        edge_indices_edge[:, 0] = torch.clamp(edge_indices_edge[:, 0], x_min)
        edge_indices_edge[:, 1] = torch.clamp(edge_indices_edge[:, 1], y_min)
        edge_indices_edge = \
            torch.unique(edge_indices_edge, dim=0).flip(dims=[0])
        edge_indices.append(edge_indices_edge)

        # top
        x = torch.arange(x_max, x_min, -step, dtype=dtype, device=device)
        y = torch.ones(len(x)) * y_min

        edge_indices_edge = torch.stack((x, y), dim=1)
        edge_indices_edge[:, 0] = torch.clamp(edge_indices_edge[:, 0], x_min)
        edge_indices_edge[:, 1] = torch.clamp(edge_indices_edge[:, 1], y_min)
        edge_indices_edge = \
            torch.unique(edge_indices_edge, dim=0).flip(dims=[0])
        edge_indices.append(edge_indices_edge)

        # concatenate
        edge_indices = \
            torch.cat([index.long() for index in edge_indices], dim=0)
        edge_indices_list.append(edge_indices)

    return edge_indices_list
