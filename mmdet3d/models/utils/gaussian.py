# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor


def gaussian_2d(shape: Tuple[int, int], sigma: float = 1) -> np.ndarray:
    """Generate gaussian map.

    Args:
        shape (Tuple[int]): Shape of the map.
        sigma (float): Sigma to generate gaussian map.
            Defaults to 1.

    Returns:
        np.ndarray: Generated gaussian map.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_heatmap_gaussian(heatmap: Tensor,
                          center: Tensor,
                          radius: int,
                          k: int = 1) -> Tensor:
    """Get gaussian masked heatmap.

    Args:
        heatmap (Tensor): Heatmap to be masked.
        center (Tensor): Center coord of the heatmap.
        radius (int): Radius of gaussian.
        k (int): Multiple of masked_gaussian. Defaults to 1.

    Returns:
        Tensor: Masked heatmap.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[radius - top:radius + bottom,
                 radius - left:radius + right]).to(heatmap.device,
                                                   torch.float32)
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian_radius(det_size: Tuple[Tensor, Tensor],
                    min_overlap: float = 0.5) -> Tensor:
    """Get radius of gaussian.

    Args:
        det_size (Tuple[Tensor]): Size of the detection result.
        min_overlap (float): Gaussian_overlap. Defaults to 0.5.

    Returns:
        Tensor: Computed radius.
    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = torch.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def get_ellip_gaussian_2D(heatmap: Tensor,
                          center: List[int],
                          radius_x: int,
                          radius_y: int,
                          k: int = 1) -> Tensor:
    """Generate 2D ellipse gaussian heatmap.

    Args:
        heatmap (Tensor): Input heatmap, the gaussian kernel will cover on
            it and maintain the max value.
        center (List[int]): Coord of gaussian kernel's center.
        radius_x (int): X-axis radius of gaussian kernel.
        radius_y (int): Y-axis radius of gaussian kernel.
        k (int): Coefficient of gaussian kernel. Defaults to 1.

    Returns:
        out_heatmap (Tensor): Updated heatmap covered by gaussian kernel.
    """
    diameter_x, diameter_y = 2 * radius_x + 1, 2 * radius_y + 1
    gaussian_kernel = ellip_gaussian2D((radius_x, radius_y),
                                       sigma_x=diameter_x // 6,
                                       sigma_y=diameter_y // 6,
                                       dtype=heatmap.dtype,
                                       device=heatmap.device)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius_x), min(width - x, radius_x + 1)
    top, bottom = min(y, radius_y), min(height - y, radius_y + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian_kernel[radius_y - top:radius_y + bottom,
                                      radius_x - left:radius_x + right]
    out_heatmap = heatmap
    torch.max(
        masked_heatmap,
        masked_gaussian * k,
        out=out_heatmap[y - top:y + bottom, x - left:x + right])

    return out_heatmap


def ellip_gaussian2D(radius: Tuple[int, int],
                     sigma_x: int,
                     sigma_y: int,
                     dtype: torch.dtype = torch.float32,
                     device: str = 'cpu') -> Tensor:
    """Generate 2D ellipse gaussian kernel.

    Args:
        radius (Tuple[int]): Ellipse radius (radius_x, radius_y) of gaussian
            kernel.
        sigma_x (int): X-axis sigma of gaussian function.
        sigma_y (int): Y-axis sigma of gaussian function.
        dtype (torch.dtype): Dtype of gaussian tensor.
            Defaults to torch.float32.
        device (str): Device of gaussian tensor.
            Defaults to 'cpu'.

    Returns:
        h (Tensor): Gaussian kernel with a
            ``(2 * radius_y + 1) * (2 * radius_x + 1)`` shape.
    """
    x = torch.arange(
        -radius[0], radius[0] + 1, dtype=dtype, device=device).view(1, -1)
    y = torch.arange(
        -radius[1], radius[1] + 1, dtype=dtype, device=device).view(-1, 1)

    h = (-(x * x) / (2 * sigma_x * sigma_x) - (y * y) /
         (2 * sigma_y * sigma_y)).exp()
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0

    return h
