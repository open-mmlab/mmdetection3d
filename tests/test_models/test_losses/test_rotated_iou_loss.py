# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
import torch
import pytest

from mmengine.device import is_musa_available, is_cuda_available

from mmdet3d.models.losses import RotatedIoU3DLoss


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not is_cuda_available(), reason="requires CUDA support"
            ),
        ),
        pytest.param(
            "musa",
            marks=pytest.mark.skipif(
                not is_musa_available(), reason="requires MUSA support"
            ),
        ),
    ],
)
def test_rotated_iou_3d_loss(device: str):

    if not is_cuda_available() and is_musa_available():
        return

    boxes1 = torch.tensor([[.5, .5, .5, 1., 1., 1., .0],
                           [.5, .5, .5, 1., 1., 1., .0],
                           [.5, .5, .5, 1., 1., 1., .0],
                           [.5, .5, .5, 1., 1., 1., .0],
                           [.5, .5, .5, 1., 1., 1., .0]]).to(device)
    boxes2 = torch.tensor([[.5, .5, .5, 1., 1., 1., .0],
                           [.5, .5, .5, 1., 1., 2., np.pi / 2],
                           [.5, .5, .5, 1., 1., 1., np.pi / 4],
                           [1., 1., 1., 1., 1., 1., .0],
                           [-1.5, -1.5, -1.5, 2.5, 2.5, 2.5, .0]]).to(device)

    expect_ious = 1 - torch.tensor([[1., .5, .7071, 1 / 15, .0]]).to(device)
    ious = RotatedIoU3DLoss(reduction='none')(boxes1, boxes2)
    assert torch.allclose(ious, expect_ious, atol=1e-4)
