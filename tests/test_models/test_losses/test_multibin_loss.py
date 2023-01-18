# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmdet3d.registry import MODELS


def test_multibin_loss():
    from mmdet3d.models.losses import MultiBinLoss

    # reduction should be in ['none', 'mean', 'sum']
    with pytest.raises(AssertionError):
        multibin_loss = MultiBinLoss(reduction='l2')

    pred = torch.tensor([[
        0.81, 0.32, 0.78, 0.52, 0.24, 0.12, 0.32, 0.11, 1.20, 1.30, 0.20, 0.11,
        0.12, 0.11, 0.23, 0.31
    ],
                         [
                             0.02, 0.19, 0.78, 0.22, 0.31, 0.12, 0.22, 0.11,
                             1.20, 1.30, 0.45, 0.51, 0.12, 0.11, 0.13, 0.61
                         ]])
    target = torch.tensor([[1, 1, 0, 0, 2.14, 3.12, 0.68, -2.15],
                           [1, 1, 0, 0, 3.12, 3.12, 2.34, 1.23]])
    multibin_loss_cfg = dict(
        type='MultiBinLoss', reduction='none', loss_weight=1.0)
    multibin_loss = MODELS.build(multibin_loss_cfg)
    output_multibin_loss = multibin_loss(pred, target, num_dir_bins=4)
    expected_multibin_loss = torch.tensor(2.1120)
    assert torch.allclose(
        output_multibin_loss, expected_multibin_loss, atol=1e-4)
