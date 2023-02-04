# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmdet3d.registry import MODELS


def test_uncertain_smooth_l1_loss():
    from mmdet3d.models.losses import UncertainL1Loss, UncertainSmoothL1Loss

    # reduction should be in ['none', 'mean', 'sum']
    with pytest.raises(AssertionError):
        uncertain_l1_loss = UncertainL1Loss(reduction='l2')
    with pytest.raises(AssertionError):
        uncertain_smooth_l1_loss = UncertainSmoothL1Loss(reduction='l2')

    pred = torch.tensor([1.5783, 0.5972, 1.4821, 0.9488])
    target = torch.tensor([1.0813, -0.3466, -1.1404, -0.9665])
    sigma = torch.tensor([-1.0053, 0.4710, -1.7784, -0.8603])

    # test uncertain l1 loss
    uncertain_l1_loss_cfg = dict(
        type='UncertainL1Loss', alpha=1.0, reduction='mean', loss_weight=1.0)
    uncertain_l1_loss = MODELS.build(uncertain_l1_loss_cfg)
    mean_l1_loss = uncertain_l1_loss(pred, target, sigma)
    expected_l1_loss = torch.tensor(4.7069)
    assert torch.allclose(mean_l1_loss, expected_l1_loss, atol=1e-4)

    # test uncertain smooth l1 loss
    uncertain_smooth_l1_loss_cfg = dict(
        type='UncertainSmoothL1Loss',
        alpha=1.0,
        beta=0.5,
        reduction='mean',
        loss_weight=1.0)
    uncertain_smooth_l1_loss = MODELS.build(uncertain_smooth_l1_loss_cfg)
    mean_smooth_l1_loss = uncertain_smooth_l1_loss(pred, target, sigma)
    expected_smooth_l1_loss = torch.tensor(3.9795)
    assert torch.allclose(
        mean_smooth_l1_loss, expected_smooth_l1_loss, atol=1e-4)
