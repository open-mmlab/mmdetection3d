# Copyright (c) OpenMMLab. All rights reserved.
import random

import numpy as np
import pytest
import torch
from torch import nn as nn


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def test_paconv_regularization_loss():
    from mmdet3d.models.layers import PAConv, PAConvCUDA
    from mmdet3d.models.losses import PAConvRegularizationLoss

    class ToyModel(nn.Module):

        def __init__(self):
            super(ToyModel, self).__init__()

            self.paconvs = nn.ModuleList()
            self.paconvs.append(PAConv(8, 16, 8))
            self.paconvs.append(PAConv(8, 16, 8, kernel_input='identity'))
            self.paconvs.append(PAConvCUDA(8, 16, 8))

            self.conv1 = nn.Conv1d(3, 8, 1)

    set_random_seed(0, True)
    model = ToyModel()

    # reduction should be in ['none', 'mean', 'sum']
    with pytest.raises(AssertionError):
        paconv_corr_loss = PAConvRegularizationLoss(reduction='l2')

    paconv_corr_loss = PAConvRegularizationLoss(reduction='mean')
    mean_corr_loss = paconv_corr_loss(model.modules())
    assert mean_corr_loss >= 0
    assert mean_corr_loss.requires_grad

    sum_corr_loss = paconv_corr_loss(model.modules(), reduction_override='sum')
    assert torch.allclose(sum_corr_loss, mean_corr_loss * 3)

    none_corr_loss = paconv_corr_loss(
        model.modules(), reduction_override='none')
    assert none_corr_loss.shape[0] == 3
    assert torch.allclose(none_corr_loss.mean(), mean_corr_loss)
