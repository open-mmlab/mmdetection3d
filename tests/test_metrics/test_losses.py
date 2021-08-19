# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from torch import nn as nn


def test_chamfer_disrance():
    from mmdet3d.models.losses import ChamferDistance, chamfer_distance

    with pytest.raises(AssertionError):
        # test invalid mode
        ChamferDistance(mode='smoothl1')
        # test invalid type of reduction
        ChamferDistance(mode='l2', reduction=None)

    self = ChamferDistance(
        mode='l2', reduction='sum', loss_src_weight=1.0, loss_dst_weight=1.0)
    source = torch.tensor([[[-0.9888, 0.9683, -0.8494],
                            [-6.4536, 4.5146,
                             1.6861], [2.0482, 5.6936, -1.4701],
                            [-0.5173, 5.6472, 2.1748],
                            [-2.8010, 5.4423, -1.2158],
                            [2.4018, 2.4389, -0.2403],
                            [-2.8811, 3.8486, 1.4750],
                            [-0.2031, 3.8969,
                             -1.5245], [1.3827, 4.9295, 1.1537],
                            [-2.6961, 2.2621, -1.0976]],
                           [[0.3692, 1.8409,
                             -1.4983], [1.9995, 6.3602, 0.1798],
                            [-2.1317, 4.6011,
                             -0.7028], [2.4158, 3.1482, 0.3169],
                            [-0.5836, 3.6250, -1.2650],
                            [-1.9862, 1.6182, -1.4901],
                            [2.5992, 1.2847, -0.8471],
                            [-0.3467, 5.3681, -1.4755],
                            [-0.8576, 3.3400, -1.7399],
                            [2.7447, 4.6349, 0.1994]]])

    target = torch.tensor([[[-0.4758, 1.0094, -0.8645],
                            [-0.3130, 0.8564, -0.9061],
                            [-0.1560, 2.0394, -0.8936],
                            [-0.3685, 1.6467, -0.8271],
                            [-0.2740, 2.2212, -0.7980]],
                           [[1.4856, 2.5299,
                             -1.0047], [2.3262, 3.3065, -0.9475],
                            [2.4593, 2.5870,
                             -0.9423], [0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000]]])

    loss_source, loss_target, indices1, indices2 = self(
        source, target, return_indices=True)

    assert torch.allclose(loss_source, torch.tensor(219.5936))
    assert torch.allclose(loss_target, torch.tensor(22.3705))

    expected_inds1 = [[0, 4, 4, 4, 4, 2, 4, 4, 4, 3],
                      [0, 1, 0, 1, 0, 4, 2, 0, 0, 1]]
    expected_inds2 = [[0, 4, 4, 4, 4, 2, 4, 4, 4, 3],
                      [0, 1, 0, 1, 0, 3, 2, 0, 0, 1]]
    assert (torch.equal(indices1, indices1.new_tensor(expected_inds1))
            or torch.equal(indices1, indices1.new_tensor(expected_inds2)))
    assert torch.equal(indices2,
                       indices2.new_tensor([[0, 0, 0, 0, 0], [0, 3, 6, 0, 0]]))

    loss_source, loss_target, indices1, indices2 = chamfer_distance(
        source, target, reduction='sum')

    assert torch.allclose(loss_source, torch.tensor(219.5936))
    assert torch.allclose(loss_target, torch.tensor(22.3705))
    assert (torch.equal(indices1, indices1.new_tensor(expected_inds1))
            or torch.equal(indices1, indices1.new_tensor(expected_inds2)))
    assert (indices2 == indices2.new_tensor([[0, 0, 0, 0, 0], [0, 3, 6, 0,
                                                               0]])).all()


def test_paconv_regularization_loss():
    from mmdet3d.models.losses import PAConvRegularizationLoss
    from mmdet3d.ops import PAConv, PAConvCUDA
    from mmdet.apis import set_random_seed

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

    # reduction shoule be in ['none', 'mean', 'sum']
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
