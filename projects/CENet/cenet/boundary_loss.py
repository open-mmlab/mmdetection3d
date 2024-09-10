# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mmdet3d.registry import MODELS


def one_hot(label: Tensor,
            n_classes: int,
            requires_grad: bool = True) -> Tensor:
    """Return One Hot Label."""
    device = label.device
    one_hot_label = torch.eye(
        n_classes, device=device, requires_grad=requires_grad)[label]
    one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)

    return one_hot_label


@MODELS.register_module()
class BoundaryLoss(nn.Module):
    """Boundary loss."""

    def __init__(self, theta0=3, theta=5, loss_weight: float = 1.0) -> None:
        super(BoundaryLoss, self).__init__()
        self.theta0 = theta0
        self.theta = theta
        self.loss_weight = loss_weight

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The output from model.
            gt (Tensor): Ground truth map.

        Returns:
            Tensor: Loss tensor.
        """
        pred = F.softmax(pred, dim=1)
        n, c, _, _ = pred.shape

        # one-hot vector of ground truth
        one_hot_gt = one_hot(gt, c)

        # boundary map
        gt_b = F.max_pool2d(
            1 - one_hot_gt,
            kernel_size=self.theta0,
            stride=1,
            padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool2d(
            1 - pred,
            kernel_size=self.theta0,
            stride=1,
            padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return self.loss_weight * loss
