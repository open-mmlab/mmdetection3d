# Copyright (c) OpenMMLab. All rights reserved.
"""Directly borrowed from mmsegmentation.

Modified from https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytor
ch/lovasz_losses.py Lovasz-Softmax and Jaccard hinge loss in PyTorch Maxim
Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import weight_reduce_loss
from mmengine.utils import is_list_of

from mmdet3d.registry import MODELS


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Computes gradient of the Lovasz extension w.r.t sorted errors.

    See Alg. 1 in paper.
    `The Lovasz-Softmax loss. <https://arxiv.org/abs/1705.08790>`_.

    Args:
        gt_sorted (torch.Tensor): Sorted ground truth.

    Return:
        torch.Tensor: Gradient of the Lovasz extension.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def flatten_binary_logits(
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten predictions and labels in the batch (binary case). Remove
    tensors whose labels equal to 'ignore_index'.

    Args:
        probs (torch.Tensor): Predictions to be modified.
        labels (torch.Tensor): Labels to be modified.
        ignore_index (int, optional): The label index to be ignored.
            Defaults to None.

    Return:
        tuple(torch.Tensor, torch.Tensor): Modified predictions and labels.
    """
    logits = logits.view(-1)
    labels = labels.view(-1)
    if ignore_index is None:
        return logits, labels
    valid = (labels != ignore_index)
    vlogits = logits[valid]
    vlabels = labels[valid]
    return vlogits, vlabels


def flatten_probs(
        probs: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten predictions and labels in the batch. Remove tensors whose labels
    equal to 'ignore_index'.

    Args:
        probs (torch.Tensor): Predictions to be modified.
        labels (torch.Tensor): Labels to be modified.
        ignore_index (int, optional): The label index to be ignored.
            Defaults to None.

    Return:
        tuple(torch.Tensor, torch.Tensor): Modified predictions and labels.
    """
    if probs.dim() != 2:  # for input with P*C
        if probs.dim() == 3:
            # assumes output of a sigmoid layer
            B, H, W = probs.size()
            probs = probs.view(B, 1, H, W)
        B, C, H, W = probs.size()
        probs = probs.permute(0, 2, 3, 1).contiguous().view(-1,
                                                            C)  # B*H*W, C=P,C
        labels = labels.view(-1)
    if ignore_index is None:
        return probs, labels
    valid = (labels != ignore_index)
    vprobs = probs[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobs, vlabels


def lovasz_hinge_flat(logits: torch.Tensor,
                      labels: torch.Tensor) -> torch.Tensor:
    """Binary Lovasz hinge loss.

    Args:
        logits (torch.Tensor): Logits at each prediction
            (between -infty and +infty) with shape [P].
        labels (torch.Tensor): Binary ground truth labels (0 or 1)
            with shape [P].

    Returns:
        torch.Tensor: The calculated loss.
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def lovasz_hinge(logits: torch.Tensor,
                 labels: torch.Tensor,
                 classes: Optional[Union[str, List[int]]] = None,
                 per_sample: bool = False,
                 class_weight: Optional[List[float]] = None,
                 reduction: str = 'mean',
                 avg_factor: Optional[int] = None,
                 ignore_index: int = 255) -> torch.Tensor:
    """Binary Lovasz hinge loss.

    Args:
        logits (torch.Tensor): Logits at each pixel
            (between -infty and +infty) with shape [B, H, W].
        labels (torch.Tensor): Binary ground truth masks (0 or 1)
            with shape [B, H, W].
        classes (Union[str, list[int]], optional): Placeholder, to be
            consistent with other loss. Defaults to None.
        per_sample (bool): If per_sample is True, compute the loss per
            sample instead of per batch. Defaults to False.
        class_weight (list[float], optional): Placeholder, to be consistent
            with other loss. Defaults to None.
        reduction (str): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_sample is True. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. This parameter only works when per_sample is True.
            Defaults to None.
        ignore_index (Union[int, None]): The label index to be ignored.
            Defaults to 255.

    Returns:
        torch.Tensor: The calculated loss.
    """
    if per_sample:
        loss = [
            lovasz_hinge_flat(*flatten_binary_logits(
                logit.unsqueeze(0), label.unsqueeze(0), ignore_index))
            for logit, label in zip(logits, labels)
        ]
        loss = weight_reduce_loss(
            torch.stack(loss), None, reduction, avg_factor)
    else:
        loss = lovasz_hinge_flat(
            *flatten_binary_logits(logits, labels, ignore_index))
    return loss


def lovasz_softmax_flat(
        probs: torch.Tensor,
        labels: torch.Tensor,
        classes: Union[str, List[int]] = 'present',
        class_weight: Optional[List[float]] = None) -> torch.Tensor:
    """Multi-class Lovasz-Softmax loss.

    Args:
        probs (torch.Tensor): Class probabilities at each prediction
            (between 0 and 1) with shape [P, C]
        labels (torch.Tensor): Ground truth labels (between 0 and C - 1)
            with shape [P].
        classes (Union[str, list[int]]): Classes chosen to calculate loss.
            'all' for all classes, 'present' for classes present in labels, or
            a list of classes to average. Defaults to 'present'.
        class_weight (list[float], optional): The weight for each class.
            Defaults to None.

    Returns:
        torch.Tensor: The calculated loss.
    """
    if probs.numel() == 0:
        # only void pixels, the gradients should be 0
        return probs * 0.
    C = probs.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probs[:, 0]
        else:
            class_pred = probs[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        loss = torch.dot(errors_sorted, lovasz_grad(fg_sorted))
        if class_weight is not None:
            loss *= class_weight[c]
        losses.append(loss)
    return torch.stack(losses).mean()


def lovasz_softmax(probs: torch.Tensor,
                   labels: torch.Tensor,
                   classes: Union[str, List[int]] = 'present',
                   per_sample: bool = False,
                   class_weight: List[float] = None,
                   reduction: str = 'mean',
                   avg_factor: Optional[int] = None,
                   ignore_index: int = 255) -> torch.Tensor:
    """Multi-class Lovasz-Softmax loss.

    Args:
        probs (torch.Tensor): Class probabilities at each
            prediction (between 0 and 1) with shape [B, C, H, W].
        labels (torch.Tensor): Ground truth labels (between 0 and
            C - 1) with shape [B, H, W].
        classes (Union[str, list[int]]): Classes chosen to calculate loss.
            'all' for all classes, 'present' for classes present in labels, or
            a list of classes to average. Defaults to 'present'.
        per_sample (bool): If per_sample is True, compute the loss per
            sample instead of per batch. Defaults to False.
        class_weight (list[float], optional): The weight for each class.
            Defaults to None.
        reduction (str): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_sample is True. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. This parameter only works when per_sample is True.
            Defaults to None.
        ignore_index (Union[int, None]): The label index to be ignored.
            Defaults to 255.

    Returns:
        torch.Tensor: The calculated loss.
    """

    if per_sample:
        loss = [
            lovasz_softmax_flat(
                *flatten_probs(
                    prob.unsqueeze(0), label.unsqueeze(0), ignore_index),
                classes=classes,
                class_weight=class_weight)
            for prob, label in zip(probs, labels)
        ]
        loss = weight_reduce_loss(
            torch.stack(loss), None, reduction, avg_factor)
    else:
        loss = lovasz_softmax_flat(
            *flatten_probs(probs, labels, ignore_index),
            classes=classes,
            class_weight=class_weight)
    return loss


@MODELS.register_module()
class LovaszLoss(nn.Module):
    """LovaszLoss.

    This loss is proposed in `The Lovasz-Softmax loss: A tractable surrogate
    for the optimization of the intersection-over-union measure in neural
    networks <https://arxiv.org/abs/1705.08790>`_.

    Args:
        loss_type (str): Binary or multi-class loss.
            Defaults to 'multi_class'. Options are "binary" and "multi_class".
        classes (Union[str, list[int]]): Classes chosen to calculate loss.
            'all' for all classes, 'present' for classes present in labels, or
            a list of classes to average. Defaults to 'present'.
        per_sample (bool): If per_sample is True, compute the loss per
            sample instead of per batch. Defaults to False.
        reduction (str): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_sample is True. Defaults to 'mean'.
        class_weight ([list[float], optional): Weight of each class.
            Defaults to None.
        loss_weight (float): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 loss_type: str = 'multi_class',
                 classes: Union[str, List[int]] = 'present',
                 per_sample: bool = False,
                 reduction: str = 'mean',
                 class_weight: Optional[List[float]] = None,
                 loss_weight: float = 1.0):
        super().__init__()
        assert loss_type in ('binary', 'multi_class'), "loss_type should be \
                                                    'binary' or 'multi_class'."

        if loss_type == 'binary':
            self.cls_criterion = lovasz_hinge
        else:
            self.cls_criterion = lovasz_softmax
        assert classes in ('all', 'present') or is_list_of(classes, int)
        if not per_sample:
            assert reduction == 'none', "reduction should be 'none' when \
                                                        per_sample is False."

        self.classes = classes
        self.per_sample = per_sample
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

    def forward(self,
                cls_score: torch.Tensor,
                label: torch.Tensor,
                avg_factor: int = None,
                reduction_override: str = None,
                **kwargs) -> torch.Tensor:
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None

        # if multi-class loss, transform logits to probs
        if self.cls_criterion == lovasz_softmax:
            cls_score = F.softmax(cls_score, dim=1)

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            self.classes,
            self.per_sample,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls
