# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmdet.models.task_modules import BaseBBoxCoder

from mmdet3d.registry import TASK_UTILS


@TASK_UTILS.register_module()
class PointXYZWHLRBBoxCoder(BaseBBoxCoder):
    """Point based bbox coder for 3D boxes.

    Args:
        code_size (int): The dimension of boxes to be encoded.
        use_mean_size (bool, optional): Whether using anchors based on class.
            Defaults to True.
        mean_size (list[list[float]], optional): Mean size of bboxes in
            each class. Defaults to None.
    """

    def __init__(self, code_size=7, use_mean_size=True, mean_size=None):
        super(PointXYZWHLRBBoxCoder, self).__init__()
        self.code_size = code_size
        self.use_mean_size = use_mean_size
        if self.use_mean_size:
            self.mean_size = torch.from_numpy(np.array(mean_size)).float()
            assert self.mean_size.min() > 0, \
                f'The min of mean_size should > 0, however currently it is '\
                f'{self.mean_size.min()}, please check it in your config.'

    def encode(self, gt_bboxes_3d, points, gt_labels_3d=None):
        """Encode ground truth to prediction targets.

        Args:
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): Ground truth bboxes
                with shape (N, 7 + C).
            points (torch.Tensor): Point cloud with shape (N, 3).
            gt_labels_3d (torch.Tensor, optional): Ground truth classes.
                Defaults to None.

        Returns:
            torch.Tensor: Encoded boxes with shape (N, 8 + C).
        """
        gt_bboxes_3d[:, 3:6] = torch.clamp_min(gt_bboxes_3d[:, 3:6], min=1e-5)

        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(
            gt_bboxes_3d, 1, dim=-1)
        xa, ya, za = torch.split(points, 1, dim=-1)

        if self.use_mean_size:
            assert gt_labels_3d.max() <= self.mean_size.shape[0] - 1, \
                f'the max gt label {gt_labels_3d.max()} is bigger than' \
                f'anchor types {self.mean_size.shape[0] - 1}.'
            self.mean_size = self.mean_size.to(gt_labels_3d.device)
            point_anchor_size = self.mean_size[gt_labels_3d]
            dxa, dya, dza = torch.split(point_anchor_size, 1, dim=-1)
            diagonal = torch.sqrt(dxa**2 + dya**2)
            xt = (xg - xa) / diagonal
            yt = (yg - ya) / diagonal
            zt = (zg - za) / dza
            dxt = torch.log(dxg / dxa)
            dyt = torch.log(dyg / dya)
            dzt = torch.log(dzg / dza)
        else:
            xt = (xg - xa)
            yt = (yg - ya)
            zt = (zg - za)
            dxt = torch.log(dxg)
            dyt = torch.log(dyg)
            dzt = torch.log(dzg)

        return torch.cat(
            [xt, yt, zt, dxt, dyt, dzt,
             torch.cos(rg),
             torch.sin(rg), *cgs],
            dim=-1)

    def decode(self, box_encodings, points, pred_labels_3d=None):
        """Decode predicted parts and points to bbox3d.

        Args:
            box_encodings (torch.Tensor): Encoded boxes with shape (N, 8 + C).
            points (torch.Tensor): Point cloud with shape (N, 3).
            pred_labels_3d (torch.Tensor): Bbox predicted labels (N, M).

        Returns:
            torch.Tensor: Decoded boxes with shape (N, 7 + C)
        """
        xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(
            box_encodings, 1, dim=-1)
        xa, ya, za = torch.split(points, 1, dim=-1)

        if self.use_mean_size:
            assert pred_labels_3d.max() <= self.mean_size.shape[0] - 1, \
                f'The max pred label {pred_labels_3d.max()} is bigger than' \
                f'anchor types {self.mean_size.shape[0] - 1}.'
            self.mean_size = self.mean_size.to(pred_labels_3d.device)
            point_anchor_size = self.mean_size[pred_labels_3d]
            dxa, dya, dza = torch.split(point_anchor_size, 1, dim=-1)
            diagonal = torch.sqrt(dxa**2 + dya**2)
            xg = xt * diagonal + xa
            yg = yt * diagonal + ya
            zg = zt * dza + za

            dxg = torch.exp(dxt) * dxa
            dyg = torch.exp(dyt) * dya
            dzg = torch.exp(dzt) * dza
        else:
            xg = xt + xa
            yg = yt + ya
            zg = zt + za
            dxg, dyg, dzg = torch.split(
                torch.exp(box_encodings[..., 3:6]), 1, dim=-1)

        rg = torch.atan2(sint, cost)

        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cts], dim=-1)
