# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from abc import ABCMeta
from typing import Optional, Tuple

from mmengine.model import BaseModule
from torch import Tensor

from mmdet3d.utils import OptMultiConfig


class BasePointNet(BaseModule, metaclass=ABCMeta):
    """Base class for PointNet."""

    def __init__(self,
                 init_cfg: OptMultiConfig = None,
                 pretrained: Optional[str] = None):
        super(BasePointNet, self).__init__(init_cfg)
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

    @staticmethod
    def _split_point_feats(points: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Split coordinates and features of input points.

        Args:
            points (torch.Tensor): Point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).

        Returns:
            torch.Tensor: Coordinates of input points.
            torch.Tensor: Features of input points.
        """
        xyz = points[..., 0:3].contiguous()
        if points.size(-1) > 3:
            features = points[..., 3:].transpose(1, 2).contiguous()
        else:
            features = None

        return xyz, features
