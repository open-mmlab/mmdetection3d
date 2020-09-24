from abc import ABCMeta
from mmcv.runner import load_checkpoint
from torch import nn as nn


class BasePointNet(nn.Module, metaclass=ABCMeta):
    """Base class for PointNet."""

    def __init__(self):
        super(BasePointNet, self).__init__()
        self.fp16_enabled = False

    def init_weights(self, pretrained=None):
        """Initialize the weights of PointNet backbone."""
        # Do not initialize the conv layers
        # to follow the original implementation
        if isinstance(pretrained, str):
            from mmdet3d.utils import get_root_logger
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    @staticmethod
    def _split_point_feats(points):
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
