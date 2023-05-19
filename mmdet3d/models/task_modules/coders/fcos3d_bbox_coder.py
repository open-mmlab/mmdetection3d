# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import numpy as np
import torch
from mmdet.models.task_modules import BaseBBoxCoder
from torch import Tensor

from mmdet3d.registry import TASK_UTILS
from mmdet3d.structures.bbox_3d import limit_period


@TASK_UTILS.register_module()
class FCOS3DBBoxCoder(BaseBBoxCoder):
    """Bounding box coder for FCOS3D.

    Args:
        base_depths (tuple[tuple[float]]): Depth references for decode box
            depth. Defaults to None.
        base_dims (tuple[tuple[float]]): Dimension references for decode box
            dimension. Defaults to None.
        code_size (int): The dimension of boxes to be encoded. Defaults to 7.
        norm_on_bbox (bool): Whether to apply normalization on the bounding
            box 2D attributes. Defaults to True.
    """

    def __init__(self,
                 base_depths: Optional[Tuple[Tuple[float]]] = None,
                 base_dims: Optional[Tuple[Tuple[float]]] = None,
                 code_size: int = 7,
                 norm_on_bbox: bool = True) -> None:
        super(FCOS3DBBoxCoder, self).__init__()
        self.base_depths = base_depths
        self.base_dims = base_dims
        self.bbox_code_size = code_size
        self.norm_on_bbox = norm_on_bbox

    def encode(self, gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels):
        # TODO: refactor the encoder in the FCOS3D and PGD head
        pass

    def decode(self,
               bbox: Tensor,
               scale: tuple,
               stride: int,
               training: bool,
               cls_score: Optional[Tensor] = None) -> Tensor:
        """Decode regressed results into 3D predictions.

        Note that offsets are not transformed to the projected 3D centers.

        Args:
            bbox (torch.Tensor): Raw bounding box predictions in shape
                [N, C, H, W].
            scale (tuple[`Scale`]): Learnable scale parameters.
            stride (int): Stride for a specific feature level.
            training (bool): Whether the decoding is in the training
                procedure.
            cls_score (torch.Tensor): Classification score map for deciding
                which base depth or dim is used. Defaults to None.

        Returns:
            torch.Tensor: Decoded boxes.
        """
        # scale the bbox of different level
        # only apply to offset, depth and size prediction
        scale_offset, scale_depth, scale_size = scale[0:3]

        clone_bbox = bbox.clone()
        bbox[:, :2] = scale_offset(clone_bbox[:, :2]).float()
        bbox[:, 2] = scale_depth(clone_bbox[:, 2]).float()
        bbox[:, 3:6] = scale_size(clone_bbox[:, 3:6]).float()

        if self.base_depths is None:
            bbox[:, 2] = bbox[:, 2].exp()
        elif len(self.base_depths) == 1:  # only single prior
            mean = self.base_depths[0][0]
            std = self.base_depths[0][1]
            bbox[:, 2] = mean + bbox.clone()[:, 2] * std
        else:  # multi-class priors
            assert len(self.base_depths) == cls_score.shape[1], \
                'The number of multi-class depth priors should be equal to ' \
                'the number of categories.'
            indices = cls_score.max(dim=1)[1]
            depth_priors = cls_score.new_tensor(
                self.base_depths)[indices, :].permute(0, 3, 1, 2)
            mean = depth_priors[:, 0]
            std = depth_priors[:, 1]
            bbox[:, 2] = mean + bbox.clone()[:, 2] * std

        bbox[:, 3:6] = bbox[:, 3:6].exp()
        if self.base_dims is not None:
            assert len(self.base_dims) == cls_score.shape[1], \
                'The number of anchor sizes should be equal to the number ' \
                'of categories.'
            indices = cls_score.max(dim=1)[1]
            size_priors = cls_score.new_tensor(
                self.base_dims)[indices, :].permute(0, 3, 1, 2)
            bbox[:, 3:6] = size_priors * bbox.clone()[:, 3:6]

        assert self.norm_on_bbox is True, 'Setting norm_on_bbox to False '\
            'has not been thoroughly tested for FCOS3D.'
        if self.norm_on_bbox:
            if not training:
                # Note that this line is conducted only when testing
                bbox[:, :2] *= stride

        return bbox

    @staticmethod
    def decode_yaw(bbox: Tensor, centers2d: Tensor, dir_cls: Tensor,
                   dir_offset: float, cam2img: Tensor) -> Tensor:
        """Decode yaw angle and change it from local to global.i.

        Args:
            bbox (torch.Tensor): Bounding box predictions in shape
                [N, C] with yaws to be decoded.
            centers2d (torch.Tensor): Projected 3D-center on the image planes
                corresponding to the box predictions.
            dir_cls (torch.Tensor): Predicted direction classes.
            dir_offset (float): Direction offset before dividing all the
                directions into several classes.
            cam2img (torch.Tensor): Camera intrinsic matrix in shape [4, 4].

        Returns:
            torch.Tensor: Bounding boxes with decoded yaws.
        """
        if bbox.shape[0] > 0:
            dir_rot = limit_period(bbox[..., 6] - dir_offset, 0, np.pi)
            bbox[..., 6] = \
                dir_rot + dir_offset + np.pi * dir_cls.to(bbox.dtype)

        bbox[:, 6] = torch.atan2(centers2d[:, 0] - cam2img[0, 2],
                                 cam2img[0, 0]) + bbox[:, 6]

        return bbox
