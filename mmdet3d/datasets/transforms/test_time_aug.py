# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import mmengine
from mmcv import BaseTransform
from mmengine.dataset import Compose

from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class MultiScaleFlipAug3D(BaseTransform):
    """Test-time augmentation with multiple scales and flipping.

    Args:
        transforms (list[dict]): Transforms to apply in each augmentation.
        img_scale (tuple | list[tuple]): Images scales for resizing.
        pts_scale_ratio (float | list[float]): Points scale ratios for
            resizing.
        flip (bool): Whether apply flip augmentation. Defaults to False.
        flip_direction (str | list[str]): Flip augmentation directions
            for images, options are "horizontal" and "vertical".
            If flip_direction is list, multiple flip augmentations will
            be applied. It has no effect when ``flip == False``.
            Defaults to 'horizontal'.
        pcd_horizontal_flip (bool): Whether to apply horizontal flip
            augmentation to point cloud. Defaults to False.
            Note that it works only when 'flip' is turned on.
        pcd_vertical_flip (bool): Whether to apply vertical flip
            augmentation to point cloud. Defaults to False.
            Note that it works only when 'flip' is turned on.
    """

    def __init__(self,
                 transforms: List[dict],
                 img_scale: Optional[Union[Tuple[int], List[Tuple[int]]]],
                 pts_scale_ratio: Union[float, List[float]],
                 flip: bool = False,
                 flip_direction: str = 'horizontal',
                 pcd_horizontal_flip: bool = False,
                 pcd_vertical_flip: bool = False) -> None:
        self.transforms = Compose(transforms)
        self.img_scale = img_scale if isinstance(img_scale,
                                                 list) else [img_scale]
        self.pts_scale_ratio = pts_scale_ratio \
            if isinstance(pts_scale_ratio, list) else [float(pts_scale_ratio)]

        assert mmengine.is_list_of(self.img_scale, tuple)
        assert mmengine.is_list_of(self.pts_scale_ratio, float)

        self.flip = flip
        self.pcd_horizontal_flip = pcd_horizontal_flip
        self.pcd_vertical_flip = pcd_vertical_flip

        self.flip_direction = flip_direction if isinstance(
            flip_direction, list) else [flip_direction]
        assert mmengine.is_list_of(self.flip_direction, str)
        if not self.flip and self.flip_direction != ['horizontal']:
            warnings.warn(
                'flip_direction has no effect when flip is set to False')
        if (self.flip and not any([(t['type'] == 'RandomFlip3D'
                                    or t['type'] == 'RandomFlip')
                                   for t in transforms])):
            warnings.warn(
                'flip has no effect when RandomFlip is not in transforms')

    def transform(self, results: Dict) -> List[Dict]:
        """Call function to augment common fields in results.

        Args:
            results (dict): Result dict contains the data to augment.

        Returns:
            List[dict]: The list contains the data that is augmented with
            different scales and flips.
        """
        aug_data_list = []

        # modified from `flip_aug = [False, True] if self.flip else [False]`
        # to reduce unnecessary scenes when using double flip augmentation
        # during test time
        flip_aug = [True] if self.flip else [False]
        pcd_horizontal_flip_aug = [False, True] \
            if self.flip and self.pcd_horizontal_flip else [False]
        pcd_vertical_flip_aug = [False, True] \
            if self.flip and self.pcd_vertical_flip else [False]
        for scale in self.img_scale:
            # TODO refactor according to augtest docs
            self.transforms.transforms[0].scale = scale
            for pts_scale_ratio in self.pts_scale_ratio:
                for flip in flip_aug:
                    for pcd_horizontal_flip in pcd_horizontal_flip_aug:
                        for pcd_vertical_flip in pcd_vertical_flip_aug:
                            for direction in self.flip_direction:
                                # results.copy will cause bug
                                # since it is shallow copy
                                _results = deepcopy(results)
                                _results['scale'] = scale
                                _results['flip'] = flip
                                _results['pcd_scale_factor'] = \
                                    pts_scale_ratio
                                _results['flip_direction'] = direction
                                _results['pcd_horizontal_flip'] = \
                                    pcd_horizontal_flip
                                _results['pcd_vertical_flip'] = \
                                    pcd_vertical_flip
                                data = self.transforms(_results)
                                aug_data_list.append(data)

        return aug_data_list

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'img_scale={self.img_scale}, flip={self.flip}, '
        repr_str += f'pts_scale_ratio={self.pts_scale_ratio}, '
        repr_str += f'flip_direction={self.flip_direction})'
        return repr_str
