# Copyright (c) OpenMMLab. All rights reserved.
import collections

from mmcv.utils import build_from_cfg

from mmdet.datasets.builder import PIPELINES as MMDET_PIPELINES
from ..builder import PIPELINES


@PIPELINES.register_module()
class Compose:
    """Compose multiple transforms sequentially. The pipeline registry of
    mmdet3d separates with mmdet, however, sometimes we may need to use mmdet's
    pipeline. So the class is rewritten to be able to use pipelines from both
    mmdet3d and mmdet.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                _, key = PIPELINES.split_scope_key(transform['type'])
                if key in PIPELINES._module_dict.keys():
                    transform = build_from_cfg(transform, PIPELINES)
                else:
                    transform = build_from_cfg(transform, MMDET_PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
