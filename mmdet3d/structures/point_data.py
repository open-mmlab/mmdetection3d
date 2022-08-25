# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Sized
from typing import Union

import numpy as np
import torch
from mmengine.structures import BaseDataElement

IndexType = Union[str, slice, int, list, torch.LongTensor,
                  torch.cuda.LongTensor, torch.BoolTensor,
                  torch.cuda.BoolTensor, np.ndarray]


class PointData(BaseDataElement):
    """Data structure for point-level annnotations or predictions.

    All data items in ``data_fields`` of ``PointData`` meet the following
    requirements:

    - They are all one dimension.
    - They should have the same length.

    `PointData` is used to save point-level semantic and instance mask,
    it also can save `instances_labels` and `instances_scores` temporarily.
    In the future, we would consider to move the instance-level info into
    `gt_instances_3d` and `pred_instances_3d`.

    Examples:
        >>> metainfo = dict(
        ...     sample_id=random.randint(0, 100))
        >>> points = np.random.randint(0, 255, (100, 3))
        >>> point_data = PointData(metainfo=metainfo,
        ...                        points=points)
        >>> print(len(point_data))
        >>> (100)

        >>> # slice
        >>> slice_data = pixel_data[10:60]
        >>> assert slice_data.shape == (50,)

        >>> # set
        >>> point_data.pts_semantic_mask = torch.randint(0, 255, (100))
        >>> point_data.pts_instance_mask = torch.randint(0, 255, (100))
        >>> assert tuple(point_data.pts_semantic_mask.shape) == (100)
        >>> assert tuple(point_data.pts_instance_mask.shape) == (100)
    """

    def __setattr__(self, name: str, value: Sized):
        """setattr is only used to set data.

        the value must have the attribute of `__len__` and have the same length
        of PointData.
        """
        if name in ('_metainfo_fields', '_data_fields'):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(
                    f'{name} has been used as a '
                    f'private attribute, which is immutable. ')

        else:
            assert isinstance(value,
                              Sized), 'value must contain `_len__` attribute'
            super().__setattr__(name, value)

    __setitem__ = __setattr__

    def __getitem__(self, item: IndexType) -> 'PointData':
        """
        Args:
            item (str, obj:`slice`,
                obj`torch.LongTensor`, obj:`torch.BoolTensor`):
                get the corresponding values according to item.

        Returns:
            obj:`PointData`: Corresponding values.
        """
        if isinstance(item, list):
            item = np.array(item)
        if isinstance(item, np.ndarray):
            item = torch.from_numpy(item)
        assert isinstance(
            item, (str, slice, int, torch.LongTensor, torch.cuda.LongTensor,
                   torch.BoolTensor, torch.cuda.BoolTensor))

        if isinstance(item, str):
            return getattr(self, item)

        if type(item) == int:
            if item >= len(self) or item < -len(self):  # type:ignore
                raise IndexError(f'Index {item} out of range!')
            else:
                # keep the dimension
                item = slice(item, None, len(self))

        new_data = self.__class__(metainfo=self.metainfo)
        if isinstance(item, torch.Tensor):
            assert item.dim() == 1, 'Only support to get the' \
                                    ' values along the first dimension.'
            if isinstance(item, (torch.BoolTensor, torch.cuda.BoolTensor)):
                assert len(item) == len(self), f'The shape of the' \
                                               f' input(BoolTensor)) ' \
                                               f'{len(item)} ' \
                                               f' does not match the shape ' \
                                               f'of the indexed tensor ' \
                                               f'in results_filed ' \
                                               f'{len(self)} at ' \
                                               f'first dimension. '

            for k, v in self.items():
                if isinstance(v, torch.Tensor):
                    new_data[k] = v[item]
                elif isinstance(v, np.ndarray):
                    new_data[k] = v[item.cpu().numpy()]
                elif isinstance(
                        v, (str, list, tuple)) or (hasattr(v, '__getitem__')
                                                   and hasattr(v, 'cat')):
                    # convert to indexes from boolTensor
                    if isinstance(item,
                                  (torch.BoolTensor, torch.cuda.BoolTensor)):
                        indexes = torch.nonzero(item).view(
                            -1).cpu().numpy().tolist()
                    else:
                        indexes = item.cpu().numpy().tolist()
                    slice_list = []
                    if indexes:
                        for index in indexes:
                            slice_list.append(slice(index, None, len(v)))
                    else:
                        slice_list.append(slice(None, 0, None))
                    r_list = [v[s] for s in slice_list]
                    if isinstance(v, (str, list, tuple)):
                        new_value = r_list[0]
                        for r in r_list[1:]:
                            new_value = new_value + r
                    else:
                        new_value = v.cat(r_list)
                    new_data[k] = new_value
                else:
                    raise ValueError(
                        f'The type of `{k}` is `{type(v)}`, which has no '
                        'attribute of `cat`, so it does not '
                        f'support slice with `bool`')

        else:
            # item is a slice
            for k, v in self.items():
                new_data[k] = v[item]
        return new_data  # type:ignore

    def __len__(self) -> int:
        """int: the length of PointData"""
        if len(self._data_fields) > 0:
            return len(self.values()[0])
        else:
            return 0
