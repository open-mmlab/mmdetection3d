from enum import IntEnum, unique

import numpy as np
import torch


@unique
class Box3DMode(IntEnum):
    r"""Enum of different ways to represent a box.

    Coordinates in LiDAR:
    .. code-block:: none
                    up z    x front
                       ^   ^
                       |  /
                       | /
        left y <------ 0
    The relative coordinate of bottom center in a LiDAR box is [0.5, 0.5, 0],
    and the yaw is around the z axis, thus the rotation axis=2.

    Coordinates in camera:
    .. code-block:: none
                           x right
                          /
                         /
        front z <------ 0
                        |
                        |
                        v
                   down y
    The relative coordinate of bottom center in a CAM box is [0.5, 1.0, 0.5],
    and the yaw is around the y axis, thus the rotation axis=1.

    Coordinates in Depth mode:
    .. code-block:: none
                     up z   x right
                        ^   ^
                        |  /
                        | /
        front y <------ 0
    The relative coordinate of bottom center in a DEPTH box is [0.5, 0.5, 0],
    and the yaw is around the z axis, thus the rotation axis=2.
    """

    LIDAR = 0
    CAM = 1
    DEPTH = 2

    @staticmethod
    def convert(box, src, dst):
        """Convert boxes from `src` mode to `dst` mode.

        Args:
            box (tuple | list | np.ndarray | torch.Tensor):
                can be a k-tuple, k-list or an Nxk array/tensor, where k = 7
            src (BoxMode): the src Box mode
            dst (BoxMode): the target Box mode

        Returns:
            The converted box of the same type.
        """
        if src == dst:
            return box

        is_numpy = isinstance(box, np.ndarray)
        single_box = isinstance(box, (list, tuple))
        if single_box:
            assert len(box) >= 7, (
                'BoxMode.convert takes either a k-tuple/list or '
                'an Nxk array/tensor, where k >= 7')
            arr = torch.tensor(box)[None, :]
        else:
            # avoid modifying the input box
            if is_numpy:
                arr = torch.from_numpy(np.asarray(box)).clone()
            else:
                arr = box.clone()

        # converting logic
        # TODO: add converting logic to support box conversion
        original_type = type(box)
        if single_box:
            return original_type(arr.flatten().tolist())
        if is_numpy:
            return arr.numpy()
        else:
            return arr
