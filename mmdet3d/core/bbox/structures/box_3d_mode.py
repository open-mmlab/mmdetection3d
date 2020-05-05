from enum import IntEnum, unique

import numpy as np
import torch


@unique
class Box3DMode(IntEnum):
    """
    Enum of different ways to represent a box.
    """

    LIDAR = 0
    """
    Coordinates in velodyne/LiDAR sensors.
                up z    x front
                   ^   ^
                   |  /
                   | /
    left y <------ 0
    """
    CAM = 1
    """
    Coordinates in camera.
                       x right
                      /
                     /
    front z <------ 0
                    |
                    |
                    v
               down y
    """
    DEPTH = 2
    """
    Coordinates in Depth mode.
                 up z   x right
                    ^   ^
                    |  /
                    | /
    front y <------ 0
    """

    @staticmethod
    def convert(box, from_mode, to_mode):
        """
        Args:
            box: can be a k-tuple, k-list or an Nxk array/tensor, where k = 7
            from_mode, to_mode (BoxMode)

        Returns:
            The converted box of the same type.
        """
        if from_mode == to_mode:
            return box

        original_type = type(box)
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

        if single_box:
            return original_type(arr.flatten().tolist())
        if is_numpy:
            return arr.numpy()
        else:
            return arr
