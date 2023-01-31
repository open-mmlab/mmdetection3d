# Copyright (c) OpenMMLab. All rights reserved.
import os
from collections import OrderedDict
from os import path as osp
from typing import List, Tuple, Union

import mmcv
import numpy as np
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box

from mmdet3d.core.bbox import points_cam2img


def create_once_infos(root_path,
                      info_prefix,
                      version,
                      max_sweeps=10):
    """Create info file of once dataset.
    
    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str, optional): Version of the data.
        max_sweeps (int, optional): Max number of sweeps.
            Default: 10.
    """
    pass