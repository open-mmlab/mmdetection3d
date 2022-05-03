# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import collect_env as collect_base_env
from mmcv.utils import get_git_hash

import mmdet
import mmdet3d
import mmseg
from mmdet3d.ops.spconv import IS_SPCONV2_AVAILABLE


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info['MMDetection'] = mmdet.__version__
    env_info['MMSegmentation'] = mmseg.__version__
    env_info['MMDetection3D'] = mmdet3d.__version__ + '+' + get_git_hash()[:7]
    env_info['spconv2.0'] = IS_SPCONV2_AVAILABLE
    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')
