# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import collect_env as collect_base_env
from mmcv.utils import get_git_hash

import mmdet
import mmdet3d
import mmseg
from mmdet3d.ops.spconv import spconv2_is_avalible


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info['MMDetection'] = mmdet.__version__
    env_info['MMSegmentation'] = mmseg.__version__
    env_info['MMDetection3D'] = mmdet3d.__version__ + '+' + get_git_hash()[:7]
    env_info['Spconv2 is avalible'] = spconv2_is_avalible
    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')
