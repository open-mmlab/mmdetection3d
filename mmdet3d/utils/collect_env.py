# Copyright (c) OpenMMLab. All rights reserved.
import mmdet
from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env as collect_base_env

import mmdet3d


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info['MMDetection'] = mmdet.__version__
    env_info['MMDetection3D'] = mmdet3d.__version__ + '+' + get_git_hash()[:7]
    from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
    env_info['spconv2.0'] = IS_SPCONV2_AVAILABLE

    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')
