# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import mmdet
import mmengine
from mmengine.utils import digit_version

from .version import __version__, version_info

mmcv_minimum_version = '2.0.0rc0'
mmcv_maximum_version = '2.1.0'
mmcv_version = digit_version(mmcv.__version__)

mmengine_minimum_version = '0.1.0'
mmengine_maximum_version = '1.0.0'
mmengine_version = digit_version(mmengine.__version__)

mmdet_minimum_version = '3.0.0rc0'
mmdet_maximum_version = '3.1.0'
mmdet_version = digit_version(mmdet.__version__)

assert (mmcv_version >= digit_version(mmcv_minimum_version)
        and mmcv_version < digit_version(mmcv_maximum_version)), \
    f'MMCV=={mmcv.__version__} is used but incompatible. ' \
    f'Please install mmcv>={mmcv_minimum_version}, <{mmcv_maximum_version}.'

assert (mmengine_version >= digit_version(mmengine_minimum_version)
        and mmengine_version < digit_version(mmengine_maximum_version)), \
    f'MMEngine=={mmengine.__version__} is used but incompatible. ' \
    f'Please install mmengine>={mmengine_minimum_version}, ' \
    f'<{mmengine_maximum_version}.'

assert (mmdet_version >= digit_version(mmdet_minimum_version)
        and mmdet_version < digit_version(mmdet_maximum_version)), \
    f'MMDET=={mmdet.__version__} is used but incompatible. ' \
    f'Please install mmdet>={mmdet_minimum_version}, ' \
    f'<{mmdet_maximum_version}.'

__all__ = ['__version__', 'version_info', 'digit_version']
