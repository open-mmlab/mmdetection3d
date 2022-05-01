# Copyright (c) OpenMMLab. All rights reserved.
from .overwrite_spconv.write_spconv2 import register_spconv2

try:
    import spconv
except ImportError:
    IS_SPCONV2_AVAILABLE = False
else:
    if hasattr(spconv, '__version__') and spconv.__version__ >= '2.0.0':
        IS_SPCONV2_AVAILABLE = register_spconv2()
    else:
        IS_SPCONV2_AVAILABLE = False

__all__ = ['IS_SPCONV2_AVAILABLE']
