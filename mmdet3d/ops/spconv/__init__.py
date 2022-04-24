# Copyright (c) OpenMMLab. All rights reserved.
from .overwrite_spconv.write_spconv2 import register_spconv2

try:
    import spconv
except ImportError:
    spconv2_is_avalible = False
else:
    if hasattr(spconv, '__version__') and spconv.__version__ >= '2.0.0':
        register_spconv2()
        spconv2_is_avalible = True
    else:
        spconv2_is_avalible = False

__all__ = ['spconv2_is_avalible']
