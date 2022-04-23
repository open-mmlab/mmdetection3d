# Copyright (c) OpenMMLab. All rights reserved.
from .overwrite_spconv import write_spconv

try:
    import spconv
except ImportError:
    spconv2_is_avalible = False
else:
    if hasattr(spconv, '__version__') and spconv.__version__ >= '2.0.0':
        write_spconv()
        spconv2_is_avalible = True
    else:
        spconv2_is_avalible = False

__all__ = ['spconv2_is_avalible']
