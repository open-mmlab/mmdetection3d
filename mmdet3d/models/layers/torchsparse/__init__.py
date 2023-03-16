# Copyright (c) OpenMMLab. All rights reserved.
from .torchsparse_wrapper import register_torchsparse

try:
    import torchsparse  # noqa
except ImportError:
    IS_TORCHSPARSE_AVAILABLE = False
else:
    IS_TORCHSPARSE_AVAILABLE = register_torchsparse()

__all__ = ['IS_TORCHSPARSE_AVAILABLE']
