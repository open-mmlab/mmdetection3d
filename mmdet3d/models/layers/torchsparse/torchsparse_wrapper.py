# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.registry import MODELS


def register_torchsparse() -> bool:
    """This func registers torchsparse modules."""
    try:
        from torchsparse.nn import (BatchNorm, Conv3d, GroupNorm, LeakyReLU,
                                    ReLU)
    except ImportError:
        return False
    else:
        MODELS._register_module(Conv3d, 'TorchSparse.Conv3d')
        MODELS._register_module(BatchNorm, 'TorchSparse.BatchNorm')
        MODELS._register_module(GroupNorm, 'TorchSparse.GroupNorm')
        MODELS._register_module(ReLU, 'TorchSparse.ReLU')
        MODELS._register_module(LeakyReLU, 'TorchSparse.LeakyReLU')
        return True
