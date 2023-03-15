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
        MODELS._register_module(Conv3d, 'TorchSparseConv3d')
        MODELS._register_module(BatchNorm, 'TorchSparseBatchNorm')
        MODELS._register_module(GroupNorm, 'TorchSparseGroupNorm')
        MODELS._register_module(ReLU, 'TorchSparseReLU')
        MODELS._register_module(LeakyReLU, 'TorchSparseLeakyReLU')
        return True
