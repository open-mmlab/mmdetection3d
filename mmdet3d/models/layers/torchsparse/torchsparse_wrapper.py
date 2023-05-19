# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmengine.registry import MODELS


def register_torchsparse() -> bool:
    """This func registers torchsparse modules."""
    try:
        from torchsparse.nn import (BatchNorm, Conv3d, GroupNorm, LeakyReLU,
                                    ReLU)
        from torchsparse.nn.utils import fapply
        from torchsparse.tensor import SparseTensor
    except ImportError:
        return False
    else:

        class SyncBatchNorm(nn.SyncBatchNorm):

            def forward(self, input: SparseTensor) -> SparseTensor:
                return fapply(input, super().forward)

        MODELS._register_module(Conv3d, 'TorchSparseConv3d')
        MODELS._register_module(BatchNorm, 'TorchSparseBN')
        MODELS._register_module(SyncBatchNorm, 'TorchSparseSyncBN')
        MODELS._register_module(GroupNorm, 'TorchSparseGN')
        MODELS._register_module(ReLU, 'TorchSparseReLU')
        MODELS._register_module(LeakyReLU, 'TorchSparseLeakyReLU')
        return True
