import torch
from torch.autograd import Function

try:
    from . import ingroup_inds_cuda

    # import ingroup_indices
except ImportError:
    ingroup_indices = None
    print('Can not import ingroup indices')

ingroup_indices = ingroup_inds_cuda


class IngroupIndicesFunction(Function):

    @staticmethod
    def forward(ctx, group_inds):

        out_inds = torch.zeros_like(group_inds) - 1

        ingroup_indices.forward(group_inds, out_inds)

        ctx.mark_non_differentiable(out_inds)

        return out_inds

    @staticmethod
    def backward(ctx, g):

        return None


ingroup_inds = IngroupIndicesFunction.apply
