import torch
from torch.autograd import Function, Variable

from . import pointnet2_cuda as pointnet2


class ThreeNN(Function):

    @staticmethod
    def forward(ctx, unknown, known):
        """
        Find the three nearest neighbors of unknown in known
        :param ctx:
        :param unknown: (N, 3)
        :param known: (M, 3)
        :return:
            dist: (N, 3) l2 distance to the three nearest neighbors
            idx: (N, 3) index of 3 nearest neighbors
        """
        assert unknown.is_contiguous()
        assert known.is_contiguous()

        N, _ = unknown.size()
        m = known.size(0)
        dist2 = torch.cuda.FloatTensor(N, 3)
        idx = torch.cuda.IntTensor(N, 3)

        pointnet2.three_nn_wrapper(N, m, unknown, known, dist2, idx)
        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn_2d = ThreeNN.apply


class ThreeInterpolate(Function):

    @staticmethod
    def forward(ctx, features, idx, weight):
        """
        Performs weight linear interpolation on 3 features
        :param ctx:
        :param features: (M, C) Features descriptors to be interpolated from
        :param idx: (n, 3) three nearest neighbors of the target features in
                    features
        :param weight: (n, 3) weights
        :return:
            output: (N, C) tensor of the interpolated features
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        assert weight.is_contiguous()

        m, c = features.size()
        n = idx.size(0)
        ctx.three_interpolate_for_backward = (idx, weight, m)
        output = torch.cuda.FloatTensor(n, c)

        pointnet2.three_interpolate_wrapper(
                c, m, n, features, idx, weight, output)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        """
        :param ctx:
        :param grad_out: (N, C) tensor with gradients of outputs
        :return:
            grad_features: (M, C) tensor with gradients of features
            None:
            None:
        """
        idx, weight, m = ctx.three_interpolate_for_backward
        n, c = grad_out.size()

        grad_features = Variable(torch.cuda.FloatTensor(m, c).zero_())
        grad_out_data = grad_out.data.contiguous()

        pointnet2.three_interpolate_grad_wrapper(
                c, n, m, grad_out_data, idx, weight, grad_features.data)
        return grad_features, None, None


three_interpolate_2d = ThreeInterpolate.apply
