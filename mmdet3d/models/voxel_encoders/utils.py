import torch
from torch import nn
from torch.nn import functional as F

from mmdet.ops import build_norm_layer


class Empty(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Empty, self).__init__()

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            return args[0]
        elif len(args) == 0:
            return None
        return args


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num ([type]): [description]
        max_num ([type]): [description]

    Returns:
        [type]: [description]
    """
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


class VFELayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 max_out=True,
                 cat_max=True):
        super(VFELayer, self).__init__()
        self.cat_max = cat_max
        self.max_out = max_out
        # self.units = int(out_channels / 2)
        if norm_cfg:
            norm_name, norm_layer = build_norm_layer(norm_cfg, out_channels)
            self.norm = norm_layer
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
        else:
            self.norm = Empty(out_channels)
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, inputs):
        # [K, T, 7] tensordot [7, units] = [K, T, units]
        voxel_count = inputs.shape[1]
        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()
        pointwise = F.relu(x)
        # [K, T, units]
        if self.max_out:
            aggregated = torch.max(pointwise, dim=1, keepdim=True)[0]
        else:
            # this is for fusion layer
            return pointwise

        if not self.cat_max:
            return aggregated.squeeze(1)
        else:
            # [K, 1, units]
            repeated = aggregated.repeat(1, voxel_count, 1)
            concatenated = torch.cat([pointwise, repeated], dim=2)
            # [K, T, 2 * units]
            return concatenated


class PFNLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False,
                 mode='max'):
        """ Pillar Feature Net Layer.

        The Pillar Feature Net is composed of a series of these layers, but the
        PointPillars paper results only used a single PFNLayer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            use_norm (bool): Whether to include BatchNorm.
            last_layer (bool): If last_layer, there is no concatenation of
                features.
        """

        super().__init__()
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        if use_norm:
            self.norm = nn.BatchNorm1d(self.units, eps=1e-3, momentum=0.01)
            self.linear = nn.Linear(in_channels, self.units, bias=False)
        else:
            self.norm = Empty(self.unints)
            self.linear = nn.Linear(in_channels, self.units, bias=True)

        self.mode = mode

    def forward(self, inputs, num_voxels=None, aligned_distance=None):

        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()
        x = F.relu(x)

        if self.mode == 'max':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = torch.max(x, dim=1, keepdim=True)[0]
        elif self.mode == 'avg':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = x.sum(
                dim=1, keepdim=True) / num_voxels.type_as(inputs).view(
                    -1, 1, 1)

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated
