import torch
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmengine.model import BaseModule
from torch import nn

from mmdet3d.registry import MODELS


class LastLevelMaxPool(nn.Module):
    def __init__(self):
        super(LastLevelMaxPool, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)

    def forward(self, x):
        return self.pool(x)

@MODELS.register_module()
class SQUEEZEFPN(BaseModule):
    """FPN using SqueezeNet architecture.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        init_cfg (dict or :obj:`ConfigDict` or list[dict or :obj:`ConfigDict`],
            optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 out_channels=[256, 256, 256, 256],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 upsample_cfg=dict(type='deconv', bias=False),
                 conv_cfg=dict(type='Conv2d', bias=False),
                 init_cfg=None):
        super(SQUEEZEFPN, self).__init__(init_cfg=init_cfg)
        print("out_channels", len(out_channels), "in_channels", len(in_channels))
        print("out_channels", out_channels, len(out_channels), "in_channels", in_channels, len(in_channels))
        assert len(out_channels) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels


        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channel, out_channels[0], kernel_size=1)
            for in_channel in in_channels
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, padding=1)
            for _ in range(len(in_channels))
        ])
        self.last_level_pool = LastLevelMaxPool()

        # self.deblocks = nn.ModuleList()
        # for i, out_channel in enumerate(out_channels):
        #     upsample_layer = build_upsample_layer(
        #         upsample_cfg,
        #         in_channels=in_channels[i],
        #         out_channels=out_channel,
        #         kernel_size=2,
        #         stride=2)
        #     deblock = nn.Sequential(
        #         upsample_layer,
        #         build_norm_layer(norm_cfg, out_channel)[1],
        #         nn.ReLU(inplace=True)
        #     )
        #    self.deblocks.append(deblock)

    def forward(self, x):
        """Forward function.

        Args:
            x (List[torch.Tensor]): Multi-level features with 4D Tensor in
                (N, C, H, W) shape.

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """
        # print("x", len(x), "in_channels", len(self.in_channels))
        assert len(x) == len(self.in_channels)

        lateral_features = [lateral_conv(feat) for lateral_conv, feat in zip(self.lateral_convs, x)]

        for i in range(len(lateral_features) - 2, -1, -1):
            # print(i)
            # print(i,lateral_features[i].shape)
            # print(i+1,lateral_features[i+1].shape)
            shape_of_tensor = lateral_features[i].size()

            # Extract specific dimensions for upsampleing
            batch_size = shape_of_tensor[0] # not using
            y_dimension = shape_of_tensor[1]
            x_height = shape_of_tensor[2]
            x_width = shape_of_tensor[3]
            lateral_features[i] += nn.functional.interpolate(lateral_features[i + 1],  size=(x_height, x_width), mode='nearest')
            #print(i,lateral_features[i].shape)


        # Apply the FPN convolutions
        fpn_features = [fpn_conv(feat) for fpn_conv, feat in zip(self.fpn_convs, lateral_features)]
        # for i, feature in enumerate(fpn_features):
        #     print(f"FPN Feature {i} shape: {feature.shape}")
        pool = self.last_level_pool(lateral_features[0])
        # fpn_features.append(pool)
        # for i, feature in enumerate(fpn_features):
        #     print(f"FPN Feature {i} shape: {feature.shape}")
        # print(pool.shape)
        return tuple(lateral_features)
