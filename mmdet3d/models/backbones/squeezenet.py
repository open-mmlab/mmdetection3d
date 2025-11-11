from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from mmcv.cnn import build_conv_layer, build_norm_layer
import torch
import torch.nn as nn
from typing import Sequence, Optional, Tuple
from torch import Tensor
#from torch import nn
@MODELS.register_module()
class SQUEEZE(BaseModule):
    """Backbone network using SqueezeNet architecture.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        conv_cfg (dict): Config dict of convolutional layers.
    """

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: Sequence[int] = [64, 128, 256],
                 norm_cfg: dict = dict(type='BN', eps=1e-3, momentum=0.01),
                 conv_cfg: dict = dict(type='Conv2d', bias=False),
                 init_cfg: Optional[dict] = None,
                 pretrained: Optional[str] = None) -> None:
        super(SQUEEZE, self).__init__(init_cfg=init_cfg)
        self.conv_cfg = conv_cfg;
        self.norm_cfg = norm_cfg;


        # Define the SqueezeNet fire modules
        self.features = nn.Sequential(
            #build_conv_layer(conv_cfg, in_channels, 96, kernel_size=7, stride=2),
            build_conv_layer(conv_cfg, in_channels, 64, kernel_size=3, stride=2),
            #build_norm_layer(norm_cfg, 96)[1],
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            #self._make_fire_module(96, 16, 64, 64),
            self._make_fire_module(64, 16, 64, 64),
            self._make_fire_module(128, 16, 64, 64),
            self._make_fire_module(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            self._make_fire_module(256, 32, 128, 128),
            self._make_fire_module(256, 48, 192, 192),
            self._make_fire_module(384, 48, 192, 192),
            self._make_fire_module(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            self._make_fire_module(512, 64, 256, 256),
        )

        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        else:
            self.init_cfg = dict(type='Kaiming', layer='Conv2d')



    def _make_fire_module(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
      layers = nn.Sequential()
      
      # Squeeze layer
      squeeze = nn.Sequential(
          build_conv_layer(self.conv_cfg, in_channels, squeeze_channels, kernel_size=1),
          build_norm_layer(self.norm_cfg, squeeze_channels)[1],
          nn.ReLU(inplace=True)
      )
      layers.add_module('squeeze', squeeze)
      
      # Expand 1x1 layer
      expand1x1 = nn.Sequential(
          build_conv_layer(self.conv_cfg, squeeze_channels, expand1x1_channels, kernel_size=1),
          build_norm_layer(self.norm_cfg, expand1x1_channels)[1],
          nn.ReLU(inplace=True)
      )
      layers.add_module('expand1x1', expand1x1)
      
      # Expand 3x3 layer
      expand3x3 = nn.Sequential(
          build_conv_layer(self.conv_cfg, squeeze_channels, expand3x3_channels, kernel_size=3, padding=1),
          build_norm_layer(self.norm_cfg, expand3x3_channels)[1],
          nn.ReLU(inplace=True)
      )
      layers.add_module('expand3x3', expand3x3)
      
      # Concatenation layer (handled within the forward function)
      return layers

    def forward(self, x):
        """Forward function with correct concatenation for Fire modules."""
        x = self.features[0](x)  # handled here as the initial layers are not fire modules
        targeted_layers = [1,5, 8, 13]
        outs = []
        for idx, layer in enumerate(self.features[1:], 1):
            #print(idx,":",layer)
            if isinstance(layer, nn.Sequential) and 'squeeze' in layer._modules:
                # This is a Fire module, handle separately
                squeeze_output = layer.squeeze(x)
                x1 = layer.expand1x1(squeeze_output)
                x3 = layer.expand3x3(squeeze_output)
                x = torch.cat([x1, x3], 1)
            else:
                # Normal layer
                x = layer(x)
            if(idx in targeted_layers):
              outs.append(x)
              #print("Outs x",idx , x.shape)
        #print(len(outs))
        return outs
