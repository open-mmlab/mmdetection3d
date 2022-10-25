# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models import BACKBONES
from mmdet.models.backbones.resnet import ResNet


@BACKBONES.register_module()
class CustomResNet(ResNet):
    """Custom ResNet by removing the stem or input convs."""

    def __init__(self, **kwargs):
        super(CustomResNet, self).__init__(**kwargs)
        if self.deep_stem:
            del self.stem
        else:
            del self.conv1
            del self._modules[self.norm1_name]
            del self.relu
        del self.maxpool

    def forward(self, x):
        """Forward function."""
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
