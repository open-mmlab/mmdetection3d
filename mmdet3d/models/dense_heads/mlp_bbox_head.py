from mmcv.cnn import ConvModule
from torch import nn as nn

from mmdet.models.builder import HEADS


@HEADS.register_module()
class MLPBBoxHead(nn.Module):
    r"""More general bbox head, with shared mlp layers and two optional
    separated branches.

    .. code-block:: none

                   /-> cls mlps -> cls_score
        shared mlps
                   \-> reg mlps -> bbox_pred
    """

    def __init__(self,
                 in_channels=0,
                 shared_mlp_channels=(),
                 cls_mlp_channels=(),
                 num_cls_out_channels=0,
                 reg_mlp_channels=(),
                 num_reg_out_channels=0,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU'),
                 bias='auto',
                 *args,
                 **kwargs):
        super(MLPBBoxHead, self).__init__(*args, **kwargs)
        assert in_channels > 0
        assert num_cls_out_channels > 0
        assert num_reg_out_channels > 0
        self.in_channels = in_channels
        self.shared_mlp_channels = shared_mlp_channels
        self.cls_mlp_channels = cls_mlp_channels
        self.num_cls_out_channels = num_cls_out_channels
        self.reg_mlp_channels = reg_mlp_channels
        self.num_reg_out_channels = num_reg_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.bias = bias

        # add shared convs and fcs
        if len(self.shared_mlp_channels) > 0:
            self.shared_mlps = self._add_mlp_branch(self.in_channels,
                                                    self.shared_mlp_channels)
            out_channels = self.shared_mlp_channels[-1]
        else:
            out_channels = self.in_channels

        # add cls specific branch
        prev_channel = out_channels
        if len(self.cls_mlp_channels) > 0:
            self.cls_mlps = self._add_mlp_branch(prev_channel,
                                                 self.cls_mlp_channels)
            prev_channel = self.cls_mlp_channels[-1]

        self.cls_pred = nn.Conv1d(prev_channel, num_cls_out_channels, 1)

        # add reg specific branch
        prev_channel = out_channels
        if len(self.reg_mlp_channels) > 0:
            self.reg_mlps = self._add_mlp_branch(prev_channel,
                                                 self.reg_mlp_channels)
            prev_channel = self.reg_mlp_channels[-1]

        self.reg_pred = nn.Conv1d(prev_channel, num_reg_out_channels, 1)

    def _add_mlp_branch(self, in_channels, mlp_channels):
        """Add shared or separable branch."""
        mlp_spec = [in_channels] + list(mlp_channels)
        # add branch specific mlp layers
        mlp = nn.Sequential()
        for i in range(len(mlp_spec) - 1):
            mlp.add_module(
                f'layer{i}',
                ConvModule(
                    mlp_spec[i],
                    mlp_spec[i + 1],
                    kernel_size=1,
                    padding=0,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    bias=self.bias,
                    inplace=True))
        return mlp

    def init_weights(self):
        # conv layers are already initialized by ConvModule
        pass

    def forward(self, feats):
        """Forward.

        Args:
            feats (Tensor): Input features

        Returns:
            Tensor: Class scores predictions
            Tensor: Regression predictions
        """
        # shared part
        if len(self.shared_mlp_channels) > 0:
            x = self.shared_mlps(feats)

        # separate branches
        x_cls = x
        x_reg = x

        if len(self.cls_mlp_channels) > 0:
            x_cls = self.cls_mlps(x_cls)
        cls_score = self.cls_pred(x_cls)

        if len(self.reg_mlp_channels) > 0:
            x_reg = self.reg_mlps(x_reg)
        bbox_pred = self.reg_pred(x_reg)

        return cls_score, bbox_pred
