from mmcv.runner import BaseModule
from torch import nn as nn

from mmdet.models.builder import HEADS


@HEADS.register_module()
class BaseLinearBboxHead(BaseModule):
    r"""More general bbox head, with linear layers and two optional
    separated branches.

    .. code-block:: none

           /-> cls fc layers -> cls_score
    featurs
           \-> reg fc layers -> bbox_pred
    """

    def __init__(self,
                 in_channels=0,
                 cls_linear_channels=(),
                 num_cls_out_channels=0,
                 reg_linear_channels=(),
                 num_reg_out_channels=0,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        assert in_channels > 0
        assert num_cls_out_channels > 0
        assert num_reg_out_channels > 0
        self.in_channels = in_channels
        self.cls_linear_channels = cls_linear_channels
        self.num_cls_out_channels = num_cls_out_channels
        self.reg_linear_channels = reg_linear_channels
        self.num_reg_out_channels = num_reg_out_channels

        # add cls specific branch
        self.cls_layers = self._make_fc_layers(
            fc_cfg=cls_linear_channels,
            input_channels=in_channels,
            output_channels=num_cls_out_channels)
        # add reg specific branch
        self.reg_layers = self._make_fc_layers(
            fc_cfg=reg_linear_channels,
            input_channels=in_channels,
            output_channels=num_reg_out_channels)

    def _make_fc_layers(self, fc_cfg, input_channels, output_channels):
        fc_layers = []
        c_in = input_channels
        for k in range(0, fc_cfg.__len__()):
            fc_layers.extend([
                nn.Linear(c_in, fc_cfg[k], bias=False),
                nn.BatchNorm1d(fc_cfg[k]),
                nn.ReLU(),
            ])
            c_in = fc_cfg[k]
        fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
        return nn.Sequential(*fc_layers)

    def forward(self, feats):
        bs = feats.shape[0]
        feats = feats.permute(0, 2, 1).contiguous()
        x_cls = feats.view(-1, feats.shape[-1])
        x_reg = feats.view(-1, feats.shape[-1])

        cls_score = self.cls_layers(x_cls).reshape(
            bs, -1, self.num_cls_out_channels).transpose(2, 1)
        bbox_pred = self.reg_layers(x_reg).reshape(
            bs, -1, self.num_reg_out_channels).transpose(2, 1)

        return cls_score, bbox_pred
