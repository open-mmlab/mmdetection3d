from mmcv.cnn.bricks import build_conv_layer

from mmdet.models.builder import HEADS
from .base_conv_bbox_head import BaseConvBboxHead


@HEADS.register_module()
class BaseSeparateConvBboxHead(BaseConvBboxHead):
    r"""More general bbox head, with shared conv layers and two optional
    separated branches.

    .. code-block:: none

           /-> cls convs -> cls_score
    featurs
           \-> reg convs -> bbox_pred
    """

    def __init__(self,
                 in_channels=0,
                 cls_conv_channels=(),
                 num_cls_out_channels=0,
                 reg_conv_channels=(),
                 num_reg_out_channels=0,
                 init_cfg=None,
                 bias=False,
                 *args,
                 **kwargs):
        super().__init__(
            init_cfg=init_cfg,
            in_channels=in_channels,
            cls_conv_channels=cls_conv_channels,
            num_cls_out_channels=num_cls_out_channels,
            reg_conv_channels=reg_conv_channels,
            num_reg_out_channels=num_reg_out_channels,
            *args,
            **kwargs)
        # add cls specific branch
        self.cls_convs = self._add_conv_branch(self.in_channels,
                                               self.cls_conv_channels)
        prev_channel = self.cls_conv_channels[-1]
        self.conv_cls = build_conv_layer(
            self.conv_cfg,
            in_channels=prev_channel,
            out_channels=num_cls_out_channels,
            kernel_size=1)

        # add reg specific branch
        self.reg_convs = self._add_conv_branch(self.in_channels,
                                               self.reg_conv_channels)
        prev_channel = self.reg_conv_channels[-1]
        self.conv_reg = build_conv_layer(
            self.conv_cfg,
            in_channels=prev_channel,
            out_channels=num_reg_out_channels,
            kernel_size=1)

    def forward(self, feats):
        """Forward.

        Args:
            feats (Tensor): Input features

        Returns:
            Tensor: Class scores predictions
            Tensor: Regression predictions
        """

        # separate branches
        x_cls = feats
        x_reg = feats

        x_cls = self.cls_convs(x_cls)
        cls_score = self.conv_cls(x_cls)

        x_reg = self.reg_convs(x_reg)
        bbox_pred = self.conv_reg(x_reg)

        return cls_score, bbox_pred
