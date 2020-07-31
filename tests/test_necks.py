import torch

from mmdet3d.models.builder import build_neck


def test_centerpoint_rpn():
    centerpoint_rpn_cfg = dict(
        type='CenterPointRPN',
        layer_nums=[3, 5, 5],
        downsample_strides=[2, 2, 2],
        downsample_channels=[64, 128, 256],
        upsample_strides=[0.5, 1, 2],
        upsample_channels=[128, 128, 128],
        input_channels=64,
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01))
    centerpoint_rpn = build_neck(centerpoint_rpn_cfg)
    centerpoint_rpn.init_weights()
    input = torch.rand([4, 64, 512, 512])
    output = centerpoint_rpn(input)
    assert output.shape == torch.Size([4, 384, 128, 128])
