import pytest


def test_secfpn():
    neck_cfg = dict(
        type='SECONDFPN',
        in_channels=[2, 3],
        upsample_strides=[1, 2],
        out_channels=[4, 6],
    )
    from mmdet.models.builder import build_neck
    neck = build_neck(neck_cfg)
    assert neck.deblocks[0][0].in_channels == 2
    assert neck.deblocks[1][0].in_channels == 3
    assert neck.deblocks[0][0].out_channels == 4
    assert neck.deblocks[1][0].out_channels == 6
    assert neck.deblocks[0][0].stride == (1, 1)
    assert neck.deblocks[1][0].stride == (2, 2)
    assert neck is not None

    neck_cfg = dict(
        type='SECONDFPN',
        in_channels=[2, 2],
        upsample_strides=[1, 2, 4],
        out_channels=[2, 2],
    )
    with pytest.raises(AssertionError):
        build_neck(neck_cfg)

    neck_cfg = dict(
        type='SECONDFPN',
        in_channels=[2, 2, 4],
        upsample_strides=[1, 2, 4],
        out_channels=[2, 2],
    )
    with pytest.raises(AssertionError):
        build_neck(neck_cfg)
