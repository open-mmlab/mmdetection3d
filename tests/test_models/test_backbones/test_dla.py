import torch

from mmdet3d.registry import MODELS


def test_dla_net():
    # test DLANet used in SMOKE
    # test list config
    cfg = dict(
        type='DLANet',
        depth=34,
        in_channels=3,
        norm_cfg=dict(type='GN', num_groups=32))

    img = torch.randn((4, 3, 32, 32))
    self = MODELS.build(cfg)
    self.init_weights()

    results = self(img)
    assert len(results) == 6
    assert results[0].shape == torch.Size([4, 16, 32, 32])
    assert results[1].shape == torch.Size([4, 32, 16, 16])
    assert results[2].shape == torch.Size([4, 64, 8, 8])
    assert results[3].shape == torch.Size([4, 128, 4, 4])
    assert results[4].shape == torch.Size([4, 256, 2, 2])
    assert results[5].shape == torch.Size([4, 512, 1, 1])
