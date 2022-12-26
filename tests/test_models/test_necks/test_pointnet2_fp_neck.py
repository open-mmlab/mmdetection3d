import pytest
import torch

from mmdet3d.registry import MODELS


def test_pointnet2_fp_neck():
    if not torch.cuda.is_available():
        pytest.skip()

    xyzs = [16384, 4096, 1024, 256, 64]
    feat_channels = [1, 96, 256, 512, 1024]
    channel_num = 5

    sa_xyz = [torch.rand(3, xyzs[i], 3) for i in range(channel_num)]
    sa_features = [
        torch.rand(3, feat_channels[i], xyzs[i]) for i in range(channel_num)
    ]

    neck_cfg = dict(
        type='PointNetFPNeck',
        fp_channels=((1536, 512, 512), (768, 512, 512), (608, 256, 256),
                     (257, 128, 128)))

    neck = MODELS.build(neck_cfg)
    neck.init_weights()

    if torch.cuda.is_available():
        sa_xyz = [x.cuda() for x in sa_xyz]
        sa_features = [x.cuda() for x in sa_features]
        neck.cuda()

    feats_sa = {'sa_xyz': sa_xyz, 'sa_features': sa_features}
    outputs = neck(feats_sa)
    assert outputs['fp_xyz'].cpu().numpy().shape == (3, 16384, 3)
    assert outputs['fp_features'].detach().cpu().numpy().shape == (3, 128,
                                                                   16384)
