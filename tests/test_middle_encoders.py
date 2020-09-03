import pytest
import torch

from mmdet3d.models.builder import build_middle_encoder


def test_sparse_encoder():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    sparse_encoder_cfg = dict(
        type='SparseEncoder',
        in_channels=5,
        sparse_shape=[40, 1024, 1024],
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1,
                                                                       1)),
        block_type='basicblock')

    sparse_encoder = build_middle_encoder(sparse_encoder_cfg).cuda()
    voxel_features = torch.rand([207842, 5]).cuda()
    coors = torch.randint(0, 4, [207842, 4]).cuda()

    ret = sparse_encoder(voxel_features, coors, 4)
    assert ret.shape == torch.Size([4, 256, 128, 128])
