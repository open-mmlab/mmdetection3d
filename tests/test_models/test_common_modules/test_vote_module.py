import torch


def test_vote_module():
    from mmdet3d.models.model_utils import VoteModule

    vote_loss = dict(
        type='ChamferDistance',
        mode='l1',
        reduction='none',
        loss_dst_weight=10.0)
    self = VoteModule(vote_per_seed=3, in_channels=8, vote_loss=vote_loss)

    seed_xyz = torch.rand([2, 64, 3], dtype=torch.float32)  # (b, npoints, 3)
    seed_features = torch.rand(
        [2, 8, 64], dtype=torch.float32)  # (b, in_channels, npoints)

    # test forward
    vote_xyz, vote_features, vote_offset = self(seed_xyz, seed_features)
    assert vote_xyz.shape == torch.Size([2, 192, 3])
    assert vote_features.shape == torch.Size([2, 8, 192])
    assert vote_offset.shape == torch.Size([2, 3, 192])

    # test clip offset and without feature residual
    self = VoteModule(
        vote_per_seed=1,
        in_channels=8,
        num_points=32,
        with_res_feat=False,
        vote_xyz_range=(2.0, 2.0, 2.0))

    vote_xyz, vote_features, vote_offset = self(seed_xyz, seed_features)
    assert vote_xyz.shape == torch.Size([2, 32, 3])
    assert vote_features.shape == torch.Size([2, 8, 32])
    assert vote_offset.shape == torch.Size([2, 3, 32])
    assert torch.allclose(seed_features[..., :32], vote_features)
    assert vote_offset.max() <= 2.0
    assert vote_offset.min() >= -2.0
