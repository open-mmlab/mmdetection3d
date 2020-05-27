import torch


def test_voting_module():
    from mmdet3d.ops import VoteModule

    self = VoteModule(vote_per_seed=3, in_channels=8)

    seed_xyz = torch.rand([2, 64, 3], dtype=torch.float32)  # (b, npoints, 3)
    seed_features = torch.rand(
        [2, 8, 64], dtype=torch.float32)  # (b, in_channels, npoints)

    # test forward
    vote_xyz, vote_features = self(seed_xyz, seed_features)
    assert vote_xyz.shape == torch.Size([2, 192, 3])
    assert vote_features.shape == torch.Size([2, 8, 192])
