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
    vote_xyz, vote_features, _ = self(seed_xyz, seed_features)
    assert vote_xyz.shape == torch.Size([2, 192, 3])
    assert vote_features.shape == torch.Size([2, 8, 192])
