import torch
from mmcv import ConfigDict


def test_groupfree3d_transformer_decoder():
    from mmdet3d.models.model_utils import GroupFree3DTransformerDecoder

    bbox_coder = dict(
        type='GroupFree3DBBoxCoder',
        num_sizes=18,
        num_dir_bins=1,
        with_rot=False,
        mean_sizes=[[0.76966727, 0.8116021, 0.92573744],
                    [1.876858, 1.8425595, 1.1931566],
                    [0.61328, 0.6148609, 0.7182701],
                    [1.3955007, 1.5121545, 0.83443564],
                    [0.97949594, 1.0675149, 0.6329687],
                    [0.531663, 0.5955577, 1.7500148],
                    [0.9624706, 0.72462326, 1.1481868],
                    [0.83221924, 1.0490936, 1.6875663],
                    [0.21132214, 0.4206159, 0.5372846],
                    [1.4440073, 1.8970833, 0.26985747],
                    [1.0294262, 1.4040797, 0.87554324],
                    [1.3766412, 0.65521795, 1.6813129],
                    [0.6650819, 0.71111923, 1.298853],
                    [0.41999173, 0.37906948, 1.7513971],
                    [0.59359556, 0.5912492, 0.73919016],
                    [0.50867593, 0.50656086, 0.30136237],
                    [1.1511526, 1.0546296, 0.49706793],
                    [0.47535285, 0.49249494, 0.5802117]])

    pred_layer_cfg = dict(
        in_channels=288, shared_conv_channels=(288, 288), bias=True)

    transformerlayers = dict(
        type='BaseTransformerLayer',
        attn_cfgs=[
            dict(
                type='GroupFree3DMultiheadAttention',
                embed_dims=288,
                num_heads=8,
                dropout=0.1),
            dict(
                type='GroupFree3DMultiheadAttention',
                embed_dims=288,
                num_heads=8,
                dropout=0.1)
        ],
        feedforward_channels=2048,
        ffn_dropout=0.1,
        operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn',
                         'norm'))
    transformerlayers = ConfigDict(transformerlayers)

    self = GroupFree3DTransformerDecoder(
        num_layers=6,
        bbox_coder=bbox_coder,
        transformerlayers=transformerlayers,
        pred_layer_cfg=pred_layer_cfg)

    seed_features = torch.rand([2, 288, 256],
                               dtype=torch.float32)  # (b, in_channels, nseed)
    seed_xyz = torch.rand([2, 256, 3], dtype=torch.float32)  # (b, nseed, 3)
    candidate_features = torch.rand(
        [2, 288, 128], dtype=torch.float32)  # (b, in_channels, ncandidate)
    candidate_xyz = torch.rand([2, 128, 3],
                               dtype=torch.float32)  # (b, ncandidate, 3)
    base_bbox3d = torch.rand([2, 128, 6],
                             dtype=torch.float32)  # (b, ncandidate, 6)

    # test forward
    transformer_res = self(
        candidate_features=candidate_features,
        candidate_xyz=candidate_xyz,
        seed_features=seed_features,
        seed_xyz=seed_xyz,
        base_bbox3d=base_bbox3d)

    assert transformer_res['center_5'].shape == torch.Size([2, 128, 3])
    assert transformer_res['dir_class_5'].shape == torch.Size([2, 128, 1])
    assert transformer_res['dir_res_5'].shape == torch.Size([2, 128, 1])
    assert transformer_res['size_class_5'].shape == torch.Size([2, 128, 18])
    assert transformer_res['size_res_5'].shape == torch.Size([2, 128, 18, 3])
    assert transformer_res['obj_scores_5'].shape == torch.Size([2, 128, 1])
    assert transformer_res['sem_scores_5'].shape == torch.Size([2, 128, 18])


if __name__ == '__main__':
    test_groupfree3d_transformer_decoder()
