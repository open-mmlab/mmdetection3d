# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmdet3d.core.bbox import LiDARInstance3DBoxes


def test_PointwiseSemanticHead():
    # PointwiseSemanticHead only support gpu version currently.
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    from mmdet3d.models.builder import build_head

    head_cfg = dict(
        type='PointwiseSemanticHead',
        in_channels=8,
        extra_width=0.2,
        seg_score_thr=0.3,
        num_classes=3,
        loss_seg=dict(
            type='FocalLoss',
            use_sigmoid=True,
            reduction='sum',
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_part=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0))

    self = build_head(head_cfg)
    self.cuda()

    # test forward
    voxel_features = torch.rand([4, 8], dtype=torch.float32).cuda()
    feats_dict = self.forward(voxel_features)
    assert feats_dict['seg_preds'].shape == torch.Size(
        [voxel_features.shape[0], 1])
    assert feats_dict['part_preds'].shape == torch.Size(
        [voxel_features.shape[0], 3])
    assert feats_dict['part_feats'].shape == torch.Size(
        [voxel_features.shape[0], 4])

    voxel_centers = torch.tensor(
        [[6.56126, 0.9648336, -1.7339306], [6.8162713, -2.480431, -1.3616394],
         [11.643568, -4.744306, -1.3580885], [23.482342, 6.5036807, 0.5806964]
         ],
        dtype=torch.float32).cuda()  # n, point_features
    coordinates = torch.tensor(
        [[0, 12, 819, 131], [0, 16, 750, 136], [1, 16, 705, 232],
         [1, 35, 930, 469]],
        dtype=torch.int32).cuda()  # n, 4(batch, ind_x, ind_y, ind_z)
    voxel_dict = dict(voxel_centers=voxel_centers, coors=coordinates)
    gt_bboxes = [
        LiDARInstance3DBoxes(
            torch.tensor(
                [[6.4118, -3.4305, -1.7291, 1.7033, 3.4693, 1.6197, -0.9091]],
                dtype=torch.float32).cuda()),
        LiDARInstance3DBoxes(
            torch.tensor(
                [[16.9107, 9.7925, -1.9201, 1.6097, 3.2786, 1.5307, -2.4056]],
                dtype=torch.float32).cuda())
    ]
    # batch size is 2 in the unit test
    gt_labels = list(torch.tensor([[0], [1]], dtype=torch.int64).cuda())

    # test get_targets
    target_dict = self.get_targets(voxel_dict, gt_bboxes, gt_labels)

    assert target_dict['seg_targets'].shape == torch.Size(
        [voxel_features.shape[0]])
    assert torch.allclose(target_dict['seg_targets'],
                          target_dict['seg_targets'].new_tensor([3, -1, 3, 3]))
    assert target_dict['part_targets'].shape == torch.Size(
        [voxel_features.shape[0], 3])
    assert target_dict['part_targets'].sum() == 0

    # test loss
    loss_dict = self.loss(feats_dict, target_dict)
    assert loss_dict['loss_seg'] > 0
    assert loss_dict['loss_part'] == 0  # no points in gt_boxes
    total_loss = loss_dict['loss_seg'] + loss_dict['loss_part']
    total_loss.backward()
