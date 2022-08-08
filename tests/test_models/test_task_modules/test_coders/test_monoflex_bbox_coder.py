# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmdet3d.registry import TASK_UTILS
from mmdet3d.structures import CameraInstance3DBoxes


def test_monoflex_bbox_coder():
    bbox_coder_cfg = dict(
        type='MonoFlexCoder',
        depth_mode='exp',
        base_depth=(26.494627, 16.05988),
        depth_range=[0.1, 100],
        combine_depth=True,
        uncertainty_range=[-10, 10],
        base_dims=((3.8840, 1.5261, 1.6286, 0.4259, 0.1367,
                    0.1022), (0.8423, 1.7607, 0.6602, 0.2349, 0.1133, 0.1427),
                   (1.7635, 1.7372, 0.5968, 0.1766, 0.0948, 0.1242)),
        dims_mode='linear',
        multibin=True,
        num_dir_bins=4,
        bin_centers=[0, np.pi / 2, np.pi, -np.pi / 2],
        bin_margin=np.pi / 6,
        code_size=7)
    bbox_coder = TASK_UTILS.build(bbox_coder_cfg)
    gt_bboxes_3d = CameraInstance3DBoxes(torch.rand([6, 7]))
    orientation_target = bbox_coder.encode(gt_bboxes_3d)
    assert orientation_target.shape == torch.Size([6, 8])

    regression = torch.rand([100, 50])
    base_centers2d = torch.rand([100, 2])
    labels = torch.ones([100])
    downsample_ratio = 4
    cam2imgs = torch.rand([100, 4, 4])

    preds = bbox_coder.decode(regression, base_centers2d, labels,
                              downsample_ratio, cam2imgs)

    assert preds['bboxes2d'].shape == torch.Size([100, 4])
    assert preds['dimensions'].shape == torch.Size([100, 3])
    assert preds['offsets2d'].shape == torch.Size([100, 2])
    assert preds['keypoints2d'].shape == torch.Size([100, 10, 2])
    assert preds['orientations'].shape == torch.Size([100, 16])
    assert preds['direct_depth'].shape == torch.Size([
        100,
    ])
    assert preds['keypoints_depth'].shape == torch.Size([100, 3])
    assert preds['combined_depth'].shape == torch.Size([
        100,
    ])
    assert preds['direct_depth_uncertainty'].shape == torch.Size([
        100,
    ])
    assert preds['keypoints_depth_uncertainty'].shape == torch.Size([100, 3])

    offsets_2d = torch.randn([100, 2])
    depths = torch.randn([
        100,
    ])
    locations = bbox_coder.decode_location(base_centers2d, offsets_2d, depths,
                                           cam2imgs, downsample_ratio)
    assert locations.shape == torch.Size([100, 3])

    orientations = torch.randn([100, 16])
    yaws, local_yaws = bbox_coder.decode_orientation(orientations, locations)
    assert yaws.shape == torch.Size([
        100,
    ])
    assert local_yaws.shape == torch.Size([
        100,
    ])
