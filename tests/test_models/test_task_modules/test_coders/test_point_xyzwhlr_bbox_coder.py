# Copyright (c) OpenMMLab. All rights reserved.

import torch

from mmdet3d.registry import TASK_UTILS


def test_point_xyzwhlr_bbox_coder():
    bbox_coder_cfg = dict(
        type='PointXYZWHLRBBoxCoder',
        use_mean_size=True,
        mean_size=[[3.9, 1.6, 1.56], [0.8, 0.6, 1.73], [1.76, 0.6, 1.73]])
    boxcoder = TASK_UTILS.build(bbox_coder_cfg)

    # test encode
    gt_bboxes_3d = torch.tensor(
        [[13.3329, 2.3514, -0.7004, 1.7508, 0.4702, 1.7909, -3.0522],
         [2.2068, -2.6994, -0.3277, 3.8703, 1.6602, 1.6913, -1.9057],
         [5.5269, 2.5085, -1.0129, 1.1496, 0.8006, 1.8887, 2.1756]])

    points = torch.tensor([[13.70, 2.40, 0.12], [3.20, -3.00, 0.2],
                           [5.70, 2.20, -0.4]])

    gt_labels_3d = torch.tensor([2, 0, 1])

    bbox_target = boxcoder.encode(gt_bboxes_3d, points, gt_labels_3d)
    expected_bbox_target = torch.tensor([[
        -0.1974, -0.0261, -0.4742, -0.0052, -0.2438, 0.0346, -0.9960, -0.0893
    ], [-0.2356, 0.0713, -0.3383, -0.0076, 0.0369, 0.0808, -0.3287, -0.9444
        ], [-0.1731, 0.3085, -0.3543, 0.3626, 0.2884, 0.0878, -0.5686,
            0.8226]])
    assert torch.allclose(expected_bbox_target, bbox_target, atol=1e-4)
    # test decode
    bbox3d_out = boxcoder.decode(bbox_target, points, gt_labels_3d)
    assert torch.allclose(bbox3d_out, gt_bboxes_3d, atol=1e-4)
