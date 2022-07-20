# Copyright (c) OpenMMLab. All rights reserved.
"""Tests the core function of point fusion.

CommandLine:
    pytest tests/test_models/test_fusion/test_point_fusion.py
"""

import torch

from mmdet3d.models.layers.fusion_layers import PointFusion


def test_sample_single():
    # this function makes sure the rewriting of 3d coords transformation
    # in point fusion does not change the original behaviour
    lidar2img = torch.tensor(
        [[6.0294e+02, -7.0791e+02, -1.2275e+01, -1.7094e+02],
         [1.7678e+02, 8.8088e+00, -7.0794e+02, -1.0257e+02],
         [9.9998e-01, -1.5283e-03, -5.2907e-03, -3.2757e-01],
         [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]])

    #  all use default
    img_meta = {
        'transformation_3d_flow': ['R', 'S', 'T', 'HF'],
        'input_shape': [370, 1224],
        'img_shape': [370, 1224],
        'lidar2img': lidar2img,
    }

    #  dummy parameters
    fuse = PointFusion(1, 1, 1, 1)
    img_feat = torch.arange(370 * 1224)[None, ...].view(
        370, 1224)[None, None, ...].float() / (370 * 1224)
    pts = torch.tensor([[8.356, -4.312, -0.445], [11.777, -6.724, -0.564],
                        [6.453, 2.53, -1.612], [6.227, -3.839, -0.563]])
    out = fuse.sample_single(img_feat, pts, img_meta)

    expected_tensor = torch.tensor(
        [0.5560822, 0.5476625, 0.9687978, 0.6241757])
    assert torch.allclose(expected_tensor, out, 1e-4)

    pcd_rotation = torch.tensor([[8.660254e-01, 0.5, 0],
                                 [-0.5, 8.660254e-01, 0], [0, 0, 1.0e+00]])
    pcd_scale_factor = 1.111
    pcd_trans = torch.tensor([1.0, -1.0, 0.5])
    pts = pts @ pcd_rotation
    pts *= pcd_scale_factor
    pts += pcd_trans
    pts[:, 1] = -pts[:, 1]

    # not use default
    img_meta.update({
        'pcd_scale_factor': pcd_scale_factor,
        'pcd_rotation': pcd_rotation,
        'pcd_trans': pcd_trans,
        'pcd_horizontal_flip': True
    })
    out = fuse.sample_single(img_feat, pts, img_meta)
    expected_tensor = torch.tensor(
        [0.5560822, 0.5476625, 0.9687978, 0.6241757])
    assert torch.allclose(expected_tensor, out, 1e-4)
