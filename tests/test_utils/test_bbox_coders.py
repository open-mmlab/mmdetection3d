# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.cnn import Scale
from torch import nn as nn

from mmdet3d.core.bbox import (CameraInstance3DBoxes, DepthInstance3DBoxes,
                               LiDARInstance3DBoxes)
from mmdet.core import build_bbox_coder


def test_partial_bin_based_box_coder():
    box_coder_cfg = dict(
        type='PartialBinBasedBBoxCoder',
        num_sizes=10,
        num_dir_bins=12,
        with_rot=True,
        mean_sizes=[[2.114256, 1.620300, 0.927272],
                    [0.791118, 1.279516, 0.718182],
                    [0.923508, 1.867419, 0.845495],
                    [0.591958, 0.552978, 0.827272],
                    [0.699104, 0.454178, 0.75625],
                    [0.69519, 1.346299, 0.736364],
                    [0.528526, 1.002642, 1.172878],
                    [0.500618, 0.632163, 0.683424],
                    [0.404671, 1.071108, 1.688889],
                    [0.76584, 1.398258, 0.472728]])
    box_coder = build_bbox_coder(box_coder_cfg)

    # test eocode
    gt_bboxes = DepthInstance3DBoxes(
        [[0.8308, 4.1168, -1.2035, 2.2493, 1.8444, 1.9245, 1.6486],
         [2.3002, 4.8149, -1.2442, 0.5718, 0.8629, 0.9510, 1.6030],
         [-1.1477, 1.8090, -1.1725, 0.6965, 1.5273, 2.0563, 0.0552]])

    gt_labels = torch.tensor([0, 1, 2])
    center_target, size_class_target, size_res_target, dir_class_target, \
        dir_res_target = box_coder.encode(gt_bboxes, gt_labels)
    expected_center_target = torch.tensor([[0.8308, 4.1168, -0.2413],
                                           [2.3002, 4.8149, -0.7687],
                                           [-1.1477, 1.8090, -0.1444]])
    expected_size_class_target = torch.tensor([0, 1, 2])
    expected_size_res_target = torch.tensor([[0.1350, 0.2241, 0.9972],
                                             [-0.2193, -0.4166, 0.2328],
                                             [-0.2270, -0.3401, 1.2108]])
    expected_dir_class_target = torch.tensor([3, 3, 0])
    expected_dir_res_target = torch.tensor([0.0778, 0.0322, 0.0552])
    assert torch.allclose(center_target, expected_center_target, atol=1e-4)
    assert torch.all(size_class_target == expected_size_class_target)
    assert torch.allclose(size_res_target, expected_size_res_target, atol=1e-4)
    assert torch.all(dir_class_target == expected_dir_class_target)
    assert torch.allclose(dir_res_target, expected_dir_res_target, atol=1e-4)

    # test decode
    center = torch.tensor([[[0.8014, 3.4134,
                             -0.6133], [2.6375, 8.4191, 2.0438],
                            [4.2017, 5.2504,
                             -0.7851], [-1.0088, 5.4107, 1.6293],
                            [1.4837, 4.0268, 0.6222]]])

    size_class = torch.tensor([[[
        -1.0061, -2.2788, 1.1322, -4.4380, -11.0526, -2.8113, -2.0642, -7.5886,
        -4.8627, -5.0437
    ],
                                [
                                    -2.2058, -0.3527, -1.9976, 0.8815, -2.7980,
                                    -1.9053, -0.5097, -2.0232, -1.4242, -4.1192
                                ],
                                [
                                    -1.4783, -0.1009, -1.1537, 0.3052, -4.3147,
                                    -2.6529, 0.2729, -0.3755, -2.6479, -3.7548
                                ],
                                [
                                    -6.1809, -3.5024, -8.3273, 1.1252, -4.3315,
                                    -7.8288, -4.6091, -5.8153, 0.7480, -10.1396
                                ],
                                [
                                    -9.0424, -3.7883, -6.0788, -1.8855,
                                    -10.2493, -9.7164, -1.0658, -4.1713,
                                    1.1173, -10.6204
                                ]]])

    size_res = torch.tensor([[[[-9.8976e-02, -5.2152e-01, -7.6421e-02],
                               [1.4593e-01, 5.6099e-01, 8.9421e-02],
                               [5.1481e-02, 3.9280e-01, 1.2705e-01],
                               [3.6869e-01, 7.0558e-01, 1.4647e-01],
                               [4.7683e-01, 3.3644e-01, 2.3481e-01],
                               [8.7346e-02, 8.4987e-01, 3.3265e-01],
                               [2.1393e-01, 8.5585e-01, 9.8948e-02],
                               [7.8530e-02, 5.9694e-02, -8.7211e-02],
                               [1.8551e-01, 1.1308e+00, -5.1864e-01],
                               [3.6485e-01, 7.3757e-01, 1.5264e-01]],
                              [[-9.5593e-01, -5.0455e-01, 1.9554e-01],
                               [-1.0870e-01, 1.8025e-01, 1.0228e-01],
                               [-8.2882e-02, -4.3771e-01, 9.2135e-02],
                               [-4.0840e-02, -5.9841e-02, 1.1982e-01],
                               [7.3448e-02, 5.2045e-02, 1.7301e-01],
                               [-4.0440e-02, 4.9532e-02, 1.1266e-01],
                               [3.5857e-02, 1.3564e-02, 1.0212e-01],
                               [-1.0407e-01, -5.9321e-02, 9.2622e-02],
                               [7.4691e-03, 9.3080e-02, -4.4077e-01],
                               [-6.0121e-02, -1.3381e-01, -6.8083e-02]],
                              [[-9.3970e-01, -9.7823e-01, -5.1075e-02],
                               [-1.2843e-01, -1.8381e-01, 7.1327e-02],
                               [-1.2247e-01, -8.1115e-01, 3.6495e-02],
                               [4.9154e-02, -4.5440e-02, 8.9520e-02],
                               [1.5653e-01, 3.5990e-02, 1.6414e-01],
                               [-5.9621e-02, 4.9357e-03, 1.4264e-01],
                               [8.5235e-04, -1.0030e-01, -3.0712e-02],
                               [-3.7255e-02, 2.8996e-02, 5.5545e-02],
                               [3.9298e-02, -4.7420e-02, -4.9147e-01],
                               [-1.1548e-01, -1.5895e-01, -3.9155e-02]],
                              [[-1.8725e+00, -7.4102e-01, 1.0524e+00],
                               [-3.3210e-01, 4.7828e-02, -3.2666e-02],
                               [-2.7949e-01, 5.5541e-02, -1.0059e-01],
                               [-8.5533e-02, 1.4870e-01, -1.6709e-01],
                               [3.8283e-01, 2.6609e-01, 2.1361e-01],
                               [-4.2156e-01, 3.2455e-01, 6.7309e-01],
                               [-2.4336e-02, -8.3366e-02, 3.9913e-01],
                               [8.2142e-03, 4.8323e-02, -1.5247e-01],
                               [-4.8142e-02, -3.0074e-01, -1.6829e-01],
                               [1.3274e-01, -2.3825e-01, -1.8127e-01]],
                              [[-1.2576e+00, -6.1550e-01, 7.9430e-01],
                               [-4.7222e-01, 1.5634e+00, -5.9460e-02],
                               [-3.5367e-01, 1.3616e+00, -1.6421e-01],
                               [-1.6611e-02, 2.4231e-01, -9.6188e-02],
                               [5.4486e-01, 4.6833e-01, 5.1151e-01],
                               [-6.1755e-01, 1.0292e+00, 1.2458e+00],
                               [-6.8152e-02, 2.4786e-01, 9.5088e-01],
                               [-4.8745e-02, 1.5134e-01, -9.9962e-02],
                               [2.4485e-03, -7.5991e-02, 1.3545e-01],
                               [4.1608e-01, -1.2093e-01, -3.1643e-01]]]])

    dir_class = torch.tensor([[[
        -1.0230, -5.1965, -5.2195, 2.4030, -2.7661, -7.3399, -1.1640, -4.0630,
        -5.2940, 0.8245, -3.1869, -6.1743
    ],
                               [
                                   -1.9503, -1.6940, -0.8716, -1.1494, -0.8196,
                                   0.2862, -0.2921, -0.7894, -0.2481, -0.9916,
                                   -1.4304, -1.2466
                               ],
                               [
                                   -1.7435, -1.2043, -0.1265, 0.5083, -0.0717,
                                   -0.9560, -1.6171, -2.6463, -2.3863, -2.1358,
                                   -1.8812, -2.3117
                               ],
                               [
                                   -1.9282, 0.3792, -1.8426, -1.4587, -0.8582,
                                   -3.4639, -3.2133, -3.7867, -7.6781, -6.4459,
                                   -6.2455, -5.4797
                               ],
                               [
                                   -3.1869, 0.4456, -0.5824, 0.9994, -1.0554,
                                   -8.4232, -7.7019, -7.1382, -10.2724,
                                   -7.8229, -8.1860, -8.6194
                               ]]])

    dir_res = torch.tensor(
        [[[
            1.1022e-01, -2.3750e-01, 2.0381e-01, 1.2177e-01, -2.8501e-01,
            1.5351e-01, 1.2218e-01, -2.0677e-01, 1.4468e-01, 1.1593e-01,
            -2.6864e-01, 1.1290e-01
        ],
          [
              -1.5788e-02, 4.1538e-02, -2.2857e-04, -1.4011e-02, 4.2560e-02,
              -3.1186e-03, -5.0343e-02, 6.8110e-03, -2.6728e-02, -3.2781e-02,
              3.6889e-02, -1.5609e-03
          ],
          [
              1.9004e-02, 5.7105e-03, 6.0329e-02, 1.3074e-02, -2.5546e-02,
              -1.1456e-02, -3.2484e-02, -3.3487e-02, 1.6609e-03, 1.7095e-02,
              1.2647e-05, 2.4814e-02
          ],
          [
              1.4482e-01, -6.3083e-02, 5.8307e-02, 9.1396e-02, -8.4571e-02,
              4.5890e-02, 5.6243e-02, -1.2448e-01, -9.5244e-02, 4.5746e-02,
              -1.7390e-02, 9.0267e-02
          ],
          [
              1.8065e-01, -2.0078e-02, 8.5401e-02, 1.0784e-01, -1.2495e-01,
              2.2796e-02, 1.1310e-01, -8.4364e-02, -1.1904e-01, 6.1180e-02,
              -1.8109e-02, 1.1229e-01
          ]]])
    bbox_out = dict(
        center=center,
        size_class=size_class,
        size_res=size_res,
        dir_class=dir_class,
        dir_res=dir_res)

    bbox3d = box_coder.decode(bbox_out)
    expected_bbox3d = torch.tensor(
        [[[0.8014, 3.4134, -0.6133, 0.9750, 2.2602, 0.9725, 1.6926],
          [2.6375, 8.4191, 2.0438, 0.5511, 0.4931, 0.9471, 2.6149],
          [4.2017, 5.2504, -0.7851, 0.6411, 0.5075, 0.9168, 1.5839],
          [-1.0088, 5.4107, 1.6293, 0.5064, 0.7017, 0.6602, 0.4605],
          [1.4837, 4.0268, 0.6222, 0.4071, 0.9951, 1.8243, 1.6786]]])
    assert torch.allclose(bbox3d, expected_bbox3d, atol=1e-4)

    # test split_pred
    cls_preds = torch.rand(2, 12, 256)
    reg_preds = torch.rand(2, 67, 256)
    base_xyz = torch.rand(2, 256, 3)
    results = box_coder.split_pred(cls_preds, reg_preds, base_xyz)
    obj_scores = results['obj_scores']
    center = results['center']
    dir_class = results['dir_class']
    dir_res_norm = results['dir_res_norm']
    dir_res = results['dir_res']
    size_class = results['size_class']
    size_res_norm = results['size_res_norm']
    size_res = results['size_res']
    sem_scores = results['sem_scores']
    assert obj_scores.shape == torch.Size([2, 256, 2])
    assert center.shape == torch.Size([2, 256, 3])
    assert dir_class.shape == torch.Size([2, 256, 12])
    assert dir_res_norm.shape == torch.Size([2, 256, 12])
    assert dir_res.shape == torch.Size([2, 256, 12])
    assert size_class.shape == torch.Size([2, 256, 10])
    assert size_res_norm.shape == torch.Size([2, 256, 10, 3])
    assert size_res.shape == torch.Size([2, 256, 10, 3])
    assert sem_scores.shape == torch.Size([2, 256, 10])


def test_anchor_free_box_coder():
    box_coder_cfg = dict(
        type='AnchorFreeBBoxCoder', num_dir_bins=12, with_rot=True)
    box_coder = build_bbox_coder(box_coder_cfg)

    # test encode
    gt_bboxes = LiDARInstance3DBoxes([[
        2.1227e+00, 5.7951e+00, -9.9900e-01, 1.6736e+00, 4.2419e+00,
        1.5473e+00, -1.5501e+00
    ],
                                      [
                                          1.1791e+01, 9.0276e+00, -8.5772e-01,
                                          1.6210e+00, 3.5367e+00, 1.4841e+00,
                                          -1.7369e+00
                                      ],
                                      [
                                          2.3638e+01, 9.6997e+00, -5.6713e-01,
                                          1.7578e+00, 4.6103e+00, 1.5999e+00,
                                          -1.4556e+00
                                      ]])
    gt_labels = torch.tensor([0, 0, 0])

    (center_targets, size_targets, dir_class_targets,
     dir_res_targets) = box_coder.encode(gt_bboxes, gt_labels)

    expected_center_target = torch.tensor([[2.1227, 5.7951, -0.2253],
                                           [11.7908, 9.0276, -0.1156],
                                           [23.6380, 9.6997, 0.2328]])
    expected_size_targets = torch.tensor([[0.8368, 2.1210, 0.7736],
                                          [0.8105, 1.7683, 0.7421],
                                          [0.8789, 2.3052, 0.8000]])
    expected_dir_class_target = torch.tensor([9, 9, 9])
    expected_dir_res_target = torch.tensor([0.0394, -0.3172, 0.2199])
    assert torch.allclose(center_targets, expected_center_target, atol=1e-4)
    assert torch.allclose(size_targets, expected_size_targets, atol=1e-4)
    assert torch.all(dir_class_targets == expected_dir_class_target)
    assert torch.allclose(dir_res_targets, expected_dir_res_target, atol=1e-3)

    # test decode
    center = torch.tensor([[[14.5954, 6.3312, 0.7671],
                            [67.5245, 22.4422, 1.5610],
                            [47.7693, -6.7980, 1.4395]]])

    size_res = torch.tensor([[[-1.0752, 1.8760, 0.7715],
                              [-0.8016, 1.1754, 0.0102],
                              [-1.2789, 0.5948, 0.4728]]])

    dir_class = torch.tensor([[[
        0.1512, 1.7914, -1.7658, 2.1572, -0.9215, 1.2139, 0.1749, 0.8606,
        1.1743, -0.7679, -1.6005, 0.4623
    ],
                               [
                                   -0.3957, 1.2026, -1.2677, 1.3863, -0.5754,
                                   1.7083, 0.2601, 0.1129, 0.7146, -0.1367,
                                   -1.2892, -0.0083
                               ],
                               [
                                   -0.8862, 1.2050, -1.3881, 1.6604, -0.9087,
                                   1.1907, -0.0280, 0.2027, 1.0644, -0.7205,
                                   -1.0738, 0.4748
                               ]]])

    dir_res = torch.tensor([[[
        1.1151, 0.5535, -0.2053, -0.6582, -0.1616, -0.1821, 0.4675, 0.6621,
        0.8146, -0.0448, -0.7253, -0.7171
    ],
                             [
                                 0.7888, 0.2478, -0.1962, -0.7267, 0.0573,
                                 -0.2398, 0.6984, 0.5859, 0.7507, -0.1980,
                                 -0.6538, -0.6602
                             ],
                             [
                                 0.9039, 0.6109, 0.1960, -0.5016, 0.0551,
                                 -0.4086, 0.3398, 0.2759, 0.7247, -0.0655,
                                 -0.5052, -0.9026
                             ]]])
    bbox_out = dict(
        center=center, size=size_res, dir_class=dir_class, dir_res=dir_res)

    bbox3d = box_coder.decode(bbox_out)
    expected_bbox3d = torch.tensor(
        [[[14.5954, 6.3312, 0.7671, 0.1000, 3.7521, 1.5429, 0.9126],
          [67.5245, 22.4422, 1.5610, 0.1000, 2.3508, 0.1000, 2.3782],
          [47.7693, -6.7980, 1.4395, 0.1000, 1.1897, 0.9456, 1.0692]]])
    assert torch.allclose(bbox3d, expected_bbox3d, atol=1e-4)

    # test split_pred
    cls_preds = torch.rand(2, 1, 256)
    reg_preds = torch.rand(2, 30, 256)
    base_xyz = torch.rand(2, 256, 3)
    results = box_coder.split_pred(cls_preds, reg_preds, base_xyz)
    obj_scores = results['obj_scores']
    center = results['center']
    center_offset = results['center_offset']
    dir_class = results['dir_class']
    dir_res_norm = results['dir_res_norm']
    dir_res = results['dir_res']
    size = results['size']
    assert obj_scores.shape == torch.Size([2, 1, 256])
    assert center.shape == torch.Size([2, 256, 3])
    assert center_offset.shape == torch.Size([2, 256, 3])
    assert dir_class.shape == torch.Size([2, 256, 12])
    assert dir_res_norm.shape == torch.Size([2, 256, 12])
    assert dir_res.shape == torch.Size([2, 256, 12])
    assert size.shape == torch.Size([2, 256, 3])


def test_centerpoint_bbox_coder():
    bbox_coder_cfg = dict(
        type='CenterPointBBoxCoder',
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        max_num=500,
        score_threshold=0.1,
        pc_range=[-51.2, -51.2],
        out_size_factor=4,
        voxel_size=[0.2, 0.2])

    bbox_coder = build_bbox_coder(bbox_coder_cfg)

    batch_dim = torch.rand([2, 3, 128, 128])
    batch_hei = torch.rand([2, 1, 128, 128])
    batch_hm = torch.rand([2, 2, 128, 128])
    batch_reg = torch.rand([2, 2, 128, 128])
    batch_rotc = torch.rand([2, 1, 128, 128])
    batch_rots = torch.rand([2, 1, 128, 128])
    batch_vel = torch.rand([2, 2, 128, 128])

    temp = bbox_coder.decode(batch_hm, batch_rots, batch_rotc, batch_hei,
                             batch_dim, batch_vel, batch_reg, 5)
    for i in range(len(temp)):
        assert temp[i]['bboxes'].shape == torch.Size([500, 9])
        assert temp[i]['scores'].shape == torch.Size([500])
        assert temp[i]['labels'].shape == torch.Size([500])


def test_point_xyzwhlr_bbox_coder():
    bbox_coder_cfg = dict(
        type='PointXYZWHLRBBoxCoder',
        use_mean_size=True,
        mean_size=[[3.9, 1.6, 1.56], [0.8, 0.6, 1.73], [1.76, 0.6, 1.73]])
    boxcoder = build_bbox_coder(bbox_coder_cfg)

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


def test_fcos3d_bbox_coder():
    # test a config without priors
    bbox_coder_cfg = dict(
        type='FCOS3DBBoxCoder',
        base_depths=None,
        base_dims=None,
        code_size=7,
        norm_on_bbox=True)
    bbox_coder = build_bbox_coder(bbox_coder_cfg)

    # test decode
    # [2, 7, 1, 1]
    batch_bbox = torch.tensor([[[[0.3130]], [[0.7094]], [[0.8743]], [[0.0570]],
                                [[0.5579]], [[0.1593]], [[0.4553]]],
                               [[[0.7758]], [[0.2298]], [[0.3925]], [[0.6307]],
                                [[0.4377]], [[0.3339]], [[0.1966]]]])
    batch_scale = nn.ModuleList([Scale(1.0) for _ in range(3)])
    stride = 2
    training = False
    cls_score = torch.randn([2, 2, 1, 1]).sigmoid()
    decode_bbox = bbox_coder.decode(batch_bbox, batch_scale, stride, training,
                                    cls_score)

    expected_bbox = torch.tensor([[[[0.6261]], [[1.4188]], [[2.3971]],
                                   [[1.0586]], [[1.7470]], [[1.1727]],
                                   [[0.4553]]],
                                  [[[1.5516]], [[0.4596]], [[1.4806]],
                                   [[1.8790]], [[1.5492]], [[1.3965]],
                                   [[0.1966]]]])
    assert torch.allclose(decode_bbox, expected_bbox, atol=1e-3)

    # test a config with priors
    prior_bbox_coder_cfg = dict(
        type='FCOS3DBBoxCoder',
        base_depths=((28., 13.), (25., 12.)),
        base_dims=((2., 3., 1.), (1., 2., 3.)),
        code_size=7,
        norm_on_bbox=True)
    prior_bbox_coder = build_bbox_coder(prior_bbox_coder_cfg)

    # test decode
    batch_bbox = torch.tensor([[[[0.3130]], [[0.7094]], [[0.8743]], [[0.0570]],
                                [[0.5579]], [[0.1593]], [[0.4553]]],
                               [[[0.7758]], [[0.2298]], [[0.3925]], [[0.6307]],
                                [[0.4377]], [[0.3339]], [[0.1966]]]])
    batch_scale = nn.ModuleList([Scale(1.0) for _ in range(3)])
    stride = 2
    training = False
    cls_score = torch.tensor([[[[0.5811]], [[0.6198]]], [[[0.4889]],
                                                         [[0.8142]]]])
    decode_bbox = prior_bbox_coder.decode(batch_bbox, batch_scale, stride,
                                          training, cls_score)
    expected_bbox = torch.tensor([[[[0.6260]], [[1.4188]], [[35.4916]],
                                   [[1.0587]], [[3.4940]], [[3.5181]],
                                   [[0.4553]]],
                                  [[[1.5516]], [[0.4596]], [[29.7100]],
                                   [[1.8789]], [[3.0983]], [[4.1892]],
                                   [[0.1966]]]])
    assert torch.allclose(decode_bbox, expected_bbox, atol=1e-3)

    # test decode_yaw
    decode_bbox = decode_bbox.permute(0, 2, 3, 1).view(-1, 7)
    batch_centers2d = torch.tensor([[100., 150.], [200., 100.]])
    batch_dir_cls = torch.tensor([0., 1.])
    dir_offset = 0.7854
    cam2img = torch.tensor([[700., 0., 450., 0.], [0., 700., 200., 0.],
                            [0., 0., 1., 0.], [0., 0., 0., 1.]])
    decode_bbox = prior_bbox_coder.decode_yaw(decode_bbox, batch_centers2d,
                                              batch_dir_cls, dir_offset,
                                              cam2img)
    expected_bbox = torch.tensor(
        [[0.6260, 1.4188, 35.4916, 1.0587, 3.4940, 3.5181, 3.1332],
         [1.5516, 0.4596, 29.7100, 1.8789, 3.0983, 4.1892, 6.1368]])
    assert torch.allclose(decode_bbox, expected_bbox, atol=1e-3)


def test_pgd_bbox_coder():
    # test a config without priors
    bbox_coder_cfg = dict(
        type='PGDBBoxCoder',
        base_depths=None,
        base_dims=None,
        code_size=7,
        norm_on_bbox=True)
    bbox_coder = build_bbox_coder(bbox_coder_cfg)

    # test decode_2d
    # [2, 27, 1, 1]
    batch_bbox = torch.tensor([[[[0.0103]], [[0.7394]], [[0.3296]], [[0.4708]],
                                [[0.1439]], [[0.0778]], [[0.9399]], [[0.8366]],
                                [[0.1264]], [[0.3030]], [[0.1898]], [[0.0714]],
                                [[0.4144]], [[0.4341]], [[0.6442]], [[0.2951]],
                                [[0.2890]], [[0.4486]], [[0.2848]], [[0.1071]],
                                [[0.9530]], [[0.9460]], [[0.3822]], [[0.9320]],
                                [[0.2611]], [[0.5580]], [[0.0397]]],
                               [[[0.8612]], [[0.1680]], [[0.5167]], [[0.8502]],
                                [[0.0377]], [[0.3615]], [[0.9550]], [[0.5219]],
                                [[0.1402]], [[0.6843]], [[0.2121]], [[0.9468]],
                                [[0.6238]], [[0.7918]], [[0.1646]], [[0.0500]],
                                [[0.6290]], [[0.3956]], [[0.2901]], [[0.4612]],
                                [[0.7333]], [[0.1194]], [[0.6999]], [[0.3980]],
                                [[0.3262]], [[0.7185]], [[0.4474]]]])
    batch_scale = nn.ModuleList([Scale(1.0) for _ in range(5)])
    stride = 2
    training = False
    cls_score = torch.randn([2, 2, 1, 1]).sigmoid()
    decode_bbox = bbox_coder.decode(batch_bbox, batch_scale, stride, training,
                                    cls_score)
    max_regress_range = 16
    pred_keypoints = True
    pred_bbox2d = True
    decode_bbox_w2d = bbox_coder.decode_2d(decode_bbox, batch_scale, stride,
                                           max_regress_range, training,
                                           pred_keypoints, pred_bbox2d)
    expected_decode_bbox_w2d = torch.tensor(
        [[[[0.0206]], [[1.4788]],
          [[1.3904]], [[1.6013]], [[1.1548]], [[1.0809]], [[0.9399]],
          [[10.9441]], [[2.0117]], [[4.7049]], [[3.0009]], [[1.1405]],
          [[6.2752]], [[6.5399]], [[9.0840]], [[4.5892]], [[4.4994]],
          [[6.7320]], [[4.4375]], [[1.7071]], [[11.8582]], [[11.8075]],
          [[5.8339]], [[1.8640]], [[0.5222]], [[1.1160]], [[0.0794]]],
         [[[1.7224]], [[0.3360]], [[1.6765]], [[2.3401]], [[1.0384]],
          [[1.4355]], [[0.9550]], [[7.6666]], [[2.2286]], [[9.5089]],
          [[3.3436]], [[11.8133]], [[8.8603]], [[10.5508]], [[2.6101]],
          [[0.7993]], [[8.9178]], [[6.0188]], [[4.5156]], [[6.8970]],
          [[10.0013]], [[1.9014]], [[9.6689]], [[0.7960]], [[0.6524]],
          [[1.4370]], [[0.8948]]]])
    assert torch.allclose(expected_decode_bbox_w2d, decode_bbox_w2d, atol=1e-3)

    # test decode_prob_depth
    # [10, 8]
    depth_cls_preds = torch.tensor([
        [-0.4383, 0.7207, -0.4092, 0.4649, 0.8526, 0.6186, -1.4312, -0.7150],
        [0.0621, 0.2369, 0.5170, 0.8484, -0.1099, 0.1829, -0.0072, 1.0618],
        [-1.6114, -0.1057, 0.5721, -0.5986, -2.0471, 0.8140, -0.8385, -0.4822],
        [0.0742, -0.3261, 0.4607, 1.8155, -0.3571, -0.0234, 0.3787, 2.3251],
        [1.0492, -0.6881, -0.0136, -1.8291, 0.8460, -1.0171, 2.5691, -0.8114],
        [0.0968, -0.5601, 1.0458, 0.2560, 1.3018, 0.1635, 0.0680, -1.0263],
        [-0.0765, 0.1498, -2.7321, 1.0047, -0.2505, 0.0871, -0.4820, -0.3003],
        [-0.4123, 0.2298, -0.1330, -0.6008, 0.6526, 0.7118, 0.9728, -0.7793],
        [1.6940, 0.3355, 1.4661, 0.5477, 0.8667, 0.0527, -0.9975, -0.0689],
        [0.4724, -0.3632, -0.0654, 0.4034, -0.3494, -0.7548, 0.7297, 1.2754]
    ])
    depth_range = (0, 70)
    depth_unit = 10
    num_depth_cls = 8
    uniform_prob_depth_preds = bbox_coder.decode_prob_depth(
        depth_cls_preds, depth_range, depth_unit, 'uniform', num_depth_cls)
    expected_preds = torch.tensor([
        32.0441, 38.4689, 36.1831, 48.2096, 46.1560, 32.7973, 33.2155, 39.9822,
        21.9905, 43.0161
    ])
    assert torch.allclose(uniform_prob_depth_preds, expected_preds, atol=1e-3)

    linear_prob_depth_preds = bbox_coder.decode_prob_depth(
        depth_cls_preds, depth_range, depth_unit, 'linear', num_depth_cls)
    expected_preds = torch.tensor([
        21.1431, 30.2421, 25.8964, 41.6116, 38.6234, 21.4582, 23.2993, 30.1111,
        13.9273, 36.8419
    ])
    assert torch.allclose(linear_prob_depth_preds, expected_preds, atol=1e-3)

    log_prob_depth_preds = bbox_coder.decode_prob_depth(
        depth_cls_preds, depth_range, depth_unit, 'log', num_depth_cls)
    expected_preds = torch.tensor([
        12.6458, 24.2487, 17.4015, 36.9375, 27.5982, 12.5510, 15.6635, 19.8408,
        9.1605, 31.3765
    ])
    assert torch.allclose(log_prob_depth_preds, expected_preds, atol=1e-3)

    loguniform_prob_depth_preds = bbox_coder.decode_prob_depth(
        depth_cls_preds, depth_range, depth_unit, 'loguniform', num_depth_cls)
    expected_preds = torch.tensor([
        6.9925, 10.3273, 8.9895, 18.6524, 16.4667, 7.3196, 7.5078, 11.3207,
        3.7987, 13.6095
    ])
    assert torch.allclose(
        loguniform_prob_depth_preds, expected_preds, atol=1e-3)


def test_smoke_bbox_coder():
    bbox_coder_cfg = dict(
        type='SMOKECoder',
        base_depth=(28.01, 16.32),
        base_dims=((3.88, 1.63, 1.53), (1.78, 1.70, 0.58), (0.88, 1.73, 0.67)),
        code_size=7)

    bbox_coder = build_bbox_coder(bbox_coder_cfg)
    regression = torch.rand([200, 8])
    points = torch.rand([200, 2])
    labels = torch.ones([2, 100])
    cam2imgs = torch.rand([2, 4, 4])
    trans_mats = torch.rand([2, 3, 3])

    img_metas = [dict(box_type_3d=CameraInstance3DBoxes) for i in range(2)]
    locations, dimensions, orientations = bbox_coder.decode(
        regression, points, labels, cam2imgs, trans_mats)
    assert locations.shape == torch.Size([200, 3])
    assert dimensions.shape == torch.Size([200, 3])
    assert orientations.shape == torch.Size([200, 1])
    bboxes = bbox_coder.encode(locations, dimensions, orientations, img_metas)
    assert bboxes.tensor.shape == torch.Size([200, 7])

    # specically designed to test orientation decode function's
    # special cases.
    ori_vector = torch.tensor([[-0.9, -0.01], [-0.9, 0.01]])
    locations = torch.tensor([[15., 2., 1.], [15., 2., -1.]])
    orientations = bbox_coder._decode_orientation(ori_vector, locations)
    assert orientations.shape == torch.Size([2, 1])


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
    bbox_coder = build_bbox_coder(bbox_coder_cfg)
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
