# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet3d.core.bbox import DepthInstance3DBoxes, LiDARInstance3DBoxes
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
