# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet3d.registry import TASK_UTILS
from mmdet3d.structures import LiDARInstance3DBoxes


def test_anchor_free_box_coder():
    box_coder_cfg = dict(
        type='AnchorFreeBBoxCoder', num_dir_bins=12, with_rot=True)
    box_coder = TASK_UTILS.build(box_coder_cfg)

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
