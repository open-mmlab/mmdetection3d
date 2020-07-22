import torch

from mmdet3d.core.bbox import DepthInstance3DBoxes
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
