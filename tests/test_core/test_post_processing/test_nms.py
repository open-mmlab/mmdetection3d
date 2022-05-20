# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch


def test_aligned_3d_nms():
    from mmdet3d.core.post_processing import aligned_3d_nms

    boxes = torch.tensor([[1.2261, 0.6679, -1.2678, 2.6547, 1.0428, 0.1000],
                          [5.0919, 0.6512, 0.7238, 5.4821, 1.2451, 2.1095],
                          [6.8392, -1.2205, 0.8570, 7.6920, 0.3220, 3.2223],
                          [3.6900, -0.4235, -1.0380, 4.4415, 0.2671, -0.1442],
                          [4.8071, -1.4311, 0.7004, 5.5788, -0.6837, 1.2487],
                          [2.1807, -1.5811, -1.1289, 3.0151, -0.1346, -0.5351],
                          [4.4631, -4.2588, -1.1403, 5.3012, -3.4463, -0.3212],
                          [4.7607, -3.3311, 0.5993, 5.2976, -2.7874, 1.2273],
                          [3.1265, 0.7113, -0.0296, 3.8944, 1.3532, 0.9785],
                          [5.5828, -3.5350, 1.0105, 8.2841, -0.0405, 3.3614],
                          [3.0003, -2.1099, -1.0608, 5.3423, 0.0328, 0.6252],
                          [2.7148, 0.6082, -1.1738, 3.6995, 1.2375, -0.0209],
                          [4.9263, -0.2152, 0.2889, 5.6963, 0.3416, 1.3471],
                          [5.0713, 1.3459, -0.2598, 5.6278, 1.9300, 1.2835],
                          [4.5985, -2.3996, -0.3393, 5.2705, -1.7306, 0.5698],
                          [4.1386, 0.5658, 0.0422, 4.8937, 1.1983, 0.9911],
                          [2.7694, -1.9822, -1.0637, 4.0691, 0.3575, -0.1393],
                          [4.6464, -3.0123, -1.0694, 5.1421, -2.4450, -0.3758],
                          [3.4754, 0.4443, -1.1282, 4.6727, 1.3786, 0.2550],
                          [2.5905, -0.3504, -1.1202, 3.1599, 0.1153, -0.3036],
                          [4.1336, -3.4813, 1.1477, 6.2091, -0.8776, 2.6757],
                          [3.9966, 0.2069, -1.1148, 5.0841, 1.0525, -0.0648],
                          [4.3216, -1.8647, 0.4733, 6.2069, 0.6671, 3.3363],
                          [4.7683, 0.4286, -0.0500, 5.5642, 1.2906, 0.8902],
                          [1.7337, 0.7625, -1.0058, 3.0675, 1.3617, 0.3849],
                          [4.7193, -3.3687, -0.9635, 5.1633, -2.7656, 1.1001],
                          [4.4704, -2.7744, -1.1127, 5.0971, -2.0228, -0.3150],
                          [2.7027, 0.6122, -0.9169, 3.3083, 1.2117, 0.6129],
                          [4.8789, -2.0025, 0.8385, 5.5214, -1.3668, 1.3552],
                          [3.7856, -1.7582, -0.1738, 5.3373, -0.6300, 0.5558]])

    scores = torch.tensor([
        3.6414e-03, 2.2901e-02, 2.7576e-04, 1.2238e-02, 5.9310e-04, 1.2659e-01,
        2.4104e-02, 5.0742e-03, 2.3581e-03, 2.0946e-07, 8.8039e-01, 1.9127e-01,
        5.0469e-05, 9.3638e-03, 3.0663e-03, 9.4350e-03, 5.3380e-02, 1.7895e-01,
        2.0048e-01, 1.1294e-03, 3.0304e-08, 2.0237e-01, 1.0894e-08, 6.7972e-02,
        6.7156e-01, 9.3986e-04, 7.9470e-01, 3.9736e-01, 1.8000e-04, 7.9151e-04
    ])

    cls = torch.tensor([
        8, 8, 8, 3, 3, 1, 3, 3, 7, 8, 0, 6, 7, 8, 3, 7, 2, 7, 6, 3, 8, 6, 6, 7,
        6, 8, 7, 6, 3, 1
    ])

    pick = aligned_3d_nms(boxes, scores, cls, 0.25)
    expected_pick = torch.tensor([
        10, 26, 24, 27, 21, 18, 17, 5, 23, 16, 6, 1, 3, 15, 13, 7, 0, 14, 8,
        19, 25, 29, 4, 2, 28, 12, 9, 20, 22
    ])

    assert torch.all(pick == expected_pick)


def test_circle_nms():
    from mmdet3d.core.post_processing import circle_nms
    boxes = torch.tensor([[-11.1100, 2.1300, 0.8823],
                          [-11.2810, 2.2422, 0.8914],
                          [-10.3966, -0.3198, 0.8643],
                          [-10.2906, -13.3159,
                           0.8401], [5.6518, 9.9791, 0.8271],
                          [-11.2652, 13.3637, 0.8267],
                          [4.7768, -13.0409, 0.7810], [5.6621, 9.0422, 0.7753],
                          [-10.5561, 18.9627, 0.7518],
                          [-10.5643, 13.2293, 0.7200]])
    keep = circle_nms(boxes.numpy(), 0.175)
    expected_keep = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert np.all(keep == expected_keep)


# copied from tests/test_ops/test_iou3d.py from mmcv<=1.5
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_nms_bev():
    from mmdet3d.core.post_processing import nms_bev

    np_boxes = np.array(
        [[6.0, 3.0, 8.0, 7.0, 2.0], [3.0, 6.0, 9.0, 11.0, 1.0],
         [3.0, 7.0, 10.0, 12.0, 1.0], [1.0, 4.0, 13.0, 7.0, 3.0]],
        dtype=np.float32)
    np_scores = np.array([0.6, 0.9, 0.7, 0.2], dtype=np.float32)
    np_inds = np.array([1, 0, 3])
    boxes = torch.from_numpy(np_boxes)
    scores = torch.from_numpy(np_scores)
    inds = nms_bev(boxes.cuda(), scores.cuda(), thresh=0.3)

    assert np.allclose(inds.cpu().numpy(), np_inds)


# copied from tests/test_ops/test_iou3d.py from mmcv<=1.5
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_nms_normal_bev():
    from mmdet3d.core.post_processing import nms_normal_bev

    np_boxes = np.array(
        [[6.0, 3.0, 8.0, 7.0, 2.0], [3.0, 6.0, 9.0, 11.0, 1.0],
         [3.0, 7.0, 10.0, 12.0, 1.0], [1.0, 4.0, 13.0, 7.0, 3.0]],
        dtype=np.float32)
    np_scores = np.array([0.6, 0.9, 0.7, 0.2], dtype=np.float32)
    np_inds = np.array([1, 0, 3])
    boxes = torch.from_numpy(np_boxes)
    scores = torch.from_numpy(np_scores)
    inds = nms_normal_bev(boxes.cuda(), scores.cuda(), thresh=0.3)

    assert np.allclose(inds.cpu().numpy(), np_inds)
