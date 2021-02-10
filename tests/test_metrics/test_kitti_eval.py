import numpy as np
import pytest
import torch

from mmdet3d.core.evaluation.kitti_utils.eval import (do_eval, eval_class,
                                                      kitti_eval)


def test_do_eval():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and CUDA')
    gt_name = np.array(
        ['Pedestrian', 'Cyclist', 'Car', 'Car', 'Car', 'DontCare', 'DontCare'])
    gt_truncated = np.array([0., 0., 0., -1., -1., -1., -1.])
    gt_occluded = np.array([0, 0, 3, -1, -1, -1, -1])
    gt_alpha = np.array([-1.57, 1.85, -1.65, -10., -10., -10., -10.])
    gt_bbox = np.array([[674.9179, 165.48549, 693.23694, 193.42134],
                        [676.21954, 165.70988, 691.63745, 193.83748],
                        [389.4093, 182.48041, 421.49072, 202.13422],
                        [232.0577, 186.16724, 301.94623, 217.4024],
                        [758.6537, 172.98509, 816.32434, 212.76743],
                        [532.37, 176.35, 542.68, 185.27],
                        [559.62, 175.83, 575.4, 183.15]])
    gt_dimensions = np.array([[12.34, 2.85, 2.63], [3.69, 1.67, 1.87],
                              [2.02, 1.86, 0.6], [-1., -1., -1.],
                              [-1., -1., -1.], [-1., -1., -1.],
                              [-1., -1., -1.]])
    gt_location = np.array([[4.700e-01, 1.490e+00, 6.944e+01],
                            [-1.653e+01, 2.390e+00, 5.849e+01],
                            [4.590e+00, 1.320e+00, 4.584e+01],
                            [-1.000e+03, -1.000e+03, -1.000e+03],
                            [-1.000e+03, -1.000e+03, -1.000e+03],
                            [-1.000e+03, -1.000e+03, -1.000e+03],
                            [-1.000e+03, -1.000e+03, -1.000e+03]])
    gt_rotation_y = [-1.56, 1.57, -1.55, -10., -10., -10., -10.]
    gt_anno = dict(
        name=gt_name,
        truncated=gt_truncated,
        occluded=gt_occluded,
        alpha=gt_alpha,
        bbox=gt_bbox,
        dimensions=gt_dimensions,
        location=gt_location,
        rotation_y=gt_rotation_y)

    dt_name = np.array(['Pedestrian', 'Cyclist', 'Car', 'Car', 'Car'])
    dt_truncated = np.array([0., 0., 0., 0., 0.])
    dt_occluded = np.array([0, 0, 0, 0, 0])
    dt_alpha = np.array([1.0744612, 1.2775835, 1.82563, 2.1145396, -1.7676563])
    dt_dimensions = np.array([[1.4441837, 1.7450154, 0.53160036],
                              [1.6501029, 1.7540325, 0.5162356],
                              [3.9313498, 1.4899347, 1.5655756],
                              [4.0111866, 1.5350999, 1.585221],
                              [3.7337692, 1.5117968, 1.5515774]])
    dt_location = np.array([[4.6671643, 1.285098, 45.836895],
                            [4.658241, 1.3088846, 45.85148],
                            [-16.598526, 2.298814, 58.618088],
                            [-18.629122, 2.2990575, 39.305355],
                            [7.0964046, 1.5178275, 29.32426]])
    dt_rotation_y = np.array(
        [1.174933, 1.3778262, 1.550529, 1.6742425, -1.5330327])
    dt_bbox = np.array([[674.9179, 165.48549, 693.23694, 193.42134],
                        [676.21954, 165.70988, 691.63745, 193.83748],
                        [389.4093, 182.48041, 421.49072, 202.13422],
                        [232.0577, 186.16724, 301.94623, 217.4024],
                        [758.6537, 172.98509, 816.32434, 212.76743]])
    dt_score = np.array(
        [0.18151495, 0.57920843, 0.27795696, 0.23100418, 0.21541929])
    dt_anno = dict(
        name=dt_name,
        truncated=dt_truncated,
        occluded=dt_occluded,
        alpha=dt_alpha,
        bbox=dt_bbox,
        dimensions=dt_dimensions,
        location=dt_location,
        rotation_y=dt_rotation_y,
        score=dt_score)
    current_classes = [1, 2, 0]
    min_overlaps = np.array([[[0.5, 0.5, 0.7], [0.5, 0.5, 0.7],
                              [0.5, 0.5, 0.7]],
                             [[0.5, 0.5, 0.7], [0.25, 0.25, 0.5],
                              [0.25, 0.25, 0.5]]])
    eval_types = ['bbox', 'bev', '3d', 'aos']
    mAP_bbox, mAP_bev, mAP_3d, mAP_aos = do_eval([gt_anno], [dt_anno],
                                                 current_classes, min_overlaps,
                                                 eval_types)
    expected_mAP_bbox = np.array([[[0., 0.], [9.09090909, 9.09090909],
                                   [9.09090909, 9.09090909]],
                                  [[0., 0.], [9.09090909, 9.09090909],
                                   [9.09090909, 9.09090909]],
                                  [[0., 0.], [9.09090909, 9.09090909],
                                   [9.09090909, 9.09090909]]])
    expected_mAP_bev = np.array([[[0., 0.], [0., 0.], [0., 0.]],
                                 [[0., 0.], [0., 0.], [0., 0.]],
                                 [[0., 0.], [0., 0.], [0., 0.]]])
    expected_mAP_3d = np.array([[[0., 0.], [0., 0.], [0., 0.]],
                                [[0., 0.], [0., 0.], [0., 0.]],
                                [[0., 0.], [0., 0.], [0., 0.]]])
    expected_mAP_aos = np.array([[[0., 0.], [0.55020816, 0.55020816],
                                  [0.55020816, 0.55020816]],
                                 [[0., 0.], [8.36633862, 8.36633862],
                                  [8.36633862, 8.36633862]],
                                 [[0., 0.], [8.63476893, 8.63476893],
                                  [8.63476893, 8.63476893]]])
    assert np.allclose(mAP_bbox, expected_mAP_bbox)
    assert np.allclose(mAP_bev, expected_mAP_bev)
    assert np.allclose(mAP_3d, expected_mAP_3d)
    assert np.allclose(mAP_aos, expected_mAP_aos)


def test_kitti_eval():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and CUDA')
    gt_name = np.array(
        ['Pedestrian', 'Cyclist', 'Car', 'Car', 'Car', 'DontCare', 'DontCare'])
    gt_truncated = np.array([0., 0., 0., -1., -1., -1., -1.])
    gt_occluded = np.array([0, 0, 3, -1, -1, -1, -1])
    gt_alpha = np.array([-1.57, 1.85, -1.65, -10., -10., -10., -10.])
    gt_bbox = np.array([[674.9179, 165.48549, 693.23694, 193.42134],
                        [676.21954, 165.70988, 691.63745, 193.83748],
                        [389.4093, 182.48041, 421.49072, 202.13422],
                        [232.0577, 186.16724, 301.94623, 217.4024],
                        [758.6537, 172.98509, 816.32434, 212.76743],
                        [532.37, 176.35, 542.68, 185.27],
                        [559.62, 175.83, 575.4, 183.15]])
    gt_dimensions = np.array([[12.34, 2.85, 2.63], [3.69, 1.67, 1.87],
                              [2.02, 1.86, 0.6], [-1., -1., -1.],
                              [-1., -1., -1.], [-1., -1., -1.],
                              [-1., -1., -1.]])
    gt_location = np.array([[4.700e-01, 1.490e+00, 6.944e+01],
                            [-1.653e+01, 2.390e+00, 5.849e+01],
                            [4.590e+00, 1.320e+00, 4.584e+01],
                            [-1.000e+03, -1.000e+03, -1.000e+03],
                            [-1.000e+03, -1.000e+03, -1.000e+03],
                            [-1.000e+03, -1.000e+03, -1.000e+03],
                            [-1.000e+03, -1.000e+03, -1.000e+03]])
    gt_rotation_y = [-1.56, 1.57, -1.55, -10., -10., -10., -10.]
    gt_anno = dict(
        name=gt_name,
        truncated=gt_truncated,
        occluded=gt_occluded,
        alpha=gt_alpha,
        bbox=gt_bbox,
        dimensions=gt_dimensions,
        location=gt_location,
        rotation_y=gt_rotation_y)

    dt_name = np.array(['Pedestrian', 'Cyclist', 'Car', 'Car', 'Car'])
    dt_truncated = np.array([0., 0., 0., 0., 0.])
    dt_occluded = np.array([0, 0, 0, 0, 0])
    dt_alpha = np.array([1.0744612, 1.2775835, 1.82563, 2.1145396, -1.7676563])
    dt_dimensions = np.array([[1.4441837, 1.7450154, 0.53160036],
                              [1.6501029, 1.7540325, 0.5162356],
                              [3.9313498, 1.4899347, 1.5655756],
                              [4.0111866, 1.5350999, 1.585221],
                              [3.7337692, 1.5117968, 1.5515774]])
    dt_location = np.array([[4.6671643, 1.285098, 45.836895],
                            [4.658241, 1.3088846, 45.85148],
                            [-16.598526, 2.298814, 58.618088],
                            [-18.629122, 2.2990575, 39.305355],
                            [7.0964046, 1.5178275, 29.32426]])
    dt_rotation_y = np.array(
        [1.174933, 1.3778262, 1.550529, 1.6742425, -1.5330327])
    dt_bbox = np.array([[674.9179, 165.48549, 693.23694, 193.42134],
                        [676.21954, 165.70988, 691.63745, 193.83748],
                        [389.4093, 182.48041, 421.49072, 202.13422],
                        [232.0577, 186.16724, 301.94623, 217.4024],
                        [758.6537, 172.98509, 816.32434, 212.76743]])
    dt_score = np.array(
        [0.18151495, 0.57920843, 0.27795696, 0.23100418, 0.21541929])
    dt_anno = dict(
        name=dt_name,
        truncated=dt_truncated,
        occluded=dt_occluded,
        alpha=dt_alpha,
        bbox=dt_bbox,
        dimensions=dt_dimensions,
        location=dt_location,
        rotation_y=dt_rotation_y,
        score=dt_score)

    current_classes = [1, 2, 0]
    result, ret_dict = kitti_eval([gt_anno], [dt_anno], current_classes)
    assert np.isclose(ret_dict['KITTI/Overall_2D_moderate'], 9.090909090909092)
    assert np.isclose(ret_dict['KITTI/Overall_2D_hard'], 9.090909090909092)


def test_eval_class():
    gt_name = np.array(
        ['Pedestrian', 'Cyclist', 'Car', 'Car', 'Car', 'DontCare', 'DontCare'])
    gt_truncated = np.array([0., 0., 0., -1., -1., -1., -1.])
    gt_occluded = np.array([0, 0, 3, -1, -1, -1, -1])
    gt_alpha = np.array([-1.57, 1.85, -1.65, -10., -10., -10., -10.])
    gt_bbox = np.array([[674.9179, 165.48549, 693.23694, 193.42134],
                        [676.21954, 165.70988, 691.63745, 193.83748],
                        [389.4093, 182.48041, 421.49072, 202.13422],
                        [232.0577, 186.16724, 301.94623, 217.4024],
                        [758.6537, 172.98509, 816.32434, 212.76743],
                        [532.37, 176.35, 542.68, 185.27],
                        [559.62, 175.83, 575.4, 183.15]])
    gt_anno = dict(
        name=gt_name,
        truncated=gt_truncated,
        occluded=gt_occluded,
        alpha=gt_alpha,
        bbox=gt_bbox)

    dt_name = np.array(['Pedestrian', 'Cyclist', 'Car', 'Car', 'Car'])
    dt_truncated = np.array([0., 0., 0., 0., 0.])
    dt_occluded = np.array([0, 0, 0, 0, 0])
    dt_alpha = np.array([1.0744612, 1.2775835, 1.82563, 2.1145396, -1.7676563])
    dt_bbox = np.array([[674.9179, 165.48549, 693.23694, 193.42134],
                        [676.21954, 165.70988, 691.63745, 193.83748],
                        [389.4093, 182.48041, 421.49072, 202.13422],
                        [232.0577, 186.16724, 301.94623, 217.4024],
                        [758.6537, 172.98509, 816.32434, 212.76743]])
    dt_score = np.array(
        [0.18151495, 0.57920843, 0.27795696, 0.23100418, 0.21541929])
    dt_anno = dict(
        name=dt_name,
        truncated=dt_truncated,
        occluded=dt_occluded,
        alpha=dt_alpha,
        bbox=dt_bbox,
        score=dt_score)
    current_classes = [1, 2, 0]
    difficultys = [0, 1, 2]
    metric = 0
    min_overlaps = np.array([[[0.5, 0.5, 0.7], [0.5, 0.5, 0.7],
                              [0.5, 0.5, 0.7]],
                             [[0.5, 0.5, 0.7], [0.25, 0.25, 0.5],
                              [0.25, 0.25, 0.5]]])

    ret_dict = eval_class([gt_anno], [dt_anno], current_classes, difficultys,
                          metric, min_overlaps, True, 1)
    recall_sum = np.sum(ret_dict['recall'])
    precision_sum = np.sum(ret_dict['precision'])
    orientation_sum = np.sum(ret_dict['orientation'])
    assert np.isclose(recall_sum, 16)
    assert np.isclose(precision_sum, 16)
    assert np.isclose(orientation_sum, 10.252829201850309)
