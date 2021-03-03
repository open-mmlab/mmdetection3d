import numpy as np
import pytest
import torch

from mmdet3d.core.evaluation.indoor_eval import average_precision, indoor_eval


def test_indoor_eval():
    if not torch.cuda.is_available():
        pytest.skip()
    from mmdet3d.core.bbox.structures import Box3DMode, DepthInstance3DBoxes
    det_infos = [{
        'labels_3d':
        torch.tensor([0, 1, 2, 2, 0, 3, 1, 2, 3, 2]),
        'boxes_3d':
        DepthInstance3DBoxes(
            torch.tensor([[
                -2.4089e-03, -3.3174e+00, 4.9438e-01, 2.1668e+00, 2.8431e-01,
                1.6506e+00, 0.0000e+00
            ],
                          [
                              -3.4269e-01, -2.7565e+00, 2.8144e-02, 6.8554e-01,
                              9.6854e-01, 6.1755e-01, 0.0000e+00
                          ],
                          [
                              -3.8320e+00, -1.0646e+00, 1.7074e-01, 2.4981e-01,
                              4.4708e-01, 6.2538e-01, 0.0000e+00
                          ],
                          [
                              4.1073e-01, 3.3757e+00, 3.4311e-01, 8.0617e-01,
                              2.8679e-01, 1.6060e+00, 0.0000e+00
                          ],
                          [
                              6.1199e-01, -3.1041e+00, 4.1873e-01, 1.2310e+00,
                              4.0162e-01, 1.7303e+00, 0.0000e+00
                          ],
                          [
                              -5.9877e-01, -2.6011e+00, 1.1148e+00, 1.5704e-01,
                              7.5957e-01, 9.6930e-01, 0.0000e+00
                          ],
                          [
                              2.7462e-01, -3.0088e+00, 6.5231e-02, 8.1208e-01,
                              4.1861e-01, 3.7339e-01, 0.0000e+00
                          ],
                          [
                              -1.4704e+00, -2.0024e+00, 2.7479e-01, 1.7888e+00,
                              1.0566e+00, 1.3704e+00, 0.0000e+00
                          ],
                          [
                              8.2727e-02, -3.1160e+00, 2.5690e-01, 1.4054e+00,
                              2.0772e-01, 9.6792e-01, 0.0000e+00
                          ],
                          [
                              2.6896e+00, 1.9881e+00, 1.1566e+00, 9.9885e-02,
                              3.5713e-01, 4.5638e-01, 0.0000e+00
                          ]]),
            origin=(0.5, 0.5, 0)),
        'scores_3d':
        torch.tensor([
            1.7516e-05, 1.0167e-06, 8.4486e-07, 7.1048e-02, 6.4274e-05,
            1.5003e-07, 5.8102e-06, 1.9399e-08, 5.3126e-07, 1.8630e-09
        ])
    }]

    label2cat = {
        0: 'cabinet',
        1: 'bed',
        2: 'chair',
        3: 'sofa',
    }
    gt_annos = [{
        'gt_num':
        10,
        'gt_boxes_upright_depth':
        np.array([[
            -2.4089e-03, -3.3174e+00, 4.9438e-01, 2.1668e+00, 2.8431e-01,
            1.6506e+00, 0.0000e+00
        ],
                  [
                      -3.4269e-01, -2.7565e+00, 2.8144e-02, 6.8554e-01,
                      9.6854e-01, 6.1755e-01, 0.0000e+00
                  ],
                  [
                      -3.8320e+00, -1.0646e+00, 1.7074e-01, 2.4981e-01,
                      4.4708e-01, 6.2538e-01, 0.0000e+00
                  ],
                  [
                      4.1073e-01, 3.3757e+00, 3.4311e-01, 8.0617e-01,
                      2.8679e-01, 1.6060e+00, 0.0000e+00
                  ],
                  [
                      6.1199e-01, -3.1041e+00, 4.1873e-01, 1.2310e+00,
                      4.0162e-01, 1.7303e+00, 0.0000e+00
                  ],
                  [
                      -5.9877e-01, -2.6011e+00, 1.1148e+00, 1.5704e-01,
                      7.5957e-01, 9.6930e-01, 0.0000e+00
                  ],
                  [
                      2.7462e-01, -3.0088e+00, 6.5231e-02, 8.1208e-01,
                      4.1861e-01, 3.7339e-01, 0.0000e+00
                  ],
                  [
                      -1.4704e+00, -2.0024e+00, 2.7479e-01, 1.7888e+00,
                      1.0566e+00, 1.3704e+00, 0.0000e+00
                  ],
                  [
                      8.2727e-02, -3.1160e+00, 2.5690e-01, 1.4054e+00,
                      2.0772e-01, 9.6792e-01, 0.0000e+00
                  ],
                  [
                      2.6896e+00, 1.9881e+00, 1.1566e+00, 9.9885e-02,
                      3.5713e-01, 4.5638e-01, 0.0000e+00
                  ]]),
        'class':
        np.array([0, 1, 2, 0, 0, 3, 1, 3, 3, 2])
    }]

    ret_value = indoor_eval(
        gt_annos,
        det_infos, [0.25, 0.5],
        label2cat,
        box_type_3d=DepthInstance3DBoxes,
        box_mode_3d=Box3DMode.DEPTH)

    assert np.isclose(ret_value['cabinet_AP_0.25'], 0.666667)
    assert np.isclose(ret_value['bed_AP_0.25'], 1.0)
    assert np.isclose(ret_value['chair_AP_0.25'], 0.5)
    assert np.isclose(ret_value['mAP_0.25'], 0.708333)
    assert np.isclose(ret_value['mAR_0.25'], 0.833333)


def test_indoor_eval_less_classes():
    if not torch.cuda.is_available():
        pytest.skip()
    from mmdet3d.core.bbox.structures import Box3DMode, DepthInstance3DBoxes
    det_infos = [{
        'labels_3d':
        torch.tensor([0]),
        'boxes_3d':
        DepthInstance3DBoxes(torch.tensor([[1., 1., 1., 1., 1., 1., 1.]])),
        'scores_3d':
        torch.tensor([.5])
    }, {
        'labels_3d':
        torch.tensor([1]),
        'boxes_3d':
        DepthInstance3DBoxes(torch.tensor([[1., 1., 1., 1., 1., 1., 1.]])),
        'scores_3d':
        torch.tensor([.5])
    }]

    label2cat = {0: 'cabinet', 1: 'bed', 2: 'chair'}
    gt_annos = [{
        'gt_num':
        2,
        'gt_boxes_upright_depth':
        np.array([[0., 0., 0., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1.]]),
        'class':
        np.array([2, 0])
    }, {
        'gt_num':
        1,
        'gt_boxes_upright_depth':
        np.array([
            [1., 1., 1., 1., 1., 1., 1.],
        ]),
        'class':
        np.array([1])
    }]

    ret_value = indoor_eval(
        gt_annos,
        det_infos, [0.25, 0.5],
        label2cat,
        box_type_3d=DepthInstance3DBoxes,
        box_mode_3d=Box3DMode.DEPTH)

    assert np.isclose(ret_value['mAP_0.25'], 0.666667)
    assert np.isclose(ret_value['mAR_0.25'], 0.666667)


def test_average_precision():
    ap = average_precision(
        np.array([[0.25, 0.5, 0.75], [0.25, 0.5, 0.75]]),
        np.array([[1., 1., 1.], [1., 1., 1.]]), '11points')
    assert abs(ap[0] - 0.06611571) < 0.001
