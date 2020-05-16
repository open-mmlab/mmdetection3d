import numpy as np
import torch

from mmdet3d.core.evaluation.indoor_eval import indoor_eval


def test_indoor_eval():
    det_infos = [[[[
        17.0,
        [
            3.2112048e+00, 5.6918913e-01, -8.6143613e-04, 1.1942449e-01,
            1.2988183e+00, 1.9952521e-01, 0.0000000e+00
        ],
        torch.as_tensor(0.9965866)
    ],
                   [
                       17.0,
                       [
                           3.248133, 0.4324184, 0.20038621, 0.17225507,
                           1.2736976, 0.32598814, 0.
                       ],
                       torch.as_tensor(0.99507546)
                   ],
                   [
                       3.0,
                       [
                           -1.2793612, -2.3155289, 0.15598366, 1.2822601,
                           2.2253945, 0.8361754, 0.
                       ],
                       torch.as_tensor(0.9916463)
                   ],
                   [
                       4.0,
                       [
                           2.8716104, -0.26416883, -0.04933786, 0.8190681,
                           0.60294986, 0.5769499, 0.
                       ],
                       torch.as_tensor(0.9702634)
                   ],
                   [
                       17.0,
                       [
                           -2.2109854, 0.19445783, -0.01614259, 0.40659013,
                           0.35370222, 0.3290567, 0.
                       ],
                       torch.as_tensor(0.95803124)
                   ],
                   [
                       4.0,
                       [
                           0.18409574, -3.3322976, 0.13188198, 0.960528,
                           0.91082716, 0.59325826, 0.
                       ],
                       torch.as_tensor(0.9483817)
                   ],
                   [
                       17.0,
                       [
                           1.9499326, 2.0099056, 0.32836294, 0.98528206,
                           1.0611539, 1.2197046, 0.
                       ],
                       torch.as_tensor(0.92196786)
                   ],
                   [
                       2.0,
                       [
                           -1.6204697, 2.3374724, 0.06042781, 0.49681002,
                           0.44362187, 0.47277915, 0.
                       ],
                       torch.as_tensor(0.87960094)
                   ],
                   [
                       17.0,
                       [
                           2.1414487, -1.7601899, 0.17694443, 1.0071366,
                           2.211764, 1.4690719, 0.
                       ],
                       torch.as_tensor(0.8586809)
                   ],
                   [
                       17.0,
                       [
                           -0.0484907, -3.639972, 0.41367513, 3.948648,
                           1.3692774, 1.0810001, 0.
                       ],
                       torch.as_tensor(0.80680436)
                   ]]]]

    label2cat = {
        0: 'cabinet',
        1: 'bed',
        2: 'chair',
        3: 'sofa',
        4: 'table',
        5: 'door',
        6: 'window',
        7: 'bookshelf',
        8: 'picture',
        9: 'counter',
        10: 'desk',
        11: 'curtain',
        12: 'refrigerator',
        13: 'showercurtrain',
        14: 'toilet',
        15: 'sink',
        16: 'bathtub',
        17: 'garbagebin'
    }
    gt_annos = [{
        'gt_num':
        12,
        'name': [
            'table', 'curtain', 'sofa', 'bookshelf', 'picture', 'chair',
            'chair', 'garbagebin', 'table', 'chair', 'chair', 'garbagebin'
        ],
        'gt_boxes_upright_depth':
        np.array([[
            3.48649406, 0.24238291, 0.48358256, 1.34014034, 0.72744983,
            0.40819243
        ],
                  [
                      -0.50371504, 3.25293231, 1.25988698, 2.12330937,
                      0.27563906, 1.80230701
                  ],
                  [
                      2.58820581, -0.99452347, 0.57732373, 2.94801593,
                      1.67463434, 0.88743341
                  ],
                  [
                      -1.9116497, -2.88811016, 0.70502496, 1.62386703,
                      0.60732293, 1.5857985
                  ],
                  [
                      -2.55324745, 0.6909315, 1.59045517, 0.07264495,
                      0.32018459, 0.3506999
                  ],
                  [
                      -2.3436017, -2.1659112, 0.254318, 0.5333302, 0.56154585,
                      0.64904487
                  ],
                  [
                      -2.32046795, -1.6880455, 0.26138437, 0.5586133,
                      0.59743834, 0.6378752
                  ],
                  [
                      -0.46495372, 3.22126102, 0.03188983, 1.92557108,
                      0.15160203, 0.24680007
                  ],
                  [
                      0.28087699, 2.88433838, 0.2495866, 0.57001019,
                      0.85177159, 0.5689255
                  ],
                  [
                      -0.05292395, 2.90586925, 0.23064148, 0.39113954,
                      0.43746281, 0.52981442
                  ],
                  [
                      0.25537968, 2.25156307, 0.24932587, 0.48192862,
                      0.51398182, 0.38040417
                  ],
                  [
                      2.60432816, 1.62303996, 0.42025632, 1.23775268,
                      0.51761389, 0.66034317
                  ]]),
        'class': [4, 11, 3, 7, 8, 2, 2, 17, 4, 2, 2, 17]
    }]

    ret_value = indoor_eval(gt_annos, det_infos, [0.25, 0.5], label2cat)
    garbagebin_AP_25 = ret_value['garbagebin_AP_0.25']
    sofa_AP_25 = ret_value['sofa_AP_0.25']
    table_AP_25 = ret_value['table_AP_0.25']
    chair_AP_25 = ret_value['chair_AP_0.25']
    mAP_25 = ret_value['mAP_0.25']
    garbagebin_rec_25 = ret_value['garbagebin_rec_0.25']
    sofa_rec_25 = ret_value['sofa_rec_0.25']
    table_rec_25 = ret_value['table_rec_0.25']
    chair_rec_25 = ret_value['chair_rec_0.25']
    mAR_25 = ret_value['mAR_0.25']
    table_AP_50 = ret_value['table_AP_0.50']
    mAP_50 = ret_value['mAP_0.50']
    table_rec_50 = ret_value['table_rec_0.50']
    mAR_50 = ret_value['mAR_0.50']
    assert garbagebin_AP_25 == 0.5
    assert sofa_AP_25 == 1.0
    assert table_AP_25 == 1.0
    assert chair_AP_25 == 0.25
    assert abs(mAP_25 - 0.392857) < 0.001
    assert garbagebin_rec_25 == 0.5
    assert sofa_rec_25 == 1.0
    assert table_rec_25 == 1.0
    assert chair_rec_25 == 0.25
    assert abs(mAR_25 - 0.392857) < 0.001
    assert table_AP_50 == 0.5
    assert abs(mAP_50 - 0.0714) < 0.001
    assert table_rec_50 == 0.5
    assert abs(mAR_50 - 0.0714) < 0.001
