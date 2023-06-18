# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmdet3d.core.evaluation.once_utils import once_eval

iou_threshold_dict = {
    'Car': 0.7,
    'Bus': 0.7,
    'Truck': 0.7,
    'Pedestrian': 0.3,
    'Cyclist': 0.5
}
CLASSES = ('Car', 'Bus', 'Truck', 'Pedestrian', 'Cyclist')


def test_once_eval():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and CUDA')
    gt_name = np.array(['Car', 'Car', 'Car', 'Bus', 'Truck', 'Pedestrian'])
    gt_boxes_2d = dict()
    gt_boxes_3d = np.array([[
        -2.8585022, 18.1365420, -0.6727633, 4.4581260, 1.9890739, 1.8130296,
        4.8961286
    ],
                            [
                                12.8541875, -24.5264692, -0.7169411, 4.7068519,
                                2.0083065, 1.8021881, 1.1704472
                            ],
                            [
                                -17.8355767, 72.4164570, -0.5144796, 4.5911440,
                                1.8013105, 1.8299062, 4.9583234
                            ],
                            [
                                -4.6526536, -8.2906180, 0.0635609, 11.8512713,
                                2.7021319, 3.4950517, 4.5645926
                            ],
                            [
                                9.3430294, -32.2851030, -0.0277876, 6.3511517,
                                2.2100205, 2.5318110, 1.1601115
                            ],
                            [
                                -38.0796513, 1.0845584, 0.5024134, 0.7685378,
                                0.8255108, 1.7059718, 3.1199616
                            ]])
    gt_anno = dict(name=gt_name, boxes_3d=gt_boxes_3d, boxes_2d=gt_boxes_2d)

    dt_name = np.array(['Car', 'Car', 'Car', 'Bus', 'Truck', 'Pedestrian'])
    dt_score = np.array([
        0.18151495, 0.57920843, 0.27795696, 0.23100418, 0.21541929, 0.234564156
    ])
    dt_boxes_3d = np.array([[
        -2.8585022, 18.1365420, -0.6727633, 4.4581260, 1.9890739, 1.8130296,
        4.8961286
    ],
                            [
                                12.8541875, -24.5264692, -0.7169411, 4.7068519,
                                2.0083065, 1.8021881, 1.1704472
                            ],
                            [
                                -17.8355767, 72.4164570, -0.5144796, 4.5911440,
                                1.8013105, 1.8299062, 4.9583234
                            ],
                            [
                                -4.6526536, -8.2906180, 0.0635609, 11.8512713,
                                2.7021319, 3.4950517, 4.5645926
                            ],
                            [
                                9.3430294, -32.2851030, -0.0277876, 6.3511517,
                                2.2100205, 2.5318110, 1.1601115
                            ],
                            [
                                -38.0796513, 1.0845584, 0.5024134, 0.7685378,
                                0.8255108, 1.7059718, 3.1199616
                            ]])
    dt_anno = dict(name=dt_name, boxes_3d=dt_boxes_3d, score=dt_score)

    ap_result_str, ap_dict = once_eval([gt_anno], [dt_anno],
                                       CLASSES,
                                       iou_thresholds=iou_threshold_dict,
                                       print_ok=True)
    assert ap_dict['AP_Car/overall'] == 100
    assert ap_dict['AP_Bus/overall'] == 100
    assert ap_dict['AP_Truck/overall'] == 100
    assert ap_dict['AP_Pedestrian/overall'] == 100
    assert ap_dict['AP_Cyclist/overall'] == 0
