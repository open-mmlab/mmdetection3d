# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmcv import Config
from mmcv.ops import SubMConv3d
from torch.nn import BatchNorm1d, ReLU

from mmdet3d.core.bbox import Box3DMode, LiDARInstance3DBoxes
from mmdet3d.core.bbox.samplers import IoUNegPiecewiseSampler
from mmdet3d.models import PartA2BboxHead
from mmdet3d.ops import make_sparse_convmodule


def test_loss():
    self = PartA2BboxHead(
        num_classes=3,
        seg_in_channels=16,
        part_in_channels=4,
        seg_conv_channels=[64, 64],
        part_conv_channels=[64, 64],
        merge_conv_channels=[128, 128],
        down_conv_channels=[128, 256],
        shared_fc_channels=[256, 512, 512, 512],
        cls_channels=[256, 256],
        reg_channels=[256, 256])

    cls_score = torch.Tensor([[-3.6810], [-3.9413], [-5.3971], [-17.1281],
                              [-5.9434], [-6.2251]])
    bbox_pred = torch.Tensor(
        [[
            -6.3016e-03, -5.2294e-03, -1.2793e-02, -1.0602e-02, -7.4086e-04,
            9.2471e-03, 7.3514e-03
        ],
         [
             -1.1975e-02, -1.1578e-02, -3.1219e-02, 2.7754e-02, 6.9775e-03,
             9.4042e-04, 9.0472e-04
         ],
         [
             3.7539e-03, -9.1897e-03, -5.3666e-03, -1.0380e-05, 4.3467e-03,
             4.2470e-03, 1.8355e-03
         ],
         [
             -7.6093e-02, -1.2497e-01, -9.2942e-02, 2.1404e-02, 2.3750e-02,
             1.0365e-01, -1.3042e-02
         ],
         [
             2.7577e-03, -1.1514e-02, -1.1097e-02, -2.4946e-03, 2.3268e-03,
             1.6797e-03, -1.4076e-03
         ],
         [
             3.9635e-03, -7.8551e-03, -3.5125e-03, 2.1229e-04, 9.7042e-03,
             1.7499e-03, -5.1254e-03
         ]])
    rois = torch.Tensor([
        [0.0000, 13.3711, -12.5483, -1.9306, 1.7027, 4.2836, 1.4283, -1.1499],
        [0.0000, 19.2472, -7.2655, -10.6641, 3.3078, 83.1976, 29.3337, 2.4501],
        [0.0000, 13.8012, -10.9791, -3.0617, 0.2504, 1.2518, 0.8807, 3.1034],
        [0.0000, 16.2736, -9.0284, -2.0494, 8.2697, 31.2336, 9.1006, 1.9208],
        [0.0000, 10.4462, -13.6879, -3.1869, 7.3366, 0.3518, 1.7199, -0.7225],
        [0.0000, 11.3374, -13.6671, -3.2332, 4.9934, 0.3750, 1.6033, -0.9665]
    ])
    labels = torch.Tensor([0.7100, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])
    bbox_targets = torch.Tensor(
        [[0.0598, 0.0243, -0.0984, -0.0454, 0.0066, 0.1114, 0.1714]])
    pos_gt_bboxes = torch.Tensor(
        [[13.6686, -12.5586, -2.1553, 1.6271, 4.3119, 1.5966, 2.1631]])
    reg_mask = torch.Tensor([1, 0, 0, 0, 0, 0])
    label_weights = torch.Tensor(
        [0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078])
    bbox_weights = torch.Tensor([1., 0., 0., 0., 0., 0.])

    loss = self.loss(cls_score, bbox_pred, rois, labels, bbox_targets,
                     pos_gt_bboxes, reg_mask, label_weights, bbox_weights)

    expected_loss_cls = torch.Tensor([
        2.0579e-02, 1.5005e-04, 3.5252e-05, 0.0000e+00, 2.0433e-05, 1.5422e-05
    ])
    expected_loss_bbox = torch.as_tensor(0.0622)
    expected_loss_corner = torch.Tensor([0.1374])

    assert torch.allclose(loss['loss_cls'], expected_loss_cls, 1e-3)
    assert torch.allclose(loss['loss_bbox'], expected_loss_bbox, 1e-3)
    assert torch.allclose(loss['loss_corner'], expected_loss_corner, 1e-3)


def test_get_targets():
    self = PartA2BboxHead(
        num_classes=3,
        seg_in_channels=16,
        part_in_channels=4,
        seg_conv_channels=[64, 64],
        part_conv_channels=[64, 64],
        merge_conv_channels=[128, 128],
        down_conv_channels=[128, 256],
        shared_fc_channels=[256, 512, 512, 512],
        cls_channels=[256, 256],
        reg_channels=[256, 256])

    sampling_result = IoUNegPiecewiseSampler(
        1,
        pos_fraction=0.55,
        neg_piece_fractions=[0.8, 0.2],
        neg_iou_piece_thrs=[0.55, 0.1],
        return_iou=True)
    sampling_result.pos_bboxes = torch.Tensor(
        [[8.1517, 0.0384, -1.9496, 1.5271, 4.1131, 1.4879, 1.2076]])
    sampling_result.pos_gt_bboxes = torch.Tensor(
        [[7.8417, -0.1405, -1.9652, 1.6122, 3.2838, 1.5331, -2.0835]])
    sampling_result.iou = torch.Tensor([
        6.7787e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 1.2839e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 7.0261e-04, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 5.8915e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 5.6628e-06,
        5.0271e-02, 0.0000e+00, 1.9608e-01, 0.0000e+00, 0.0000e+00, 2.3519e-01,
        1.6589e-02, 0.0000e+00, 1.0162e-01, 2.1634e-02, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 5.6326e-02,
        1.3810e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        4.5455e-02, 0.0000e+00, 1.0929e-03, 0.0000e+00, 8.8191e-02, 1.1012e-01,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 1.6236e-01, 0.0000e+00, 1.1342e-01,
        1.0636e-01, 9.9803e-02, 5.7394e-02, 0.0000e+00, 1.6773e-01, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 6.3464e-03,
        0.0000e+00, 2.7977e-01, 0.0000e+00, 3.1252e-01, 2.1642e-01, 2.2945e-01,
        0.0000e+00, 1.8297e-01, 0.0000e+00, 2.1908e-01, 1.1661e-01, 1.3513e-01,
        1.5898e-01, 7.4368e-03, 1.2523e-01, 1.4735e-04, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 1.0948e-01, 2.5889e-01, 4.4585e-04, 8.6483e-02, 1.6376e-01,
        0.0000e+00, 2.2894e-01, 2.7489e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        1.8334e-01, 1.0193e-01, 2.3389e-01, 1.1035e-01, 3.3700e-01, 1.4397e-01,
        1.0379e-01, 0.0000e+00, 1.1226e-01, 0.0000e+00, 0.0000e+00, 1.6201e-01,
        0.0000e+00, 1.3569e-01
    ])

    rcnn_train_cfg = Config({
        'assigner': [{
            'type': 'MaxIoUAssigner',
            'iou_calculator': {
                'type': 'BboxOverlaps3D',
                'coordinate': 'lidar'
            },
            'pos_iou_thr': 0.55,
            'neg_iou_thr': 0.55,
            'min_pos_iou': 0.55,
            'ignore_iof_thr': -1
        }, {
            'type': 'MaxIoUAssigner',
            'iou_calculator': {
                'type': 'BboxOverlaps3D',
                'coordinate': 'lidar'
            },
            'pos_iou_thr': 0.55,
            'neg_iou_thr': 0.55,
            'min_pos_iou': 0.55,
            'ignore_iof_thr': -1
        }, {
            'type': 'MaxIoUAssigner',
            'iou_calculator': {
                'type': 'BboxOverlaps3D',
                'coordinate': 'lidar'
            },
            'pos_iou_thr': 0.55,
            'neg_iou_thr': 0.55,
            'min_pos_iou': 0.55,
            'ignore_iof_thr': -1
        }],
        'sampler': {
            'type': 'IoUNegPiecewiseSampler',
            'num': 128,
            'pos_fraction': 0.55,
            'neg_piece_fractions': [0.8, 0.2],
            'neg_iou_piece_thrs': [0.55, 0.1],
            'neg_pos_ub': -1,
            'add_gt_as_proposals': False,
            'return_iou': True
        },
        'cls_pos_thr':
        0.75,
        'cls_neg_thr':
        0.25
    })

    label, bbox_targets, pos_gt_bboxes, reg_mask, label_weights, bbox_weights\
        = self.get_targets([sampling_result], rcnn_train_cfg)

    expected_label = torch.Tensor([
        0.8557, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0595, 0.0000, 0.1250, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0178, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0498, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.1740, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000
    ])

    expected_bbox_targets = torch.Tensor(
        [[-0.0632, 0.0516, 0.0047, 0.0542, -0.2252, 0.0299, -0.1495]])

    expected_pos_gt_bboxes = torch.Tensor(
        [[7.8417, -0.1405, -1.9652, 1.6122, 3.2838, 1.5331, -2.0835]])

    expected_reg_mask = torch.LongTensor([
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0
    ])

    expected_label_weights = torch.Tensor([
        0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078,
        0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078,
        0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078,
        0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078,
        0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078,
        0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078,
        0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078,
        0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078,
        0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078,
        0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078,
        0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078,
        0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078,
        0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078,
        0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078,
        0.0078, 0.0078
    ])

    expected_bbox_weights = torch.Tensor([
        1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0.
    ])

    assert torch.allclose(label, expected_label, 1e-2)
    assert torch.allclose(bbox_targets, expected_bbox_targets, 1e-2)
    assert torch.allclose(pos_gt_bboxes, expected_pos_gt_bboxes)
    assert torch.all(reg_mask == expected_reg_mask)
    assert torch.allclose(label_weights, expected_label_weights, 1e-2)
    assert torch.allclose(bbox_weights, expected_bbox_weights)


def test_get_bboxes():
    if not torch.cuda.is_available():
        pytest.skip()
    self = PartA2BboxHead(
        num_classes=3,
        seg_in_channels=16,
        part_in_channels=4,
        seg_conv_channels=[64, 64],
        part_conv_channels=[64, 64],
        merge_conv_channels=[128, 128],
        down_conv_channels=[128, 256],
        shared_fc_channels=[256, 512, 512, 512],
        cls_channels=[256, 256],
        reg_channels=[256, 256])

    rois = torch.Tensor([[
        0.0000e+00, 5.6284e+01, 2.5712e+01, -1.3196e+00, 1.5943e+00,
        3.7509e+00, 1.4969e+00, 1.2105e-03
    ],
                         [
                             0.0000e+00, 5.4685e+01, 2.9132e+01, -1.9178e+00,
                             1.6337e+00, 4.1116e+00, 1.5472e+00, -1.7312e+00
                         ],
                         [
                             0.0000e+00, 5.5927e+01, 2.5830e+01, -1.4099e+00,
                             1.5958e+00, 3.8861e+00, 1.4911e+00, -2.9276e+00
                         ],
                         [
                             0.0000e+00, 5.6306e+01, 2.6310e+01, -1.3729e+00,
                             1.5893e+00, 3.7448e+00, 1.4924e+00, 1.6071e-01
                         ],
                         [
                             0.0000e+00, 3.1633e+01, -5.8557e+00, -1.2541e+00,
                             1.6517e+00, 4.1829e+00, 1.5593e+00, -1.6037e+00
                         ],
                         [
                             0.0000e+00, 3.1789e+01, -5.5308e+00, -1.3012e+00,
                             1.6412e+00, 4.1070e+00, 1.5487e+00, -1.6517e+00
                         ]]).cuda()

    cls_score = torch.Tensor([[-2.2061], [-2.1121], [-1.4478], [-2.9614],
                              [-0.1761], [0.7357]]).cuda()

    bbox_pred = torch.Tensor(
        [[
            -4.7917e-02, -1.6504e-02, -2.2340e-02, 5.1296e-03, -2.0984e-02,
            1.0598e-02, -1.1907e-01
        ],
         [
             -1.6261e-02, -5.4005e-02, 6.2480e-03, 1.5496e-03, -1.3285e-02,
             8.1482e-03, -2.2707e-03
         ],
         [
             -3.9423e-02, 2.0151e-02, -2.1138e-02, -1.1845e-03, -1.5343e-02,
             5.7208e-03, 8.5646e-03
         ],
         [
             6.3104e-02, -3.9307e-02, 2.3005e-02, -7.0528e-03, -9.2637e-05,
             2.2656e-02, 1.6358e-02
         ],
         [
             -1.4864e-03, 5.6840e-02, 5.8247e-03, -3.5541e-03, -4.9658e-03,
             2.5036e-03, 3.0302e-02
         ],
         [
             -4.3259e-02, -1.9963e-02, 3.5004e-02, 3.7546e-03, 1.0876e-02,
             -3.9637e-04, 2.0445e-02
         ]]).cuda()

    class_labels = [torch.Tensor([2, 2, 2, 2, 2, 2]).cuda()]

    class_pred = [
        torch.Tensor([[1.0877e-05, 1.0318e-05, 2.6599e-01],
                      [1.3105e-05, 1.1904e-05, 2.4432e-01],
                      [1.4530e-05, 1.4619e-05, 2.4395e-01],
                      [1.3251e-05, 1.3038e-05, 2.3703e-01],
                      [2.9156e-05, 2.5521e-05, 2.2826e-01],
                      [3.1665e-05, 2.9054e-05, 2.2077e-01]]).cuda()
    ]

    cfg = Config(
        dict(
            use_rotate_nms=True,
            use_raw_score=True,
            nms_thr=0.01,
            score_thr=0.1))
    input_meta = dict(
        box_type_3d=LiDARInstance3DBoxes, box_mode_3d=Box3DMode.LIDAR)
    result_list = self.get_bboxes(rois, cls_score, bbox_pred, class_labels,
                                  class_pred, [input_meta], cfg)
    selected_bboxes, selected_scores, selected_label_preds = result_list[0]

    expected_selected_bboxes = torch.Tensor(
        [[56.0888, 25.6445, -1.3610, 1.6025, 3.6730, 1.5128, -0.1179],
         [54.4606, 29.2412, -1.9145, 1.6362, 4.0573, 1.5599, -1.7335],
         [31.8887, -5.8574, -1.2470, 1.6458, 4.1622, 1.5632, -1.5734]]).cuda()
    expected_selected_scores = torch.Tensor([-2.2061, -2.1121, -0.1761]).cuda()
    expected_selected_label_preds = torch.Tensor([2., 2., 2.]).cuda()
    assert torch.allclose(selected_bboxes.tensor, expected_selected_bboxes,
                          1e-3)
    assert torch.allclose(selected_scores, expected_selected_scores, 1e-3)
    assert torch.allclose(selected_label_preds, expected_selected_label_preds)


def test_multi_class_nms():
    if not torch.cuda.is_available():
        pytest.skip()

    self = PartA2BboxHead(
        num_classes=3,
        seg_in_channels=16,
        part_in_channels=4,
        seg_conv_channels=[64, 64],
        part_conv_channels=[64, 64],
        merge_conv_channels=[128, 128],
        down_conv_channels=[128, 256],
        shared_fc_channels=[256, 512, 512, 512],
        cls_channels=[256, 256],
        reg_channels=[256, 256])

    box_probs = torch.Tensor([[1.0877e-05, 1.0318e-05, 2.6599e-01],
                              [1.3105e-05, 1.1904e-05, 2.4432e-01],
                              [1.4530e-05, 1.4619e-05, 2.4395e-01],
                              [1.3251e-05, 1.3038e-05, 2.3703e-01],
                              [2.9156e-05, 2.5521e-05, 2.2826e-01],
                              [3.1665e-05, 2.9054e-05, 2.2077e-01],
                              [5.5738e-06, 6.2453e-06, 2.1978e-01],
                              [9.0193e-06, 9.2154e-06, 2.1418e-01],
                              [1.4004e-05, 1.3209e-05, 2.1316e-01],
                              [7.9210e-06, 8.1767e-06, 2.1304e-01]]).cuda()

    box_preds = torch.Tensor(
        [[
            5.6217e+01, 2.5908e+01, -1.3611e+00, 1.6025e+00, 3.6730e+00,
            1.5129e+00, 1.1786e-01
        ],
         [
             5.4653e+01, 2.8885e+01, -1.9145e+00, 1.6362e+00, 4.0574e+00,
             1.5599e+00, 1.7335e+00
         ],
         [
             5.5809e+01, 2.5686e+01, -1.4457e+00, 1.5939e+00, 3.8270e+00,
             1.4997e+00, 2.9191e+00
         ],
         [
             5.6107e+01, 2.6082e+01, -1.3557e+00, 1.5782e+00, 3.7444e+00,
             1.5266e+00, -1.7707e-01
         ],
         [
             3.1618e+01, -5.6004e+00, -1.2470e+00, 1.6459e+00, 4.1622e+00,
             1.5632e+00, 1.5734e+00
         ],
         [
             3.1605e+01, -5.6342e+00, -1.2467e+00, 1.6474e+00, 4.1519e+00,
             1.5481e+00, 1.6313e+00
         ],
         [
             5.6211e+01, 2.7294e+01, -1.5350e+00, 1.5422e+00, 3.7733e+00,
             1.5140e+00, -9.5846e-02
         ],
         [
             5.5907e+01, 2.7155e+01, -1.4712e+00, 1.5416e+00, 3.7611e+00,
             1.5142e+00, 5.2059e-02
         ],
         [
             5.4000e+01, 3.0585e+01, -1.6874e+00, 1.6495e+00, 4.0376e+00,
             1.5554e+00, 1.7900e+00
         ],
         [
             5.6007e+01, 2.6300e+01, -1.3945e+00, 1.5716e+00, 3.7064e+00,
             1.4715e+00, 2.9639e+00
         ]]).cuda()

    input_meta = dict(
        box_type_3d=LiDARInstance3DBoxes, box_mode_3d=Box3DMode.LIDAR)
    selected = self.multi_class_nms(box_probs, box_preds, 0.1, 0.001,
                                    input_meta)
    expected_selected = torch.Tensor([0, 1, 4, 8]).cuda()

    assert torch.all(selected == expected_selected)


def test_make_sparse_convmodule():
    with pytest.raises(AssertionError):
        # assert invalid order setting
        make_sparse_convmodule(
            in_channels=4,
            out_channels=8,
            kernel_size=3,
            indice_key='rcnn_part2',
            norm_cfg=dict(type='BN1d'),
            order=('norm', 'act', 'conv', 'norm'))

        # assert invalid type of order
        make_sparse_convmodule(
            in_channels=4,
            out_channels=8,
            kernel_size=3,
            indice_key='rcnn_part2',
            norm_cfg=dict(type='BN1d'),
            order=['norm', 'conv'])

        # assert invalid elements of order
        make_sparse_convmodule(
            in_channels=4,
            out_channels=8,
            kernel_size=3,
            indice_key='rcnn_part2',
            norm_cfg=dict(type='BN1d'),
            order=('conv', 'normal', 'activate'))

    sparse_convmodule = make_sparse_convmodule(
        in_channels=4,
        out_channels=64,
        kernel_size=3,
        padding=1,
        indice_key='rcnn_part0',
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01))

    assert isinstance(sparse_convmodule[0], SubMConv3d)
    assert isinstance(sparse_convmodule[1], BatchNorm1d)
    assert isinstance(sparse_convmodule[2], ReLU)
    assert sparse_convmodule[1].num_features == 64
    assert sparse_convmodule[1].eps == 0.001
    assert sparse_convmodule[1].affine is True
    assert sparse_convmodule[1].track_running_stats is True
    assert isinstance(sparse_convmodule[2], ReLU)
    assert sparse_convmodule[2].inplace is True

    pre_act = make_sparse_convmodule(
        in_channels=4,
        out_channels=8,
        kernel_size=3,
        indice_key='rcnn_part1',
        norm_cfg=dict(type='BN1d'),
        order=('norm', 'act', 'conv'))
    assert isinstance(pre_act[0], BatchNorm1d)
    assert isinstance(pre_act[1], ReLU)
    assert isinstance(pre_act[2], SubMConv3d)
