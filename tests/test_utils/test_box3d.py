# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np
import pytest
import torch

from mmdet3d.core.bbox import (BaseInstance3DBoxes, Box3DMode,
                               CameraInstance3DBoxes, Coord3DMode,
                               DepthInstance3DBoxes, LiDARInstance3DBoxes,
                               bbox3d2roi, bbox3d_mapping_back)
from mmdet3d.core.bbox.structures.utils import (get_box_type, limit_period,
                                                points_cam2img,
                                                rotation_3d_in_axis,
                                                xywhr2xyxyr)
from mmdet3d.core.points import CameraPoints, DepthPoints, LiDARPoints


def test_bbox3d_mapping_back():
    bboxes = BaseInstance3DBoxes(
        [[
            -5.24223238e+00, 4.00209696e+01, 2.97570381e-01, 2.06200000e+00,
            4.40900000e+00, 1.54800000e+00, -1.48801203e+00
        ],
         [
             -2.66751588e+01, 5.59499564e+00, -9.14345860e-01, 3.43000000e-01,
             4.58000000e-01, 7.82000000e-01, -4.62759755e+00
         ],
         [
             -5.80979675e+00, 3.54092357e+01, 2.00889888e-01, 2.39600000e+00,
             3.96900000e+00, 1.73200000e+00, -4.65203216e+00
         ],
         [
             -3.13086877e+01, 1.09007628e+00, -1.94612112e-01, 1.94400000e+00,
             3.85700000e+00, 1.72300000e+00, -2.81427027e+00
         ]])
    new_bboxes = bbox3d_mapping_back(bboxes, 1.1, True, True)
    expected_new_bboxes = torch.tensor(
        [[-4.7657, 36.3827, 0.2705, 1.8745, 4.0082, 1.4073, -1.4880],
         [-24.2501, 5.0864, -0.8312, 0.3118, 0.4164, 0.7109, -4.6276],
         [-5.2816, 32.1902, 0.1826, 2.1782, 3.6082, 1.5745, -4.6520],
         [-28.4624, 0.9910, -0.1769, 1.7673, 3.5064, 1.5664, -2.8143]])
    assert torch.allclose(new_bboxes.tensor, expected_new_bboxes, atol=1e-4)


def test_bbox3d2roi():
    bbox_0 = torch.tensor(
        [[-5.2422, 4.0020, 2.9757, 2.0620, 4.4090, 1.5480, -1.4880],
         [-5.8097, 3.5409, 2.0088, 2.3960, 3.9690, 1.7320, -4.6520]])
    bbox_1 = torch.tensor(
        [[-2.6675, 5.5949, -9.1434, 3.4300, 4.5800, 7.8200, -4.6275],
         [-3.1308, 1.0900, -1.9461, 1.9440, 3.8570, 1.7230, -2.8142]])
    bbox_list = [bbox_0, bbox_1]
    rois = bbox3d2roi(bbox_list)
    expected_rois = torch.tensor(
        [[0.0000, -5.2422, 4.0020, 2.9757, 2.0620, 4.4090, 1.5480, -1.4880],
         [0.0000, -5.8097, 3.5409, 2.0088, 2.3960, 3.9690, 1.7320, -4.6520],
         [1.0000, -2.6675, 5.5949, -9.1434, 3.4300, 4.5800, 7.8200, -4.6275],
         [1.0000, -3.1308, 1.0900, -1.9461, 1.9440, 3.8570, 1.7230, -2.8142]])
    assert torch.all(torch.eq(rois, expected_rois))


def test_base_boxes3d():
    # test empty initialization
    empty_boxes = []
    boxes = BaseInstance3DBoxes(empty_boxes)
    assert boxes.tensor.shape[0] == 0
    assert boxes.tensor.shape[1] == 7

    # Test init with origin
    gravity_center_box = np.array(
        [[
            -5.24223238e+00, 4.00209696e+01, 2.97570381e-01, 2.06200000e+00,
            4.40900000e+00, 1.54800000e+00, -1.48801203e+00
        ],
         [
             -2.66751588e+01, 5.59499564e+00, -9.14345860e-01, 3.43000000e-01,
             4.58000000e-01, 7.82000000e-01, -4.62759755e+00
         ],
         [
             -5.80979675e+00, 3.54092357e+01, 2.00889888e-01, 2.39600000e+00,
             3.96900000e+00, 1.73200000e+00, -4.65203216e+00
         ],
         [
             -3.13086877e+01, 1.09007628e+00, -1.94612112e-01, 1.94400000e+00,
             3.85700000e+00, 1.72300000e+00, -2.81427027e+00
         ]],
        dtype=np.float32)

    bottom_center_box = BaseInstance3DBoxes(
        gravity_center_box, origin=(0.5, 0.5, 0.5))

    assert bottom_center_box.yaw.shape[0] == 4


def test_lidar_boxes3d():
    # test empty initialization
    empty_boxes = []
    boxes = LiDARInstance3DBoxes(empty_boxes)
    assert boxes.tensor.shape[0] == 0
    assert boxes.tensor.shape[1] == 7

    # Test init with origin
    gravity_center_box = np.array(
        [[
            -5.24223238e+00, 4.00209696e+01, 2.97570381e-01, 2.06200000e+00,
            4.40900000e+00, 1.54800000e+00, -1.48801203e+00
        ],
         [
             -2.66751588e+01, 5.59499564e+00, -9.14345860e-01, 3.43000000e-01,
             4.58000000e-01, 7.82000000e-01, -4.62759755e+00
         ],
         [
             -5.80979675e+00, 3.54092357e+01, 2.00889888e-01, 2.39600000e+00,
             3.96900000e+00, 1.73200000e+00, -4.65203216e+00
         ],
         [
             -3.13086877e+01, 1.09007628e+00, -1.94612112e-01, 1.94400000e+00,
             3.85700000e+00, 1.72300000e+00, -2.81427027e+00
         ]],
        dtype=np.float32)
    bottom_center_box = LiDARInstance3DBoxes(
        gravity_center_box, origin=(0.5, 0.5, 0.5))
    expected_tensor = torch.tensor(
        [[
            -5.24223238e+00, 4.00209696e+01, -4.76429619e-01, 2.06200000e+00,
            4.40900000e+00, 1.54800000e+00, -1.48801203e+00
        ],
         [
             -2.66751588e+01, 5.59499564e+00, -1.30534586e+00, 3.43000000e-01,
             4.58000000e-01, 7.82000000e-01, -4.62759755e+00
         ],
         [
             -5.80979675e+00, 3.54092357e+01, -6.65110112e-01, 2.39600000e+00,
             3.96900000e+00, 1.73200000e+00, -4.65203216e+00
         ],
         [
             -3.13086877e+01, 1.09007628e+00, -1.05611211e+00, 1.94400000e+00,
             3.85700000e+00, 1.72300000e+00, -2.81427027e+00
         ]])
    assert torch.allclose(expected_tensor, bottom_center_box.tensor)

    # Test init with numpy array
    np_boxes = np.array([[
        1.7802081, 2.516249, -1.7501148, 1.75, 3.39, 1.65,
        1.48 - 0.13603681398218053 * 4
    ],
                         [
                             8.959413, 2.4567227, -1.6357126, 1.54, 4.01, 1.57,
                             1.62 - 0.13603681398218053 * 4
                         ]],
                        dtype=np.float32)
    boxes_1 = LiDARInstance3DBoxes(np_boxes)
    assert torch.allclose(boxes_1.tensor, torch.from_numpy(np_boxes))

    # test properties
    assert boxes_1.volume.size(0) == 2
    assert (boxes_1.center == boxes_1.bottom_center).all()
    assert repr(boxes) == (
        'LiDARInstance3DBoxes(\n    tensor([], size=(0, 7)))')

    # test init with torch.Tensor
    th_boxes = torch.tensor(
        [[
            28.29669987, -0.5557558, -1.30332506, 1.47000003, 2.23000002,
            1.48000002, -1.57000005 - 0.13603681398218053 * 4
        ],
         [
             26.66901946, 21.82302134, -1.73605708, 1.55999994, 3.48000002,
             1.39999998, -1.69000006 - 0.13603681398218053 * 4
         ],
         [
             31.31977974, 8.16214412, -1.62177875, 1.74000001, 3.76999998,
             1.48000002, 2.78999996 - 0.13603681398218053 * 4
         ]],
        dtype=torch.float32)
    boxes_2 = LiDARInstance3DBoxes(th_boxes)
    assert torch.allclose(boxes_2.tensor, th_boxes)

    # test clone/to/device
    boxes_2 = boxes_2.clone()
    boxes_1 = boxes_1.to(boxes_2.device)

    # test box concatenation
    expected_tensor = torch.tensor([[
        1.7802081, 2.516249, -1.7501148, 1.75, 3.39, 1.65,
        1.48 - 0.13603681398218053 * 4
    ],
                                    [
                                        8.959413, 2.4567227, -1.6357126, 1.54,
                                        4.01, 1.57,
                                        1.62 - 0.13603681398218053 * 4
                                    ],
                                    [
                                        28.2967, -0.5557558, -1.303325, 1.47,
                                        2.23, 1.48,
                                        -1.57 - 0.13603681398218053 * 4
                                    ],
                                    [
                                        26.66902, 21.82302, -1.736057, 1.56,
                                        3.48, 1.4,
                                        -1.69 - 0.13603681398218053 * 4
                                    ],
                                    [
                                        31.31978, 8.162144, -1.6217787, 1.74,
                                        3.77, 1.48,
                                        2.79 - 0.13603681398218053 * 4
                                    ]])
    boxes = LiDARInstance3DBoxes.cat([boxes_1, boxes_2])
    assert torch.allclose(boxes.tensor, expected_tensor)
    # concatenate empty list
    empty_boxes = LiDARInstance3DBoxes.cat([])
    assert empty_boxes.tensor.shape[0] == 0
    assert empty_boxes.tensor.shape[-1] == 7

    # test box flip
    points = torch.tensor([[1.2559, -0.6762, -1.4658],
                           [4.7814, -0.8784,
                            -1.3857], [6.7053, 0.2517, -0.9697],
                           [0.6533, -0.5520, -0.5265],
                           [4.5870, 0.5358, -1.4741]])
    expected_tensor = torch.tensor(
        [[
            1.7802081, -2.516249, -1.7501148, 1.75, 3.39, 1.65,
            1.6615927 - np.pi + 0.13603681398218053 * 4
        ],
         [
             8.959413, -2.4567227, -1.6357126, 1.54, 4.01, 1.57,
             1.5215927 - np.pi + 0.13603681398218053 * 4
         ],
         [
             28.2967, 0.5557558, -1.303325, 1.47, 2.23, 1.48,
             4.7115927 - np.pi + 0.13603681398218053 * 4
         ],
         [
             26.66902, -21.82302, -1.736057, 1.56, 3.48, 1.4,
             4.8315926 - np.pi + 0.13603681398218053 * 4
         ],
         [
             31.31978, -8.162144, -1.6217787, 1.74, 3.77, 1.48,
             0.35159278 - np.pi + 0.13603681398218053 * 4
         ]])
    expected_points = torch.tensor([[1.2559, 0.6762, -1.4658],
                                    [4.7814, 0.8784, -1.3857],
                                    [6.7053, -0.2517, -0.9697],
                                    [0.6533, 0.5520, -0.5265],
                                    [4.5870, -0.5358, -1.4741]])
    points = boxes.flip('horizontal', points)
    assert torch.allclose(boxes.tensor, expected_tensor)
    assert torch.allclose(points, expected_points, 1e-3)

    expected_tensor = torch.tensor(
        [[
            -1.7802, -2.5162, -1.7501, 1.7500, 3.3900, 1.6500,
            -1.6616 + np.pi * 2 - 0.13603681398218053 * 4
        ],
         [
             -8.9594, -2.4567, -1.6357, 1.5400, 4.0100, 1.5700,
             -1.5216 + np.pi * 2 - 0.13603681398218053 * 4
         ],
         [
             -28.2967, 0.5558, -1.3033, 1.4700, 2.2300, 1.4800,
             -4.7116 + np.pi * 2 - 0.13603681398218053 * 4
         ],
         [
             -26.6690, -21.8230, -1.7361, 1.5600, 3.4800, 1.4000,
             -4.8316 + np.pi * 2 - 0.13603681398218053 * 4
         ],
         [
             -31.3198, -8.1621, -1.6218, 1.7400, 3.7700, 1.4800,
             -0.3516 + np.pi * 2 - 0.13603681398218053 * 4
         ]])
    boxes_flip_vert = boxes.clone()
    points = boxes_flip_vert.flip('vertical', points)
    expected_points = torch.tensor([[-1.2559, 0.6762, -1.4658],
                                    [-4.7814, 0.8784, -1.3857],
                                    [-6.7053, -0.2517, -0.9697],
                                    [-0.6533, 0.5520, -0.5265],
                                    [-4.5870, -0.5358, -1.4741]])
    assert torch.allclose(boxes_flip_vert.tensor, expected_tensor, 1e-4)
    assert torch.allclose(points, expected_points)

    # test box rotation
    # with input torch.Tensor points and angle
    expected_tensor = torch.tensor(
        [[
            1.4225, -2.7344, -1.7501, 1.7500, 3.3900, 1.6500,
            1.7976 - np.pi + 0.13603681398218053 * 2
        ],
         [
             8.5435, -3.6491, -1.6357, 1.5400, 4.0100, 1.5700,
             1.6576 - np.pi + 0.13603681398218053 * 2
         ],
         [
             28.1106, -3.2869, -1.3033, 1.4700, 2.2300, 1.4800,
             4.8476 - np.pi + 0.13603681398218053 * 2
         ],
         [
             23.4630, -25.2382, -1.7361, 1.5600, 3.4800, 1.4000,
             4.9676 - np.pi + 0.13603681398218053 * 2
         ],
         [
             29.9235, -12.3342, -1.6218, 1.7400, 3.7700, 1.4800,
             0.4876 - np.pi + 0.13603681398218053 * 2
         ]])
    points, rot_mat_T = boxes.rotate(-0.13603681398218053, points)
    expected_points = torch.tensor([[-1.1526, 0.8403, -1.4658],
                                    [-4.6181, 1.5187, -1.3857],
                                    [-6.6775, 0.6600, -0.9697],
                                    [-0.5724, 0.6355, -0.5265],
                                    [-4.6173, 0.0912, -1.4741]])
    expected_rot_mat_T = torch.tensor([[0.9908, -0.1356, 0.0000],
                                       [0.1356, 0.9908, 0.0000],
                                       [0.0000, 0.0000, 1.0000]])
    assert torch.allclose(boxes.tensor, expected_tensor, 1e-3)
    assert torch.allclose(points, expected_points, 1e-3)
    assert torch.allclose(rot_mat_T, expected_rot_mat_T, 1e-3)

    # with input torch.Tensor points and rotation matrix
    points, rot_mat_T = boxes.rotate(0.13603681398218053, points)  # back
    rot_mat = np.array([[0.99076125, -0.13561762, 0.],
                        [0.13561762, 0.99076125, 0.], [0., 0., 1.]])
    points, rot_mat_T = boxes.rotate(rot_mat, points)
    assert torch.allclose(boxes.tensor, expected_tensor, 1e-3)
    assert torch.allclose(points, expected_points, 1e-3)
    assert torch.allclose(rot_mat_T, expected_rot_mat_T, 1e-3)

    # with input np.ndarray points and angle
    points_np = np.array([[-1.0280, 0.9888,
                           -1.4658], [-4.3695, 2.1310, -1.3857],
                          [-6.5263, 1.5595,
                           -0.9697], [-0.4809, 0.7073, -0.5265],
                          [-4.5623, 0.7166, -1.4741]])
    points_np, rot_mat_T_np = boxes.rotate(-0.13603681398218053, points_np)
    expected_points_np = np.array([[-0.8844, 1.1191, -1.4658],
                                   [-4.0401, 2.7039, -1.3857],
                                   [-6.2545, 2.4302, -0.9697],
                                   [-0.3805, 0.7660, -0.5265],
                                   [-4.4230, 1.3287, -1.4741]])
    expected_rot_mat_T_np = np.array([[0.9908, -0.1356, 0.0000],
                                      [0.1356, 0.9908, 0.0000],
                                      [0.0000, 0.0000, 1.0000]])

    assert np.allclose(points_np, expected_points_np, 1e-3)
    assert np.allclose(rot_mat_T_np, expected_rot_mat_T_np, 1e-3)

    # with input LiDARPoints and rotation matrix
    points_np, rot_mat_T_np = boxes.rotate(0.13603681398218053, points_np)
    lidar_points = LiDARPoints(points_np)
    lidar_points, rot_mat_T_np = boxes.rotate(rot_mat, lidar_points)
    points_np = lidar_points.tensor.numpy()

    assert np.allclose(points_np, expected_points_np, 1e-3)
    assert np.allclose(rot_mat_T_np, expected_rot_mat_T_np, 1e-3)

    # test box scaling
    expected_tensor = torch.tensor([[
        1.0443488, -2.9183323, -1.7599131, 1.7597977, 3.4089797, 1.6592377,
        1.9336663 - np.pi
    ],
                                    [
                                        8.014273, -4.8007393, -1.6448704,
                                        1.5486219, 4.0324507, 1.57879,
                                        1.7936664 - np.pi
                                    ],
                                    [
                                        27.558605, -7.1084175, -1.310622,
                                        1.4782301, 2.242485, 1.488286,
                                        4.9836664 - np.pi
                                    ],
                                    [
                                        19.934517, -28.344835, -1.7457767,
                                        1.5687338, 3.4994833, 1.4078381,
                                        5.1036663 - np.pi
                                    ],
                                    [
                                        28.130915, -16.369587, -1.6308585,
                                        1.7497417, 3.791107, 1.488286,
                                        0.6236664 - np.pi
                                    ]])
    boxes.scale(1.00559866335275)
    assert torch.allclose(boxes.tensor, expected_tensor)

    # test box translation
    expected_tensor = torch.tensor([[
        1.1281544, -3.0507944, -1.9169292, 1.7597977, 3.4089797, 1.6592377,
        1.9336663 - np.pi
    ],
                                    [
                                        8.098079, -4.9332013, -1.8018866,
                                        1.5486219, 4.0324507, 1.57879,
                                        1.7936664 - np.pi
                                    ],
                                    [
                                        27.64241, -7.2408795, -1.4676381,
                                        1.4782301, 2.242485, 1.488286,
                                        4.9836664 - np.pi
                                    ],
                                    [
                                        20.018322, -28.477297, -1.9027928,
                                        1.5687338, 3.4994833, 1.4078381,
                                        5.1036663 - np.pi
                                    ],
                                    [
                                        28.21472, -16.502048, -1.7878747,
                                        1.7497417, 3.791107, 1.488286,
                                        0.6236664 - np.pi
                                    ]])
    boxes.translate([0.0838056, -0.13246193, -0.15701613])
    assert torch.allclose(boxes.tensor, expected_tensor)

    # test bbox in_range_bev
    expected_tensor = torch.tensor(
        [[1.1282, -3.0508, 1.7598, 3.4090, -1.2079],
         [8.0981, -4.9332, 1.5486, 4.0325, -1.3479],
         [27.6424, -7.2409, 1.4782, 2.2425, 1.8421],
         [20.0183, -28.4773, 1.5687, 3.4995, 1.9621],
         [28.2147, -16.5020, 1.7497, 3.7911, -2.5179]])
    assert torch.allclose(boxes.bev, expected_tensor, atol=1e-3)
    expected_tensor = torch.tensor([1, 1, 1, 1, 1], dtype=torch.bool)
    mask = boxes.in_range_bev([0., -40., 70.4, 40.])
    assert (mask == expected_tensor).all()
    mask = boxes.nonempty()
    assert (mask == expected_tensor).all()

    # test bbox in_range
    expected_tensor = torch.tensor([1, 1, 0, 0, 0], dtype=torch.bool)
    mask = boxes.in_range_3d([0, -20, -2, 22, 2, 5])
    assert (mask == expected_tensor).all()

    # test bbox indexing
    index_boxes = boxes[2:5]
    expected_tensor = torch.tensor([[
        27.64241, -7.2408795, -1.4676381, 1.4782301, 2.242485, 1.488286,
        4.9836664 - np.pi
    ],
                                    [
                                        20.018322, -28.477297, -1.9027928,
                                        1.5687338, 3.4994833, 1.4078381,
                                        5.1036663 - np.pi
                                    ],
                                    [
                                        28.21472, -16.502048, -1.7878747,
                                        1.7497417, 3.791107, 1.488286,
                                        0.6236664 - np.pi
                                    ]])
    assert len(index_boxes) == 3
    assert torch.allclose(index_boxes.tensor, expected_tensor)

    index_boxes = boxes[2]
    expected_tensor = torch.tensor([[
        27.64241, -7.2408795, -1.4676381, 1.4782301, 2.242485, 1.488286,
        4.9836664 - np.pi
    ]])
    assert len(index_boxes) == 1
    assert torch.allclose(index_boxes.tensor, expected_tensor)

    index_boxes = boxes[[2, 4]]
    expected_tensor = torch.tensor([[
        27.64241, -7.2408795, -1.4676381, 1.4782301, 2.242485, 1.488286,
        4.9836664 - np.pi
    ],
                                    [
                                        28.21472, -16.502048, -1.7878747,
                                        1.7497417, 3.791107, 1.488286,
                                        0.6236664 - np.pi
                                    ]])
    assert len(index_boxes) == 2
    assert torch.allclose(index_boxes.tensor, expected_tensor)

    # test iteration
    for i, box in enumerate(index_boxes):
        torch.allclose(box, expected_tensor[i])

    # test properties
    assert torch.allclose(boxes.bottom_center, boxes.tensor[:, :3])
    expected_tensor = (
        boxes.tensor[:, :3] - boxes.tensor[:, 3:6] *
        (torch.tensor([0.5, 0.5, 0]) - torch.tensor([0.5, 0.5, 0.5])))
    assert torch.allclose(boxes.gravity_center, expected_tensor)

    boxes.limit_yaw()
    assert (boxes.tensor[:, 6] <= np.pi / 2).all()
    assert (boxes.tensor[:, 6] >= -np.pi / 2).all()

    Box3DMode.convert(boxes, Box3DMode.LIDAR, Box3DMode.LIDAR)
    expected_tensor = boxes.tensor.clone()
    assert torch.allclose(expected_tensor, boxes.tensor)

    boxes.flip()
    boxes.flip()
    boxes.limit_yaw()
    assert torch.allclose(expected_tensor, boxes.tensor)

    # test nearest_bev
    expected_tensor = torch.tensor([[-0.5763, -3.9307, 2.8326, -2.1709],
                                    [6.0819, -5.7075, 10.1143, -4.1589],
                                    [26.5212, -7.9800, 28.7637, -6.5018],
                                    [18.2686, -29.2617, 21.7681, -27.6929],
                                    [27.3398, -18.3976, 29.0896, -14.6065]])
    assert torch.allclose(
        boxes.nearest_bev, expected_tensor, rtol=1e-4, atol=1e-7)

    expected_tensor = torch.tensor([[[-7.7767e-01, -2.8332e+00, -1.9169e+00],
                                     [-7.7767e-01, -2.8332e+00, -2.5769e-01],
                                     [2.4093e+00, -1.6232e+00, -2.5769e-01],
                                     [2.4093e+00, -1.6232e+00, -1.9169e+00],
                                     [-1.5301e-01, -4.4784e+00, -1.9169e+00],
                                     [-1.5301e-01, -4.4784e+00, -2.5769e-01],
                                     [3.0340e+00, -3.2684e+00, -2.5769e-01],
                                     [3.0340e+00, -3.2684e+00, -1.9169e+00]],
                                    [[5.9606e+00, -4.6237e+00, -1.8019e+00],
                                     [5.9606e+00, -4.6237e+00, -2.2310e-01],
                                     [9.8933e+00, -3.7324e+00, -2.2310e-01],
                                     [9.8933e+00, -3.7324e+00, -1.8019e+00],
                                     [6.3029e+00, -6.1340e+00, -1.8019e+00],
                                     [6.3029e+00, -6.1340e+00, -2.2310e-01],
                                     [1.0236e+01, -5.2427e+00, -2.2310e-01],
                                     [1.0236e+01, -5.2427e+00, -1.8019e+00]],
                                    [[2.6364e+01, -6.8292e+00, -1.4676e+00],
                                     [2.6364e+01, -6.8292e+00, 2.0648e-02],
                                     [2.8525e+01, -6.2283e+00, 2.0648e-02],
                                     [2.8525e+01, -6.2283e+00, -1.4676e+00],
                                     [2.6760e+01, -8.2534e+00, -1.4676e+00],
                                     [2.6760e+01, -8.2534e+00, 2.0648e-02],
                                     [2.8921e+01, -7.6525e+00, 2.0648e-02],
                                     [2.8921e+01, -7.6525e+00, -1.4676e+00]],
                                    [[1.8102e+01, -2.8420e+01, -1.9028e+00],
                                     [1.8102e+01, -2.8420e+01, -4.9495e-01],
                                     [2.1337e+01, -2.7085e+01, -4.9495e-01],
                                     [2.1337e+01, -2.7085e+01, -1.9028e+00],
                                     [1.8700e+01, -2.9870e+01, -1.9028e+00],
                                     [1.8700e+01, -2.9870e+01, -4.9495e-01],
                                     [2.1935e+01, -2.8535e+01, -4.9495e-01],
                                     [2.1935e+01, -2.8535e+01, -1.9028e+00]],
                                    [[2.8612e+01, -1.8552e+01, -1.7879e+00],
                                     [2.8612e+01, -1.8552e+01, -2.9959e-01],
                                     [2.6398e+01, -1.5474e+01, -2.9959e-01],
                                     [2.6398e+01, -1.5474e+01, -1.7879e+00],
                                     [3.0032e+01, -1.7530e+01, -1.7879e+00],
                                     [3.0032e+01, -1.7530e+01, -2.9959e-01],
                                     [2.7818e+01, -1.4452e+01, -2.9959e-01],
                                     [2.7818e+01, -1.4452e+01, -1.7879e+00]]])

    assert torch.allclose(boxes.corners, expected_tensor, rtol=1e-4, atol=1e-7)

    # test new_box
    new_box1 = boxes.new_box([[1, 2, 3, 4, 5, 6, 7]])
    assert torch.allclose(
        new_box1.tensor,
        torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=boxes.tensor.dtype))
    assert new_box1.device == boxes.device
    assert new_box1.with_yaw == boxes.with_yaw
    assert new_box1.box_dim == boxes.box_dim

    new_box2 = boxes.new_box(np.array([[1, 2, 3, 4, 5, 6, 7]]))
    assert torch.allclose(
        new_box2.tensor,
        torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=boxes.tensor.dtype))

    new_box3 = boxes.new_box(torch.tensor([[1, 2, 3, 4, 5, 6, 7]]))
    assert torch.allclose(
        new_box3.tensor,
        torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=boxes.tensor.dtype))


def test_boxes_conversion():
    """Test the conversion of boxes between different modes.

    CommandLine:
        xdoctest tests/test_box3d.py::test_boxes_conversion zero
    """
    lidar_boxes = LiDARInstance3DBoxes(
        [[1.7802081, 2.516249, -1.7501148, 1.75, 3.39, 1.65, 1.48],
         [8.959413, 2.4567227, -1.6357126, 1.54, 4.01, 1.57, 1.62],
         [28.2967, -0.5557558, -1.303325, 1.47, 2.23, 1.48, -1.57],
         [26.66902, 21.82302, -1.736057, 1.56, 3.48, 1.4, -1.69],
         [31.31978, 8.162144, -1.6217787, 1.74, 3.77, 1.48, 2.79]])
    cam_box_tensor = Box3DMode.convert(lidar_boxes.tensor, Box3DMode.LIDAR,
                                       Box3DMode.CAM)
    expected_box = lidar_boxes.convert_to(Box3DMode.CAM)
    assert torch.equal(expected_box.tensor, cam_box_tensor)

    # Some properties should be the same
    cam_boxes = CameraInstance3DBoxes(cam_box_tensor)
    assert torch.equal(cam_boxes.height, lidar_boxes.height)
    assert torch.equal(cam_boxes.top_height, -lidar_boxes.top_height)
    assert torch.equal(cam_boxes.bottom_height, -lidar_boxes.bottom_height)
    assert torch.allclose(cam_boxes.volume, lidar_boxes.volume)

    lidar_box_tensor = Box3DMode.convert(cam_box_tensor, Box3DMode.CAM,
                                         Box3DMode.LIDAR)
    expected_tensor = torch.tensor(
        [[1.7802081, 2.516249, -1.7501148, 1.75, 3.39, 1.65, 1.48],
         [8.959413, 2.4567227, -1.6357126, 1.54, 4.01, 1.57, 1.62],
         [28.2967, -0.5557558, -1.303325, 1.47, 2.23, 1.48, -1.57],
         [26.66902, 21.82302, -1.736057, 1.56, 3.48, 1.4, -1.69],
         [31.31978, 8.162144, -1.6217787, 1.74, 3.77, 1.48, 2.79]])

    assert torch.allclose(expected_tensor, lidar_box_tensor)
    assert torch.allclose(lidar_boxes.tensor, lidar_box_tensor)

    depth_box_tensor = Box3DMode.convert(cam_box_tensor, Box3DMode.CAM,
                                         Box3DMode.DEPTH)
    depth_to_cam_box_tensor = Box3DMode.convert(depth_box_tensor,
                                                Box3DMode.DEPTH, Box3DMode.CAM)
    assert torch.allclose(cam_box_tensor, depth_to_cam_box_tensor)

    # test similar mode conversion
    same_results = Box3DMode.convert(depth_box_tensor, Box3DMode.DEPTH,
                                     Box3DMode.DEPTH)
    assert torch.equal(same_results, depth_box_tensor)

    # test conversion with a given rt_mat
    camera_boxes = CameraInstance3DBoxes(
        [[0.06, 1.77, 21.4, 3.2, 1.61, 1.66, -1.54],
         [6.59, 1.53, 6.76, 12.78, 3.66, 2.28, 1.55],
         [6.71, 1.59, 22.18, 14.73, 3.64, 2.32, 1.59],
         [7.11, 1.58, 34.54, 10.04, 3.61, 2.32, 1.61],
         [7.78, 1.65, 45.95, 12.83, 3.63, 2.34, 1.64]])

    rect = torch.tensor(
        [[0.9999239, 0.00983776, -0.00744505, 0.],
         [-0.0098698, 0.9999421, -0.00427846, 0.],
         [0.00740253, 0.00435161, 0.9999631, 0.], [0., 0., 0., 1.]],
        dtype=torch.float32)

    Trv2c = torch.tensor(
        [[7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
         [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
         [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01],
         [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]],
        dtype=torch.float32)

    # coord sys refactor (reverse sign of yaw)
    expected_tensor = torch.tensor(
        [[
            2.16902434e+01, -4.06038554e-02, -1.61906639e+00, 3.20000005e+00,
            1.65999997e+00, 1.61000001e+00, 1.53999996e+00 - np.pi / 2
        ],
         [
             7.05006905e+00, -6.57459601e+00, -1.60107949e+00, 1.27799997e+01,
             2.27999997e+00, 3.66000009e+00, -1.54999995e+00 - np.pi / 2
         ],
         [
             2.24698818e+01, -6.69203759e+00, -1.50118145e+00, 1.47299995e+01,
             2.31999993e+00, 3.64000010e+00, -1.59000003e+00 + 3 * np.pi / 2
         ],
         [
             3.48291965e+01, -7.09058388e+00, -1.36622983e+00, 1.00400000e+01,
             2.31999993e+00, 3.60999990e+00, -1.61000001e+00 + 3 * np.pi / 2
         ],
         [
             4.62394617e+01, -7.75838800e+00, -1.32405020e+00, 1.28299999e+01,
             2.33999991e+00, 3.63000011e+00, -1.63999999e+00 + 3 * np.pi / 2
         ]],
        dtype=torch.float32)

    rt_mat = rect @ Trv2c
    # test conversion with Box type
    cam_to_lidar_box = Box3DMode.convert(camera_boxes, Box3DMode.CAM,
                                         Box3DMode.LIDAR, rt_mat.inverse())
    assert torch.allclose(cam_to_lidar_box.tensor, expected_tensor)

    lidar_to_cam_box = Box3DMode.convert(cam_to_lidar_box.tensor,
                                         Box3DMode.LIDAR, Box3DMode.CAM,
                                         rt_mat)
    assert torch.allclose(lidar_to_cam_box, camera_boxes.tensor)

    # test numpy convert
    cam_to_lidar_box = Box3DMode.convert(camera_boxes.tensor.numpy(),
                                         Box3DMode.CAM, Box3DMode.LIDAR,
                                         rt_mat.inverse().numpy())
    assert np.allclose(cam_to_lidar_box, expected_tensor.numpy())

    # test list convert
    cam_to_lidar_box = Box3DMode.convert(
        camera_boxes.tensor[0].numpy().tolist(), Box3DMode.CAM,
        Box3DMode.LIDAR,
        rt_mat.inverse().numpy())
    assert np.allclose(np.array(cam_to_lidar_box), expected_tensor[0].numpy())

    # test convert from depth to lidar
    depth_boxes = torch.tensor(
        [[2.4593, 2.5870, -0.4321, 0.8597, 0.6193, 1.0204, 3.0693],
         [1.4856, 2.5299, -0.5570, 0.9385, 2.1404, 0.8954, 3.0601]],
        dtype=torch.float32)
    depth_boxes = DepthInstance3DBoxes(depth_boxes)
    depth_to_lidar_box = depth_boxes.convert_to(Box3DMode.LIDAR)
    expected_box = depth_to_lidar_box.convert_to(Box3DMode.DEPTH)
    assert torch.equal(depth_boxes.tensor, expected_box.tensor)

    lidar_to_depth_box = Box3DMode.convert(depth_to_lidar_box, Box3DMode.LIDAR,
                                           Box3DMode.DEPTH)
    assert torch.allclose(depth_boxes.tensor, lidar_to_depth_box.tensor)
    assert torch.allclose(depth_boxes.volume, lidar_to_depth_box.volume)

    # test convert from depth to camera
    depth_to_cam_box = Box3DMode.convert(depth_boxes, Box3DMode.DEPTH,
                                         Box3DMode.CAM)
    cam_to_depth_box = Box3DMode.convert(depth_to_cam_box, Box3DMode.CAM,
                                         Box3DMode.DEPTH)
    expected_tensor = depth_to_cam_box.convert_to(Box3DMode.DEPTH)
    assert torch.equal(expected_tensor.tensor, cam_to_depth_box.tensor)
    assert torch.allclose(depth_boxes.tensor, cam_to_depth_box.tensor)
    assert torch.allclose(depth_boxes.volume, cam_to_depth_box.volume)

    with pytest.raises(NotImplementedError):
        # assert invalid convert mode
        Box3DMode.convert(depth_boxes, Box3DMode.DEPTH, 3)


def test_camera_boxes3d():
    # Test init with numpy array
    np_boxes = np.array([[
        1.7802081, 2.516249, -1.7501148, 1.75, 3.39, 1.65,
        1.48 - 0.13603681398218053 * 4 - 2 * np.pi
    ],
                         [
                             8.959413, 2.4567227, -1.6357126, 1.54, 4.01, 1.57,
                             1.62 - 0.13603681398218053 * 4 - 2 * np.pi
                         ]],
                        dtype=np.float32)

    boxes_1 = Box3DMode.convert(
        LiDARInstance3DBoxes(np_boxes), Box3DMode.LIDAR, Box3DMode.CAM)
    assert isinstance(boxes_1, CameraInstance3DBoxes)

    cam_np_boxes = Box3DMode.convert(np_boxes, Box3DMode.LIDAR, Box3DMode.CAM)
    assert torch.allclose(boxes_1.tensor,
                          boxes_1.tensor.new_tensor(cam_np_boxes))

    # test init with torch.Tensor
    th_boxes = torch.tensor(
        [[
            28.29669987, -0.5557558, -1.30332506, 1.47000003, 2.23000002,
            1.48000002, -1.57000005 - 0.13603681398218053 * 4 - 2 * np.pi
        ],
         [
             26.66901946, 21.82302134, -1.73605708, 1.55999994, 3.48000002,
             1.39999998, -1.69000006 - 0.13603681398218053 * 4 - 2 * np.pi
         ],
         [
             31.31977974, 8.16214412, -1.62177875, 1.74000001, 3.76999998,
             1.48000002, 2.78999996 - 0.13603681398218053 * 4 - 2 * np.pi
         ]],
        dtype=torch.float32)
    cam_th_boxes = Box3DMode.convert(th_boxes, Box3DMode.LIDAR, Box3DMode.CAM)
    boxes_2 = CameraInstance3DBoxes(cam_th_boxes)
    assert torch.allclose(boxes_2.tensor, cam_th_boxes)

    # test clone/to/device
    boxes_2 = boxes_2.clone()
    boxes_1 = boxes_1.to(boxes_2.device)

    # test box concatenation
    expected_tensor = Box3DMode.convert(
        torch.tensor([[
            1.7802081, 2.516249, -1.7501148, 1.75, 3.39, 1.65,
            1.48 - 0.13603681398218053 * 4 - 2 * np.pi
        ],
                      [
                          8.959413, 2.4567227, -1.6357126, 1.54, 4.01, 1.57,
                          1.62 - 0.13603681398218053 * 4 - 2 * np.pi
                      ],
                      [
                          28.2967, -0.5557558, -1.303325, 1.47, 2.23, 1.48,
                          -1.57 - 0.13603681398218053 * 4 - 2 * np.pi
                      ],
                      [
                          26.66902, 21.82302, -1.736057, 1.56, 3.48, 1.4,
                          -1.69 - 0.13603681398218053 * 4 - 2 * np.pi
                      ],
                      [
                          31.31978, 8.162144, -1.6217787, 1.74, 3.77, 1.48,
                          2.79 - 0.13603681398218053 * 4 - 2 * np.pi
                      ]]), Box3DMode.LIDAR, Box3DMode.CAM)
    boxes = CameraInstance3DBoxes.cat([boxes_1, boxes_2])
    assert torch.allclose(boxes.tensor, expected_tensor)

    # test box flip
    points = torch.tensor([[0.6762, 1.4658, 1.2559], [0.8784, 1.3857, 4.7814],
                           [-0.2517, 0.9697, 6.7053], [0.5520, 0.5265, 0.6533],
                           [-0.5358, 1.4741, 4.5870]])
    expected_tensor = Box3DMode.convert(
        torch.tensor([[
            1.7802081, -2.516249, -1.7501148, 1.75, 3.39, 1.65,
            1.6615927 + 0.13603681398218053 * 4 - np.pi
        ],
                      [
                          8.959413, -2.4567227, -1.6357126, 1.54, 4.01, 1.57,
                          1.5215927 + 0.13603681398218053 * 4 - np.pi
                      ],
                      [
                          28.2967, 0.5557558, -1.303325, 1.47, 2.23, 1.48,
                          4.7115927 + 0.13603681398218053 * 4 - np.pi
                      ],
                      [
                          26.66902, -21.82302, -1.736057, 1.56, 3.48, 1.4,
                          4.8315926 + 0.13603681398218053 * 4 - np.pi
                      ],
                      [
                          31.31978, -8.162144, -1.6217787, 1.74, 3.77, 1.48,
                          0.35159278 + 0.13603681398218053 * 4 - np.pi
                      ]]), Box3DMode.LIDAR, Box3DMode.CAM)
    points = boxes.flip('horizontal', points)
    expected_points = torch.tensor([[-0.6762, 1.4658, 1.2559],
                                    [-0.8784, 1.3857, 4.7814],
                                    [0.2517, 0.9697, 6.7053],
                                    [-0.5520, 0.5265, 0.6533],
                                    [0.5358, 1.4741, 4.5870]])

    yaw_normalized_tensor = boxes.tensor.clone()
    yaw_normalized_tensor[:, -1:] = limit_period(
        yaw_normalized_tensor[:, -1:], period=np.pi * 2)
    assert torch.allclose(yaw_normalized_tensor, expected_tensor, 1e-3)
    assert torch.allclose(points, expected_points, 1e-3)

    expected_tensor = torch.tensor(
        [[
            2.5162, 1.7501, -1.7802, 1.7500, 1.6500, 3.3900,
            1.6616 + 0.13603681398218053 * 4 - np.pi / 2
        ],
         [
             2.4567, 1.6357, -8.9594, 1.5400, 1.5700, 4.0100,
             1.5216 + 0.13603681398218053 * 4 - np.pi / 2
         ],
         [
             -0.5558, 1.3033, -28.2967, 1.4700, 1.4800, 2.2300,
             4.7116 + 0.13603681398218053 * 4 - np.pi / 2
         ],
         [
             21.8230, 1.7361, -26.6690, 1.5600, 1.4000, 3.4800,
             4.8316 + 0.13603681398218053 * 4 - np.pi / 2
         ],
         [
             8.1621, 1.6218, -31.3198, 1.7400, 1.4800, 3.7700,
             0.3516 + 0.13603681398218053 * 4 - np.pi / 2
         ]])
    boxes_flip_vert = boxes.clone()
    points = boxes_flip_vert.flip('vertical', points)
    expected_points = torch.tensor([[-0.6762, 1.4658, -1.2559],
                                    [-0.8784, 1.3857, -4.7814],
                                    [0.2517, 0.9697, -6.7053],
                                    [-0.5520, 0.5265, -0.6533],
                                    [0.5358, 1.4741, -4.5870]])

    yaw_normalized_tensor = boxes_flip_vert.tensor.clone()
    yaw_normalized_tensor[:, -1:] = limit_period(
        yaw_normalized_tensor[:, -1:], period=np.pi * 2)
    expected_tensor[:, -1:] = limit_period(
        expected_tensor[:, -1:], period=np.pi * 2)
    assert torch.allclose(yaw_normalized_tensor, expected_tensor, 1e-4)
    assert torch.allclose(points, expected_points)

    # test box rotation
    # with input torch.Tensor points and angle
    expected_tensor = Box3DMode.convert(
        torch.tensor([[
            1.4225, -2.7344, -1.7501, 1.7500, 3.3900, 1.6500,
            1.7976 + 0.13603681398218053 * 2 - np.pi
        ],
                      [
                          8.5435, -3.6491, -1.6357, 1.5400, 4.0100, 1.5700,
                          1.6576 + 0.13603681398218053 * 2 - np.pi
                      ],
                      [
                          28.1106, -3.2869, -1.3033, 1.4700, 2.2300, 1.4800,
                          4.8476 + 0.13603681398218053 * 2 - np.pi
                      ],
                      [
                          23.4630, -25.2382, -1.7361, 1.5600, 3.4800, 1.4000,
                          4.9676 + 0.13603681398218053 * 2 - np.pi
                      ],
                      [
                          29.9235, -12.3342, -1.6218, 1.7400, 3.7700, 1.4800,
                          0.4876 + 0.13603681398218053 * 2 - np.pi
                      ]]), Box3DMode.LIDAR, Box3DMode.CAM)
    points, rot_mat_T = boxes.rotate(torch.tensor(0.13603681398218053), points)
    expected_points = torch.tensor([[-0.8403, 1.4658, -1.1526],
                                    [-1.5187, 1.3857, -4.6181],
                                    [-0.6600, 0.9697, -6.6775],
                                    [-0.6355, 0.5265, -0.5724],
                                    [-0.0912, 1.4741, -4.6173]])
    expected_rot_mat_T = torch.tensor([[0.9908, 0.0000, -0.1356],
                                       [0.0000, 1.0000, 0.0000],
                                       [0.1356, 0.0000, 0.9908]])
    yaw_normalized_tensor = boxes.tensor.clone()
    yaw_normalized_tensor[:, -1:] = limit_period(
        yaw_normalized_tensor[:, -1:], period=np.pi * 2)
    expected_tensor[:, -1:] = limit_period(
        expected_tensor[:, -1:], period=np.pi * 2)
    assert torch.allclose(yaw_normalized_tensor, expected_tensor, 1e-3)
    assert torch.allclose(points, expected_points, 1e-3)
    assert torch.allclose(rot_mat_T, expected_rot_mat_T, 1e-3)

    # with input torch.Tensor points and rotation matrix
    points, rot_mat_T = boxes.rotate(
        torch.tensor(-0.13603681398218053), points)  # back
    rot_mat = np.array([[0.99076125, 0., -0.13561762], [0., 1., 0.],
                        [0.13561762, 0., 0.99076125]])
    points, rot_mat_T = boxes.rotate(rot_mat, points)
    yaw_normalized_tensor = boxes.tensor.clone()
    yaw_normalized_tensor[:, -1:] = limit_period(
        yaw_normalized_tensor[:, -1:], period=np.pi * 2)
    assert torch.allclose(yaw_normalized_tensor, expected_tensor, 1e-3)
    assert torch.allclose(points, expected_points, 1e-3)
    assert torch.allclose(rot_mat_T, expected_rot_mat_T, 1e-3)

    # with input np.ndarray points and angle
    points_np = np.array([[0.6762, 1.2559, -1.4658, 2.5359],
                          [0.8784, 4.7814, -1.3857, 0.7167],
                          [-0.2517, 6.7053, -0.9697, 0.5599],
                          [0.5520, 0.6533, -0.5265, 1.0032],
                          [-0.5358, 4.5870, -1.4741, 0.0556]])
    points_np, rot_mat_T_np = boxes.rotate(
        torch.tensor(0.13603681398218053), points_np)
    expected_points_np = np.array([[0.4712, 1.2559, -1.5440, 2.5359],
                                   [0.6824, 4.7814, -1.4920, 0.7167],
                                   [-0.3809, 6.7053, -0.9266, 0.5599],
                                   [0.4755, 0.6533, -0.5965, 1.0032],
                                   [-0.7308, 4.5870, -1.3878, 0.0556]])
    expected_rot_mat_T_np = np.array([[0.9908, 0.0000, -0.1356],
                                      [0.0000, 1.0000, 0.0000],
                                      [0.1356, 0.0000, 0.9908]])

    assert np.allclose(points_np, expected_points_np, 1e-3)
    assert np.allclose(rot_mat_T_np, expected_rot_mat_T_np, 1e-3)

    # with input CameraPoints and rotation matrix
    points_np, rot_mat_T_np = boxes.rotate(
        torch.tensor(-0.13603681398218053), points_np)
    camera_points = CameraPoints(points_np, points_dim=4)
    camera_points, rot_mat_T_np = boxes.rotate(rot_mat, camera_points)
    points_np = camera_points.tensor.numpy()
    assert np.allclose(points_np, expected_points_np, 1e-3)
    assert np.allclose(rot_mat_T_np, expected_rot_mat_T_np, 1e-3)

    # test box scaling
    expected_tensor = Box3DMode.convert(
        torch.tensor([[
            1.0443488, -2.9183323, -1.7599131, 1.7597977, 3.4089797, 1.6592377,
            1.9336663 - np.pi
        ],
                      [
                          8.014273, -4.8007393, -1.6448704, 1.5486219,
                          4.0324507, 1.57879, 1.7936664 - np.pi
                      ],
                      [
                          27.558605, -7.1084175, -1.310622, 1.4782301,
                          2.242485, 1.488286, 4.9836664 - np.pi
                      ],
                      [
                          19.934517, -28.344835, -1.7457767, 1.5687338,
                          3.4994833, 1.4078381, 5.1036663 - np.pi
                      ],
                      [
                          28.130915, -16.369587, -1.6308585, 1.7497417,
                          3.791107, 1.488286, 0.6236664 - np.pi
                      ]]), Box3DMode.LIDAR, Box3DMode.CAM)
    boxes.scale(1.00559866335275)
    yaw_normalized_tensor = boxes.tensor.clone()
    yaw_normalized_tensor[:, -1:] = limit_period(
        yaw_normalized_tensor[:, -1:], period=np.pi * 2)
    expected_tensor[:, -1:] = limit_period(
        expected_tensor[:, -1:], period=np.pi * 2)
    assert torch.allclose(yaw_normalized_tensor, expected_tensor)

    # test box translation
    expected_tensor = Box3DMode.convert(
        torch.tensor([[
            1.1281544, -3.0507944, -1.9169292, 1.7597977, 3.4089797, 1.6592377,
            1.9336663 - np.pi
        ],
                      [
                          8.098079, -4.9332013, -1.8018866, 1.5486219,
                          4.0324507, 1.57879, 1.7936664 - np.pi
                      ],
                      [
                          27.64241, -7.2408795, -1.4676381, 1.4782301,
                          2.242485, 1.488286, 4.9836664 - np.pi
                      ],
                      [
                          20.018322, -28.477297, -1.9027928, 1.5687338,
                          3.4994833, 1.4078381, 5.1036663 - np.pi
                      ],
                      [
                          28.21472, -16.502048, -1.7878747, 1.7497417,
                          3.791107, 1.488286, 0.6236664 - np.pi
                      ]]), Box3DMode.LIDAR, Box3DMode.CAM)
    boxes.translate(torch.tensor([0.13246193, 0.15701613, 0.0838056]))
    yaw_normalized_tensor = boxes.tensor.clone()
    yaw_normalized_tensor[:, -1:] = limit_period(
        yaw_normalized_tensor[:, -1:], period=np.pi * 2)
    expected_tensor[:, -1:] = limit_period(
        expected_tensor[:, -1:], period=np.pi * 2)
    assert torch.allclose(yaw_normalized_tensor, expected_tensor)

    # test bbox in_range_bev
    expected_tensor = torch.tensor([1, 1, 1, 1, 1], dtype=torch.bool)
    mask = boxes.in_range_bev([0., -40., 70.4, 40.])
    assert (mask == expected_tensor).all()
    mask = boxes.nonempty()
    assert (mask == expected_tensor).all()

    # test bbox in_range
    expected_tensor = torch.tensor([1, 1, 0, 0, 0], dtype=torch.bool)
    mask = boxes.in_range_3d([-2, -5, 0, 20, 2, 22])
    assert (mask == expected_tensor).all()

    expected_tensor = torch.tensor(
        [[3.0508, 1.1282, 1.7598, 3.4090, -5.9203],
         [4.9332, 8.0981, 1.5486, 4.0325, -6.0603],
         [7.2409, 27.6424, 1.4782, 2.2425, -2.8703],
         [28.4773, 20.0183, 1.5687, 3.4995, -2.7503],
         [16.5020, 28.2147, 1.7497, 3.7911, -0.9471]])
    assert torch.allclose(boxes.bev, expected_tensor, atol=1e-3)

    # test properties
    assert torch.allclose(boxes.bottom_center, boxes.tensor[:, :3])
    expected_tensor = (
        boxes.tensor[:, :3] - boxes.tensor[:, 3:6] *
        (torch.tensor([0.5, 1.0, 0.5]) - torch.tensor([0.5, 0.5, 0.5])))
    assert torch.allclose(boxes.gravity_center, expected_tensor)

    boxes.limit_yaw()
    assert (boxes.tensor[:, 6] <= np.pi / 2).all()
    assert (boxes.tensor[:, 6] >= -np.pi / 2).all()

    Box3DMode.convert(boxes, Box3DMode.LIDAR, Box3DMode.LIDAR)
    expected_tensor = boxes.tensor.clone()
    assert torch.allclose(expected_tensor, boxes.tensor)

    boxes.flip()
    boxes.flip()
    boxes.limit_yaw()
    assert torch.allclose(expected_tensor, boxes.tensor)

    # test nearest_bev
    # BEV box in lidar coordinates (x, y)
    lidar_expected_tensor = torch.tensor(
        [[-0.5763, -3.9307, 2.8326, -2.1709],
         [6.0819, -5.7075, 10.1143, -4.1589],
         [26.5212, -7.9800, 28.7637, -6.5018],
         [18.2686, -29.2617, 21.7681, -27.6929],
         [27.3398, -18.3976, 29.0896, -14.6065]])
    # BEV box in camera coordinate (-y, x)
    expected_tensor = lidar_expected_tensor.clone()
    expected_tensor[:, 0::2] = -lidar_expected_tensor[:, [3, 1]]
    expected_tensor[:, 1::2] = lidar_expected_tensor[:, 0::2]
    assert torch.allclose(
        boxes.nearest_bev, expected_tensor, rtol=1e-4, atol=1e-7)

    expected_tensor = torch.tensor([[[2.8332e+00, 2.5769e-01, -7.7767e-01],
                                     [1.6232e+00, 2.5769e-01, 2.4093e+00],
                                     [1.6232e+00, 1.9169e+00, 2.4093e+00],
                                     [2.8332e+00, 1.9169e+00, -7.7767e-01],
                                     [4.4784e+00, 2.5769e-01, -1.5302e-01],
                                     [3.2684e+00, 2.5769e-01, 3.0340e+00],
                                     [3.2684e+00, 1.9169e+00, 3.0340e+00],
                                     [4.4784e+00, 1.9169e+00, -1.5302e-01]],
                                    [[4.6237e+00, 2.2310e-01, 5.9606e+00],
                                     [3.7324e+00, 2.2310e-01, 9.8933e+00],
                                     [3.7324e+00, 1.8019e+00, 9.8933e+00],
                                     [4.6237e+00, 1.8019e+00, 5.9606e+00],
                                     [6.1340e+00, 2.2310e-01, 6.3029e+00],
                                     [5.2427e+00, 2.2310e-01, 1.0236e+01],
                                     [5.2427e+00, 1.8019e+00, 1.0236e+01],
                                     [6.1340e+00, 1.8019e+00, 6.3029e+00]],
                                    [[6.8292e+00, -2.0648e-02, 2.6364e+01],
                                     [6.2283e+00, -2.0648e-02, 2.8525e+01],
                                     [6.2283e+00, 1.4676e+00, 2.8525e+01],
                                     [6.8292e+00, 1.4676e+00, 2.6364e+01],
                                     [8.2534e+00, -2.0648e-02, 2.6760e+01],
                                     [7.6525e+00, -2.0648e-02, 2.8921e+01],
                                     [7.6525e+00, 1.4676e+00, 2.8921e+01],
                                     [8.2534e+00, 1.4676e+00, 2.6760e+01]],
                                    [[2.8420e+01, 4.9495e-01, 1.8102e+01],
                                     [2.7085e+01, 4.9495e-01, 2.1337e+01],
                                     [2.7085e+01, 1.9028e+00, 2.1337e+01],
                                     [2.8420e+01, 1.9028e+00, 1.8102e+01],
                                     [2.9870e+01, 4.9495e-01, 1.8700e+01],
                                     [2.8535e+01, 4.9495e-01, 2.1935e+01],
                                     [2.8535e+01, 1.9028e+00, 2.1935e+01],
                                     [2.9870e+01, 1.9028e+00, 1.8700e+01]],
                                    [[1.4452e+01, 2.9959e-01, 2.7818e+01],
                                     [1.7530e+01, 2.9959e-01, 3.0032e+01],
                                     [1.7530e+01, 1.7879e+00, 3.0032e+01],
                                     [1.4452e+01, 1.7879e+00, 2.7818e+01],
                                     [1.5474e+01, 2.9959e-01, 2.6398e+01],
                                     [1.8552e+01, 2.9959e-01, 2.8612e+01],
                                     [1.8552e+01, 1.7879e+00, 2.8612e+01],
                                     [1.5474e+01, 1.7879e+00, 2.6398e+01]]])

    assert torch.allclose(boxes.corners, expected_tensor, rtol=1e-3, atol=1e-4)

    th_boxes = torch.tensor(
        [[
            28.29669987, -0.5557558, -1.30332506, 1.47000003, 2.23000002,
            1.48000002, -1.57000005
        ],
         [
             26.66901946, 21.82302134, -1.73605708, 1.55999994, 3.48000002,
             1.39999998, -1.69000006
         ],
         [
             31.31977974, 8.16214412, -1.62177875, 1.74000001, 3.76999998,
             1.48000002, 2.78999996
         ]],
        dtype=torch.float32)

    # test init with a given origin
    boxes_origin_given = CameraInstance3DBoxes(
        th_boxes.clone(), box_dim=7, origin=(0.5, 0.5, 0.5))
    expected_tensor = th_boxes.clone()
    expected_tensor[:, :3] = th_boxes[:, :3] + th_boxes[:, 3:6] * (
        th_boxes.new_tensor((0.5, 1.0, 0.5)) - th_boxes.new_tensor(
            (0.5, 0.5, 0.5)))
    assert torch.allclose(boxes_origin_given.tensor, expected_tensor)


def test_boxes3d_overlaps():
    """Test the iou calculation of boxes in different modes.

    CommandLine:
        xdoctest tests/test_box3d.py::test_boxes3d_overlaps zero
    """
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')

    # Test LiDAR boxes 3D overlaps
    boxes1_tensor = torch.tensor(
        [[1.8, -2.5, -1.8, 1.75, 3.39, 1.65, -1.6615927],
         [8.9, -2.5, -1.6, 1.54, 4.01, 1.57, -1.5215927],
         [28.3, 0.5, -1.3, 1.47, 2.23, 1.48, -4.7115927],
         [31.3, -8.2, -1.6, 1.74, 3.77, 1.48, -0.35]],
        device='cuda')
    boxes1 = LiDARInstance3DBoxes(boxes1_tensor)

    boxes2_tensor = torch.tensor([[1.2, -3.0, -1.9, 1.8, 3.4, 1.7, -1.9],
                                  [8.1, -2.9, -1.8, 1.5, 4.1, 1.6, -1.8],
                                  [31.3, -8.2, -1.6, 1.74, 3.77, 1.48, -0.35],
                                  [20.1, -28.5, -1.9, 1.6, 3.5, 1.4, -5.1]],
                                 device='cuda')
    boxes2 = LiDARInstance3DBoxes(boxes2_tensor)

    expected_iou_tensor = torch.tensor(
        [[0.3710, 0.0000, 0.0000, 0.0000], [0.0000, 0.3322, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 1.0000, 0.0000]],
        device='cuda')
    overlaps_3d_iou = boxes1.overlaps(boxes1, boxes2)
    assert torch.allclose(
        expected_iou_tensor, overlaps_3d_iou, rtol=1e-4, atol=1e-7)

    expected_iof_tensor = torch.tensor(
        [[0.5582, 0.0000, 0.0000, 0.0000], [0.0000, 0.5025, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 1.0000, 0.0000]],
        device='cuda')
    overlaps_3d_iof = boxes1.overlaps(boxes1, boxes2, mode='iof')
    assert torch.allclose(
        expected_iof_tensor, overlaps_3d_iof, rtol=1e-4, atol=1e-7)

    empty_boxes = []
    boxes3 = LiDARInstance3DBoxes(empty_boxes)
    overlaps_3d_empty = boxes1.overlaps(boxes3, boxes2)
    assert overlaps_3d_empty.shape[0] == 0
    assert overlaps_3d_empty.shape[1] == 4
    # Test camera boxes 3D overlaps
    cam_boxes1_tensor = Box3DMode.convert(boxes1_tensor, Box3DMode.LIDAR,
                                          Box3DMode.CAM)
    cam_boxes1 = CameraInstance3DBoxes(cam_boxes1_tensor)

    cam_boxes2_tensor = Box3DMode.convert(boxes2_tensor, Box3DMode.LIDAR,
                                          Box3DMode.CAM)
    cam_boxes2 = CameraInstance3DBoxes(cam_boxes2_tensor)
    cam_overlaps_3d = cam_boxes1.overlaps(cam_boxes1, cam_boxes2)

    # same boxes under different coordinates should have the same iou
    assert torch.allclose(
        expected_iou_tensor, cam_overlaps_3d, rtol=1e-3, atol=1e-4)
    assert torch.allclose(
        cam_overlaps_3d, overlaps_3d_iou, rtol=1e-3, atol=1e-4)

    with pytest.raises(AssertionError):
        cam_boxes1.overlaps(cam_boxes1, boxes1)
    with pytest.raises(AssertionError):
        boxes1.overlaps(cam_boxes1, boxes1)


def test_depth_boxes3d():
    # test empty initialization
    empty_boxes = []
    boxes = DepthInstance3DBoxes(empty_boxes)
    assert boxes.tensor.shape[0] == 0
    assert boxes.tensor.shape[1] == 7

    # Test init with numpy array
    np_boxes = np.array(
        [[1.4856, 2.5299, -0.5570, 0.9385, 2.1404, 0.8954, 3.0601],
         [2.3262, 3.3065, --0.44255, 0.8234, 0.5325, 1.0099, 2.9971]],
        dtype=np.float32)
    boxes_1 = DepthInstance3DBoxes(np_boxes)
    assert torch.allclose(boxes_1.tensor, torch.from_numpy(np_boxes))

    # test properties

    assert boxes_1.volume.size(0) == 2
    assert (boxes_1.center == boxes_1.bottom_center).all()
    expected_tensor = torch.tensor([[1.4856, 2.5299, -0.1093],
                                    [2.3262, 3.3065, 0.9475]])
    assert torch.allclose(boxes_1.gravity_center, expected_tensor)
    expected_tensor = torch.tensor([[1.4856, 2.5299, 0.9385, 2.1404, 3.0601],
                                    [2.3262, 3.3065, 0.8234, 0.5325, 2.9971]])
    assert torch.allclose(boxes_1.bev, expected_tensor)
    expected_tensor = torch.tensor([[1.0164, 1.4597, 1.9548, 3.6001],
                                    [1.9145, 3.0402, 2.7379, 3.5728]])
    assert torch.allclose(boxes_1.nearest_bev, expected_tensor, 1e-4)
    assert repr(boxes) == (
        'DepthInstance3DBoxes(\n    tensor([], size=(0, 7)))')

    # test init with torch.Tensor
    th_boxes = torch.tensor(
        [[2.4593, 2.5870, -0.4321, 0.8597, 0.6193, 1.0204, 3.0693],
         [1.4856, 2.5299, -0.5570, 0.9385, 2.1404, 0.8954, 3.0601]],
        dtype=torch.float32)
    boxes_2 = DepthInstance3DBoxes(th_boxes)
    assert torch.allclose(boxes_2.tensor, th_boxes)

    # test clone/to/device
    boxes_2 = boxes_2.clone()
    boxes_1 = boxes_1.to(boxes_2.device)

    # test box concatenation
    expected_tensor = torch.tensor(
        [[1.4856, 2.5299, -0.5570, 0.9385, 2.1404, 0.8954, 3.0601],
         [2.3262, 3.3065, 0.44255, 0.8234, 0.5325, 1.0099, 2.9971],
         [2.4593, 2.5870, -0.4321, 0.8597, 0.6193, 1.0204, 3.0693],
         [1.4856, 2.5299, -0.5570, 0.9385, 2.1404, 0.8954, 3.0601]])
    boxes = DepthInstance3DBoxes.cat([boxes_1, boxes_2])
    assert torch.allclose(boxes.tensor, expected_tensor)
    # concatenate empty list
    empty_boxes = DepthInstance3DBoxes.cat([])
    assert empty_boxes.tensor.shape[0] == 0
    assert empty_boxes.tensor.shape[-1] == 7

    # test box flip
    points = torch.tensor([[0.6762, 1.2559, -1.4658, 2.5359],
                           [0.8784, 4.7814, -1.3857, 0.7167],
                           [-0.2517, 6.7053, -0.9697, 0.5599],
                           [0.5520, 0.6533, -0.5265, 1.0032],
                           [-0.5358, 4.5870, -1.4741, 0.0556]])
    expected_tensor = torch.tensor(
        [[-1.4856, 2.5299, -0.5570, 0.9385, 2.1404, 0.8954, 0.0815],
         [-2.3262, 3.3065, 0.4426, 0.8234, 0.5325, 1.0099, 0.1445],
         [-2.4593, 2.5870, -0.4321, 0.8597, 0.6193, 1.0204, 0.0723],
         [-1.4856, 2.5299, -0.5570, 0.9385, 2.1404, 0.8954, 0.0815]])
    points = boxes.flip(bev_direction='horizontal', points=points)
    expected_points = torch.tensor([[-0.6762, 1.2559, -1.4658, 2.5359],
                                    [-0.8784, 4.7814, -1.3857, 0.7167],
                                    [0.2517, 6.7053, -0.9697, 0.5599],
                                    [-0.5520, 0.6533, -0.5265, 1.0032],
                                    [0.5358, 4.5870, -1.4741, 0.0556]])
    assert torch.allclose(boxes.tensor, expected_tensor, 1e-3)
    assert torch.allclose(points, expected_points)
    expected_tensor = torch.tensor(
        [[-1.4856, -2.5299, -0.5570, 0.9385, 2.1404, 0.8954, -0.0815],
         [-2.3262, -3.3065, 0.4426, 0.8234, 0.5325, 1.0099, -0.1445],
         [-2.4593, -2.5870, -0.4321, 0.8597, 0.6193, 1.0204, -0.0723],
         [-1.4856, -2.5299, -0.5570, 0.9385, 2.1404, 0.8954, -0.0815]])
    points = boxes.flip(bev_direction='vertical', points=points)
    expected_points = torch.tensor([[-0.6762, -1.2559, -1.4658, 2.5359],
                                    [-0.8784, -4.7814, -1.3857, 0.7167],
                                    [0.2517, -6.7053, -0.9697, 0.5599],
                                    [-0.5520, -0.6533, -0.5265, 1.0032],
                                    [0.5358, -4.5870, -1.4741, 0.0556]])
    assert torch.allclose(boxes.tensor, expected_tensor, 1e-3)
    assert torch.allclose(points, expected_points)

    # test box rotation
    # with input torch.Tensor points and angle
    boxes_rot = boxes.clone()
    expected_tensor = torch.tensor(
        [[-1.5434, -2.4951, -0.5570, 0.9385, 2.1404, 0.8954, -0.0585],
         [-2.4016, -3.2521, 0.4426, 0.8234, 0.5325, 1.0099, -0.1215],
         [-2.5181, -2.5298, -0.4321, 0.8597, 0.6193, 1.0204, -0.0493],
         [-1.5434, -2.4951, -0.5570, 0.9385, 2.1404, 0.8954, -0.0585]])
    expected_tensor[:, -1:] -= 0.022998953275003075 * 2
    points, rot_mat_T = boxes_rot.rotate(-0.022998953275003075, points)
    expected_points = torch.tensor([[-0.7049, -1.2400, -1.4658, 2.5359],
                                    [-0.9881, -4.7599, -1.3857, 0.7167],
                                    [0.0974, -6.7093, -0.9697, 0.5599],
                                    [-0.5669, -0.6404, -0.5265, 1.0032],
                                    [0.4302, -4.5981, -1.4741, 0.0556]])
    expected_rot_mat_T = torch.tensor([[0.9997, -0.0230, 0.0000],
                                       [0.0230, 0.9997, 0.0000],
                                       [0.0000, 0.0000, 1.0000]])
    assert torch.allclose(boxes_rot.tensor, expected_tensor, 1e-3)
    assert torch.allclose(points, expected_points, 1e-3)
    assert torch.allclose(rot_mat_T, expected_rot_mat_T, 1e-3)

    # with input torch.Tensor points and rotation matrix
    points, rot_mat_T = boxes.rotate(-0.022998953275003075, points)  # back
    rot_mat = np.array([[0.99973554, 0.02299693, 0.],
                        [-0.02299693, 0.99973554, 0.], [0., 0., 1.]])
    points, rot_mat_T = boxes.rotate(rot_mat, points)
    expected_rot_mat_T = torch.tensor([[0.99973554, 0.02299693, 0.0000],
                                       [-0.02299693, 0.99973554, 0.0000],
                                       [0.0000, 0.0000, 1.0000]])
    assert torch.allclose(boxes_rot.tensor, expected_tensor, 1e-3)
    assert torch.allclose(points, expected_points, 1e-3)
    assert torch.allclose(rot_mat_T, expected_rot_mat_T, 1e-3)

    # with input np.ndarray points and angle
    points_np = np.array([[0.6762, 1.2559, -1.4658, 2.5359],
                          [0.8784, 4.7814, -1.3857, 0.7167],
                          [-0.2517, 6.7053, -0.9697, 0.5599],
                          [0.5520, 0.6533, -0.5265, 1.0032],
                          [-0.5358, 4.5870, -1.4741, 0.0556]])
    points_np, rot_mat_T_np = boxes.rotate(-0.022998953275003075, points_np)
    expected_points_np = np.array([[0.7049, 1.2400, -1.4658, 2.5359],
                                   [0.9881, 4.7599, -1.3857, 0.7167],
                                   [-0.0974, 6.7093, -0.9697, 0.5599],
                                   [0.5669, 0.6404, -0.5265, 1.0032],
                                   [-0.4302, 4.5981, -1.4741, 0.0556]])
    expected_rot_mat_T_np = np.array([[0.99973554, -0.02299693, 0.0000],
                                      [0.02299693, 0.99973554, 0.0000],
                                      [0.0000, 0.0000, 1.0000]])
    expected_tensor = torch.tensor(
        [[-1.5434, -2.4951, -0.5570, 0.9385, 2.1404, 0.8954, -0.0585],
         [-2.4016, -3.2521, 0.4426, 0.8234, 0.5325, 1.0099, -0.1215],
         [-2.5181, -2.5298, -0.4321, 0.8597, 0.6193, 1.0204, -0.0493],
         [-1.5434, -2.4951, -0.5570, 0.9385, 2.1404, 0.8954, -0.0585]])
    expected_tensor[:, -1:] -= 0.022998953275003075 * 2
    assert torch.allclose(boxes.tensor, expected_tensor, 1e-3)
    assert np.allclose(points_np, expected_points_np, 1e-3)
    assert np.allclose(rot_mat_T_np, expected_rot_mat_T_np, 1e-3)

    # with input DepthPoints and rotation matrix
    points_np, rot_mat_T_np = boxes.rotate(-0.022998953275003075, points_np)
    depth_points = DepthPoints(points_np, points_dim=4)
    depth_points, rot_mat_T_np = boxes.rotate(rot_mat, depth_points)
    points_np = depth_points.tensor.numpy()
    expected_rot_mat_T_np = expected_rot_mat_T_np.T
    assert torch.allclose(boxes.tensor, expected_tensor, 1e-3)
    assert np.allclose(points_np, expected_points_np, 1e-3)
    assert np.allclose(rot_mat_T_np, expected_rot_mat_T_np, 1e-3)

    expected_tensor = torch.tensor([[[-2.1217, -3.5105, -0.5570],
                                     [-2.1217, -3.5105, 0.3384],
                                     [-1.8985, -1.3818, 0.3384],
                                     [-1.8985, -1.3818, -0.5570],
                                     [-1.1883, -3.6084, -0.5570],
                                     [-1.1883, -3.6084, 0.3384],
                                     [-0.9651, -1.4796, 0.3384],
                                     [-0.9651, -1.4796, -0.5570]],
                                    [[-2.8519, -3.4460, 0.4426],
                                     [-2.8519, -3.4460, 1.4525],
                                     [-2.7632, -2.9210, 1.4525],
                                     [-2.7632, -2.9210, 0.4426],
                                     [-2.0401, -3.5833, 0.4426],
                                     [-2.0401, -3.5833, 1.4525],
                                     [-1.9513, -3.0582, 1.4525],
                                     [-1.9513, -3.0582, 0.4426]],
                                    [[-2.9755, -2.7971, -0.4321],
                                     [-2.9755, -2.7971, 0.5883],
                                     [-2.9166, -2.1806, 0.5883],
                                     [-2.9166, -2.1806, -0.4321],
                                     [-2.1197, -2.8789, -0.4321],
                                     [-2.1197, -2.8789, 0.5883],
                                     [-2.0608, -2.2624, 0.5883],
                                     [-2.0608, -2.2624, -0.4321]],
                                    [[-2.1217, -3.5105, -0.5570],
                                     [-2.1217, -3.5105, 0.3384],
                                     [-1.8985, -1.3818, 0.3384],
                                     [-1.8985, -1.3818, -0.5570],
                                     [-1.1883, -3.6084, -0.5570],
                                     [-1.1883, -3.6084, 0.3384],
                                     [-0.9651, -1.4796, 0.3384],
                                     [-0.9651, -1.4796, -0.5570]]])

    assert torch.allclose(boxes.corners, expected_tensor, 1e-3)

    th_boxes = torch.tensor(
        [[0.61211395, 0.8129094, 0.10563634, 1.497534, 0.16927195, 0.27956772],
         [1.430009, 0.49797538, 0.9382923, 0.07694054, 0.9312509, 1.8919173]],
        dtype=torch.float32)
    boxes = DepthInstance3DBoxes(th_boxes, box_dim=6, with_yaw=False)
    expected_tensor = torch.tensor([[
        0.64884546, 0.78390356, 0.10563634, 1.50373348, 0.23795205, 0.27956772,
        0
    ],
                                    [
                                        1.45139421, 0.43169443, 0.93829232,
                                        0.11967964, 0.93380373, 1.89191735, 0
                                    ]])
    boxes_3 = boxes.clone()
    boxes_3.rotate(-0.04599790655000615)
    assert torch.allclose(boxes_3.tensor, expected_tensor)
    boxes.rotate(torch.tensor(-0.04599790655000615))
    assert torch.allclose(boxes.tensor, expected_tensor)

    # test bbox in_range_bev
    expected_tensor = torch.tensor([1, 1], dtype=torch.bool)
    mask = boxes.in_range_bev([0., -40., 70.4, 40.])
    assert (mask == expected_tensor).all()
    mask = boxes.nonempty()
    assert (mask == expected_tensor).all()

    # test bbox in_range
    expected_tensor = torch.tensor([0, 1], dtype=torch.bool)
    mask = boxes.in_range_3d([1, 0, -2, 2, 1, 5])
    assert (mask == expected_tensor).all()

    expected_tensor = torch.tensor([[[-0.1030, 0.6649, 0.1056],
                                     [-0.1030, 0.6649, 0.3852],
                                     [-0.1030, 0.9029, 0.3852],
                                     [-0.1030, 0.9029, 0.1056],
                                     [1.4007, 0.6649, 0.1056],
                                     [1.4007, 0.6649, 0.3852],
                                     [1.4007, 0.9029, 0.3852],
                                     [1.4007, 0.9029, 0.1056]],
                                    [[1.3916, -0.0352, 0.9383],
                                     [1.3916, -0.0352, 2.8302],
                                     [1.3916, 0.8986, 2.8302],
                                     [1.3916, 0.8986, 0.9383],
                                     [1.5112, -0.0352, 0.9383],
                                     [1.5112, -0.0352, 2.8302],
                                     [1.5112, 0.8986, 2.8302],
                                     [1.5112, 0.8986, 0.9383]]])
    assert torch.allclose(boxes.corners, expected_tensor, 1e-3)

    # test points in boxes
    if torch.cuda.is_available():
        box_idxs_of_pts = boxes.points_in_boxes_all(points.cuda())
        expected_idxs_of_pts = torch.tensor(
            [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
            device='cuda:0',
            dtype=torch.int32)
        assert torch.all(box_idxs_of_pts == expected_idxs_of_pts)

    # test get_surface_line_center
    boxes = torch.tensor(
        [[0.3294, 1.0359, 0.1171, 1.0822, 1.1247, 1.3721, -0.4916],
         [-2.4630, -2.6324, -0.1616, 0.9202, 1.7896, 0.1992, -0.3185]])
    boxes = DepthInstance3DBoxes(
        boxes, box_dim=boxes.shape[-1], with_yaw=True, origin=(0.5, 0.5, 0.5))
    surface_center, line_center = boxes.get_surface_line_center()

    expected_surface_center = torch.tensor([[0.3294, 1.0359, 0.8031],
                                            [0.3294, 1.0359, -0.5689],
                                            [0.5949, 1.5317, 0.1171],
                                            [0.1533, 0.5018, 0.1171],
                                            [0.8064, 0.7805, 0.1171],
                                            [-0.1845, 1.2053, 0.1171],
                                            [-2.4630, -2.6324, -0.0620],
                                            [-2.4630, -2.6324, -0.2612],
                                            [-2.0406, -1.8436, -0.1616],
                                            [-2.7432, -3.4822, -0.1616],
                                            [-2.0574, -2.8496, -0.1616],
                                            [-2.9000, -2.4883, -0.1616]])

    expected_line_center = torch.tensor([[0.8064, 0.7805, 0.8031],
                                         [-0.1845, 1.2053, 0.8031],
                                         [0.5949, 1.5317, 0.8031],
                                         [0.1533, 0.5018, 0.8031],
                                         [0.8064, 0.7805, -0.5689],
                                         [-0.1845, 1.2053, -0.5689],
                                         [0.5949, 1.5317, -0.5689],
                                         [0.1533, 0.5018, -0.5689],
                                         [1.0719, 1.2762, 0.1171],
                                         [0.6672, 0.3324, 0.1171],
                                         [0.1178, 1.7871, 0.1171],
                                         [-0.3606, 0.6713, 0.1171],
                                         [-2.0574, -2.8496, -0.0620],
                                         [-2.9000, -2.4883, -0.0620],
                                         [-2.0406, -1.8436, -0.0620],
                                         [-2.7432, -3.4822, -0.0620],
                                         [-2.0574, -2.8496, -0.2612],
                                         [-2.9000, -2.4883, -0.2612],
                                         [-2.0406, -1.8436, -0.2612],
                                         [-2.7432, -3.4822, -0.2612],
                                         [-1.6350, -2.0607, -0.1616],
                                         [-2.3062, -3.6263, -0.1616],
                                         [-2.4462, -1.6264, -0.1616],
                                         [-3.1802, -3.3381, -0.1616]])

    assert torch.allclose(surface_center, expected_surface_center, atol=1e-04)
    assert torch.allclose(line_center, expected_line_center, atol=1e-04)


def test_rotation_3d_in_axis():
    # clockwise
    points = torch.tensor([[[-0.4599, -0.0471, 0.0000],
                            [-0.4599, -0.0471, 1.8433],
                            [-0.4599, 0.0471, 1.8433]],
                           [[-0.2555, -0.2683, 0.0000],
                            [-0.2555, -0.2683, 0.9072],
                            [-0.2555, 0.2683, 0.9072]]])
    rotated = rotation_3d_in_axis(
        points,
        torch.tensor([-np.pi / 10, np.pi / 10]),
        axis=0,
        clockwise=True)
    expected_rotated = torch.tensor(
        [[[-0.4599, -0.0448, -0.0146], [-0.4599, -0.6144, 1.7385],
          [-0.4599, -0.5248, 1.7676]],
         [[-0.2555, -0.2552, 0.0829], [-0.2555, 0.0252, 0.9457],
          [-0.2555, 0.5355, 0.7799]]],
        dtype=torch.float32)
    assert torch.allclose(rotated, expected_rotated, atol=1e-3)

    # anti-clockwise with return rotation mat
    points = torch.tensor([[[-0.4599, -0.0471, 0.0000],
                            [-0.4599, -0.0471, 1.8433]]])
    rotated = rotation_3d_in_axis(points, torch.tensor([np.pi / 2]), axis=0)
    expected_rotated = torch.tensor([[[-0.4599, 0.0000, -0.0471],
                                      [-0.4599, -1.8433, -0.0471]]])
    assert torch.allclose(rotated, expected_rotated, 1e-3)

    points = torch.tensor([[[-0.4599, -0.0471, 0.0000],
                            [-0.4599, -0.0471, 1.8433]]])
    rotated, mat = rotation_3d_in_axis(
        points, torch.tensor([np.pi / 2]), axis=0, return_mat=True)
    expected_rotated = torch.tensor([[[-0.4599, 0.0000, -0.0471],
                                      [-0.4599, -1.8433, -0.0471]]])
    expected_mat = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0]]]).float()
    assert torch.allclose(rotated, expected_rotated, atol=1e-6)
    assert torch.allclose(mat, expected_mat, atol=1e-6)

    points = torch.tensor([[[-0.4599, -0.0471, 0.0000],
                            [-0.4599, -0.0471, 1.8433]],
                           [[-0.2555, -0.2683, 0.0000],
                            [-0.2555, -0.2683, 0.9072]]])
    rotated = rotation_3d_in_axis(points, np.pi / 2, axis=0)
    expected_rotated = torch.tensor([[[-0.4599, 0.0000, -0.0471],
                                      [-0.4599, -1.8433, -0.0471]],
                                     [[-0.2555, 0.0000, -0.2683],
                                      [-0.2555, -0.9072, -0.2683]]])
    assert torch.allclose(rotated, expected_rotated, atol=1e-3)

    points = np.array([[[-0.4599, -0.0471, 0.0000], [-0.4599, -0.0471,
                                                     1.8433]],
                       [[-0.2555, -0.2683, 0.0000],
                        [-0.2555, -0.2683, 0.9072]]]).astype(np.float32)

    rotated = rotation_3d_in_axis(points, np.pi / 2, axis=0)
    expected_rotated = np.array([[[-0.4599, 0.0000, -0.0471],
                                  [-0.4599, -1.8433, -0.0471]],
                                 [[-0.2555, 0.0000, -0.2683],
                                  [-0.2555, -0.9072, -0.2683]]])
    assert np.allclose(rotated, expected_rotated, atol=1e-3)

    points = torch.tensor([[[-0.4599, -0.0471, 0.0000],
                            [-0.4599, -0.0471, 1.8433]],
                           [[-0.2555, -0.2683, 0.0000],
                            [-0.2555, -0.2683, 0.9072]]])
    angles = [np.pi / 2, -np.pi / 2]
    rotated = rotation_3d_in_axis(points, angles, axis=0).numpy()
    expected_rotated = np.array([[[-0.4599, 0.0000, -0.0471],
                                  [-0.4599, -1.8433, -0.0471]],
                                 [[-0.2555, 0.0000, 0.2683],
                                  [-0.2555, 0.9072, 0.2683]]])
    assert np.allclose(rotated, expected_rotated, atol=1e-3)

    points = torch.tensor([[[-0.4599, -0.0471, 0.0000],
                            [-0.4599, -0.0471, 1.8433]],
                           [[-0.2555, -0.2683, 0.0000],
                            [-0.2555, -0.2683, 0.9072]]])
    angles = [np.pi / 2, -np.pi / 2]
    rotated = rotation_3d_in_axis(points, angles, axis=1).numpy()
    expected_rotated = np.array([[[0.0000, -0.0471, 0.4599],
                                  [1.8433, -0.0471, 0.4599]],
                                 [[0.0000, -0.2683, -0.2555],
                                  [-0.9072, -0.2683, -0.2555]]])
    assert np.allclose(rotated, expected_rotated, atol=1e-3)

    points = torch.tensor([[[-0.4599, -0.0471, 0.0000],
                            [-0.4599, 0.0471, 1.8433]],
                           [[-0.2555, -0.2683, 0.0000],
                            [0.2555, -0.2683, 0.9072]]])
    angles = [np.pi / 2, -np.pi / 2]
    rotated = rotation_3d_in_axis(points, angles, axis=2).numpy()
    expected_rotated = np.array([[[0.0471, -0.4599, 0.0000],
                                  [-0.0471, -0.4599, 1.8433]],
                                 [[-0.2683, 0.2555, 0.0000],
                                  [-0.2683, -0.2555, 0.9072]]])
    assert np.allclose(rotated, expected_rotated, atol=1e-3)

    points = torch.tensor([[[-0.0471, 0.0000], [-0.0471, 1.8433]],
                           [[-0.2683, 0.0000], [-0.2683, 0.9072]]])
    angles = [np.pi / 2, -np.pi / 2]
    rotated = rotation_3d_in_axis(points, angles)
    expected_rotated = np.array([[[0.0000, -0.0471], [-1.8433, -0.0471]],
                                 [[0.0000, 0.2683], [0.9072, 0.2683]]])
    assert np.allclose(rotated, expected_rotated, atol=1e-3)


def test_rotation_2d():
    angles = np.array([3.14])
    corners = np.array([[[-0.235, -0.49], [-0.235, 0.49], [0.235, 0.49],
                         [0.235, -0.49]]])
    corners_rotated = rotation_3d_in_axis(corners, angles)
    expected_corners = np.array([[[0.2357801, 0.48962511],
                                  [0.2342193, -0.49037365],
                                  [-0.2357801, -0.48962511],
                                  [-0.2342193, 0.49037365]]])
    assert np.allclose(corners_rotated, expected_corners)


def test_limit_period():
    torch.manual_seed(0)
    val = torch.rand([5, 1])
    result = limit_period(val)
    expected_result = torch.tensor([[0.4963], [0.7682], [0.0885], [0.1320],
                                    [0.3074]])
    assert torch.allclose(result, expected_result, 1e-3)

    val = val.numpy()
    result = limit_period(val)
    expected_result = expected_result.numpy()
    assert np.allclose(result, expected_result, 1e-3)


def test_xywhr2xyxyr():
    torch.manual_seed(0)
    xywhr = torch.tensor([[1., 2., 3., 4., 5.], [0., 1., 2., 3., 4.]])
    xyxyr = xywhr2xyxyr(xywhr)
    expected_xyxyr = torch.tensor([[-0.5000, 0.0000, 2.5000, 4.0000, 5.0000],
                                   [-1.0000, -0.5000, 1.0000, 2.5000, 4.0000]])

    assert torch.allclose(xyxyr, expected_xyxyr)


class test_get_box_type(unittest.TestCase):

    def test_get_box_type(self):
        box_type_3d, box_mode_3d = get_box_type('camera')
        assert box_type_3d == CameraInstance3DBoxes
        assert box_mode_3d == Box3DMode.CAM

        box_type_3d, box_mode_3d = get_box_type('depth')
        assert box_type_3d == DepthInstance3DBoxes
        assert box_mode_3d == Box3DMode.DEPTH

        box_type_3d, box_mode_3d = get_box_type('lidar')
        assert box_type_3d == LiDARInstance3DBoxes
        assert box_mode_3d == Box3DMode.LIDAR

    def test_bad_box_type(self):
        self.assertRaises(ValueError, get_box_type, 'test')


def test_points_cam2img():
    torch.manual_seed(0)
    points = torch.rand([5, 3])
    proj_mat = torch.rand([4, 4])
    point_2d_res = points_cam2img(points, proj_mat)
    expected_point_2d_res = torch.tensor([[0.5832, 0.6496], [0.6146, 0.7910],
                                          [0.6994, 0.7782], [0.5623, 0.6303],
                                          [0.4359, 0.6532]])
    assert torch.allclose(point_2d_res, expected_point_2d_res, 1e-3)

    points = points.numpy()
    proj_mat = proj_mat.numpy()
    point_2d_res = points_cam2img(points, proj_mat)
    expected_point_2d_res = expected_point_2d_res.numpy()
    assert np.allclose(point_2d_res, expected_point_2d_res, 1e-3)

    points = torch.from_numpy(points)
    point_2d_res = points_cam2img(points, proj_mat)
    expected_point_2d_res = torch.from_numpy(expected_point_2d_res)
    assert torch.allclose(point_2d_res, expected_point_2d_res, 1e-3)

    point_2d_res = points_cam2img(points, proj_mat, with_depth=True)
    expected_point_2d_res = torch.tensor([[0.5832, 0.6496, 1.7577],
                                          [0.6146, 0.7910, 1.5477],
                                          [0.6994, 0.7782, 2.0091],
                                          [0.5623, 0.6303, 1.8739],
                                          [0.4359, 0.6532, 1.2056]])
    assert torch.allclose(point_2d_res, expected_point_2d_res, 1e-3)


def test_points_in_boxes():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    lidar_pts = torch.tensor([[1.0, 4.3, 0.1], [1.0, 4.4,
                                                0.1], [1.1, 4.3, 0.1],
                              [0.9, 4.3, 0.1], [1.0, -0.3, 0.1],
                              [1.0, -0.4, 0.1], [2.9, 0.1, 6.0],
                              [-0.9, 3.9, 6.0]]).cuda()
    lidar_boxes = torch.tensor([[1.0, 2.0, 0.0, 4.0, 4.0, 6.0, np.pi / 6],
                                [1.0, 2.0, 0.0, 4.0, 4.0, 6.0, np.pi / 2],
                                [1.0, 2.0, 0.0, 4.0, 4.0, 6.0, 7 * np.pi / 6],
                                [1.0, 2.0, 0.0, 4.0, 4.0, 6.0, -np.pi / 6]],
                               dtype=torch.float32).cuda()
    lidar_boxes = LiDARInstance3DBoxes(lidar_boxes)

    point_indices = lidar_boxes.points_in_boxes_all(lidar_pts)
    expected_point_indices = torch.tensor(
        [[1, 0, 1, 1], [0, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1], [1, 0, 1, 1],
         [0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
        dtype=torch.int32).cuda()
    assert point_indices.shape == torch.Size([8, 4])
    assert (point_indices == expected_point_indices).all()

    lidar_pts = torch.tensor([[1.0, 4.3, 0.1], [1.0, 4.4,
                                                0.1], [1.1, 4.3, 0.1],
                              [0.9, 4.3, 0.1], [1.0, -0.3, 0.1],
                              [1.0, -0.4, 0.1], [2.9, 0.1, 6.0],
                              [-0.9, 3.9, 6.0]]).cuda()
    lidar_boxes = torch.tensor([[1.0, 2.0, 0.0, 4.0, 4.0, 6.0, np.pi / 6],
                                [1.0, 2.0, 0.0, 4.0, 4.0, 6.0, np.pi / 2],
                                [1.0, 2.0, 0.0, 4.0, 4.0, 6.0, 7 * np.pi / 6],
                                [1.0, 2.0, 0.0, 4.0, 4.0, 6.0, -np.pi / 6]],
                               dtype=torch.float32).cuda()
    lidar_boxes = LiDARInstance3DBoxes(lidar_boxes)

    point_indices = lidar_boxes.points_in_boxes_part(lidar_pts)
    expected_point_indices = torch.tensor([0, -1, 0, 3, 0, -1, 1, 1],
                                          dtype=torch.int32).cuda()
    assert point_indices.shape == torch.Size([8])
    assert (point_indices == expected_point_indices).all()

    depth_boxes = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.3],
                                [-10.0, 23.0, 16.0, 10, 20, 20, 0.5]],
                               dtype=torch.float32).cuda()
    depth_boxes = DepthInstance3DBoxes(depth_boxes)
    depth_pts = torch.tensor(
        [[[1, 2, 3.3], [1.2, 2.5, 3.0], [0.8, 2.1, 3.5], [1.6, 2.6, 3.6],
          [0.8, 1.2, 3.9], [-9.2, 21.0, 18.2], [3.8, 7.9, 6.3],
          [4.7, 3.5, -12.2], [3.8, 7.6, -2], [-10.6, -12.9, -20], [
              -16, -18, 9
          ], [-21.3, -52, -5], [0, 0, 0], [6, 7, 8], [-2, -3, -4]]],
        dtype=torch.float32).cuda()

    point_indices = depth_boxes.points_in_boxes_all(depth_pts)
    expected_point_indices = torch.tensor(
        [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 0], [0, 0],
         [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
        dtype=torch.int32).cuda()
    assert point_indices.shape == torch.Size([15, 2])
    assert (point_indices == expected_point_indices).all()

    point_indices = depth_boxes.points_in_boxes_part(depth_pts)
    expected_point_indices = torch.tensor(
        [0, 0, 0, 0, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        dtype=torch.int32).cuda()
    assert point_indices.shape == torch.Size([15])
    assert (point_indices == expected_point_indices).all()

    depth_boxes = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.3],
                                [-10.0, 23.0, 16.0, 10, 20, 20, 0.5],
                                [1.0, 2.0, 0.0, 4.0, 4.0, 6.0, np.pi / 6],
                                [1.0, 2.0, 0.0, 4.0, 4.0, 6.0, np.pi / 2],
                                [1.0, 2.0, 0.0, 4.0, 4.0, 6.0, 7 * np.pi / 6],
                                [1.0, 2.0, 0.0, 4.0, 4.0, 6.0, -np.pi / 6]],
                               dtype=torch.float32).cuda()
    cam_boxes = DepthInstance3DBoxes(depth_boxes).convert_to(Box3DMode.CAM)
    depth_pts = torch.tensor(
        [[1, 2, 3.3], [1.2, 2.5, 3.0], [0.8, 2.1, 3.5], [1.6, 2.6, 3.6],
         [0.8, 1.2, 3.9], [-9.2, 21.0, 18.2], [3.8, 7.9, 6.3],
         [4.7, 3.5, -12.2], [3.8, 7.6, -2], [-10.6, -12.9, -20], [-16, -18, 9],
         [-21.3, -52, -5], [0, 0, 0], [6, 7, 8], [-2, -3, -4], [1.0, 4.3, 0.1],
         [1.0, 4.4, 0.1], [1.1, 4.3, 0.1], [0.9, 4.3, 0.1], [1.0, -0.3, 0.1],
         [1.0, -0.4, 0.1], [2.9, 0.1, 6.0], [-0.9, 3.9, 6.0]],
        dtype=torch.float32).cuda()

    cam_pts = DepthPoints(depth_pts).convert_to(Coord3DMode.CAM).tensor

    point_indices = cam_boxes.points_in_boxes_all(cam_pts)
    expected_point_indices = torch.tensor(
        [[1, 0, 1, 1, 1, 1], [1, 0, 1, 1, 1, 1], [1, 0, 1, 1, 1, 1],
         [1, 0, 1, 1, 1, 1], [1, 0, 1, 1, 1, 1], [0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1],
         [0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 0, 0],
         [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]],
        dtype=torch.int32).cuda()
    assert point_indices.shape == torch.Size([23, 6])
    assert (point_indices == expected_point_indices).all()

    point_indices = cam_boxes.points_in_boxes_batch(cam_pts)
    assert (point_indices == expected_point_indices).all()

    point_indices = cam_boxes.points_in_boxes_part(cam_pts)
    expected_point_indices = torch.tensor([
        0, 0, 0, 0, 0, 1, -1, -1, -1, -1, -1, -1, 3, -1, -1, 2, 3, 3, 2, 2, 3,
        0, 0
    ],
                                          dtype=torch.int32).cuda()
    assert point_indices.shape == torch.Size([23])
    assert (point_indices == expected_point_indices).all()

    point_indices = cam_boxes.points_in_boxes(cam_pts)
    assert (point_indices == expected_point_indices).all()
