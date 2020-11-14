import numpy as np
import torch

from mmdet3d.core.bbox import (CameraInstance3DBoxes, Coord3DMode,
                               DepthInstance3DBoxes, LiDARInstance3DBoxes)
from mmdet3d.core.points import CameraPoints, DepthPoints, LiDARPoints


def test_points_conversion():
    """Test the conversion of points between different modes."""
    points_np = np.array([[
        -5.24223238e+00, 4.00209696e+01, 2.97570381e-01, 0.6666, 0.1956,
        0.4974, 0.9409
    ],
                          [
                              -2.66751588e+01, 5.59499564e+00, -9.14345860e-01,
                              0.1502, 0.3707, 0.1086, 0.6297
                          ],
                          [
                              -5.80979675e+00, 3.54092357e+01, 2.00889888e-01,
                              0.6565, 0.6248, 0.6954, 0.2538
                          ],
                          [
                              -3.13086877e+01, 1.09007628e+00, -1.94612112e-01,
                              0.2803, 0.0258, 0.4896, 0.3269
                          ]],
                         dtype=np.float32)

    # test CAM to LIDAR and DEPTH
    cam_points = CameraPoints(
        points_np,
        points_dim=7,
        attribute_dims=dict(color=[3, 4, 5], height=6))

    convert_lidar_points = cam_points.convert_to(Coord3DMode.LIDAR)
    expected_tensor = torch.tensor([[
        2.9757e-01, 5.2422e+00, -4.0021e+01, 6.6660e-01, 1.9560e-01,
        4.9740e-01, 9.4090e-01
    ],
                                    [
                                        -9.1435e-01, 2.6675e+01, -5.5950e+00,
                                        1.5020e-01, 3.7070e-01, 1.0860e-01,
                                        6.2970e-01
                                    ],
                                    [
                                        2.0089e-01, 5.8098e+00, -3.5409e+01,
                                        6.5650e-01, 6.2480e-01, 6.9540e-01,
                                        2.5380e-01
                                    ],
                                    [
                                        -1.9461e-01, 3.1309e+01, -1.0901e+00,
                                        2.8030e-01, 2.5800e-02, 4.8960e-01,
                                        3.2690e-01
                                    ]])

    lidar_point_tensor = Coord3DMode.convert_point(cam_points.tensor,
                                                   Coord3DMode.CAM,
                                                   Coord3DMode.LIDAR)
    assert torch.allclose(expected_tensor, convert_lidar_points.tensor, 1e-4)
    assert torch.allclose(lidar_point_tensor, convert_lidar_points.tensor,
                          1e-4)

    convert_depth_points = cam_points.convert_to(Coord3DMode.DEPTH)
    expected_tensor = torch.tensor([[
        -5.2422e+00, -2.9757e-01, 4.0021e+01, 6.6660e-01, 1.9560e-01,
        4.9740e-01, 9.4090e-01
    ],
                                    [
                                        -2.6675e+01, 9.1435e-01, 5.5950e+00,
                                        1.5020e-01, 3.7070e-01, 1.0860e-01,
                                        6.2970e-01
                                    ],
                                    [
                                        -5.8098e+00, -2.0089e-01, 3.5409e+01,
                                        6.5650e-01, 6.2480e-01, 6.9540e-01,
                                        2.5380e-01
                                    ],
                                    [
                                        -3.1309e+01, 1.9461e-01, 1.0901e+00,
                                        2.8030e-01, 2.5800e-02, 4.8960e-01,
                                        3.2690e-01
                                    ]])

    depth_point_tensor = Coord3DMode.convert_point(cam_points.tensor,
                                                   Coord3DMode.CAM,
                                                   Coord3DMode.DEPTH)
    assert torch.allclose(expected_tensor, convert_depth_points.tensor, 1e-4)
    assert torch.allclose(depth_point_tensor, convert_depth_points.tensor,
                          1e-4)

    # test LIDAR to CAM and DEPTH
    lidar_points = LiDARPoints(
        points_np,
        points_dim=7,
        attribute_dims=dict(color=[3, 4, 5], height=6))

    convert_cam_points = lidar_points.convert_to(Coord3DMode.CAM)
    expected_tensor = torch.tensor([[
        -4.0021e+01, -2.9757e-01, -5.2422e+00, 6.6660e-01, 1.9560e-01,
        4.9740e-01, 9.4090e-01
    ],
                                    [
                                        -5.5950e+00, 9.1435e-01, -2.6675e+01,
                                        1.5020e-01, 3.7070e-01, 1.0860e-01,
                                        6.2970e-01
                                    ],
                                    [
                                        -3.5409e+01, -2.0089e-01, -5.8098e+00,
                                        6.5650e-01, 6.2480e-01, 6.9540e-01,
                                        2.5380e-01
                                    ],
                                    [
                                        -1.0901e+00, 1.9461e-01, -3.1309e+01,
                                        2.8030e-01, 2.5800e-02, 4.8960e-01,
                                        3.2690e-01
                                    ]])

    cam_point_tensor = Coord3DMode.convert_point(lidar_points.tensor,
                                                 Coord3DMode.LIDAR,
                                                 Coord3DMode.CAM)
    assert torch.allclose(expected_tensor, convert_cam_points.tensor, 1e-4)
    assert torch.allclose(cam_point_tensor, convert_cam_points.tensor, 1e-4)

    convert_depth_points = lidar_points.convert_to(Coord3DMode.DEPTH)
    expected_tensor = torch.tensor([[
        -4.0021e+01, -5.2422e+00, 2.9757e-01, 6.6660e-01, 1.9560e-01,
        4.9740e-01, 9.4090e-01
    ],
                                    [
                                        -5.5950e+00, -2.6675e+01, -9.1435e-01,
                                        1.5020e-01, 3.7070e-01, 1.0860e-01,
                                        6.2970e-01
                                    ],
                                    [
                                        -3.5409e+01, -5.8098e+00, 2.0089e-01,
                                        6.5650e-01, 6.2480e-01, 6.9540e-01,
                                        2.5380e-01
                                    ],
                                    [
                                        -1.0901e+00, -3.1309e+01, -1.9461e-01,
                                        2.8030e-01, 2.5800e-02, 4.8960e-01,
                                        3.2690e-01
                                    ]])

    depth_point_tensor = Coord3DMode.convert_point(lidar_points.tensor,
                                                   Coord3DMode.LIDAR,
                                                   Coord3DMode.DEPTH)
    assert torch.allclose(expected_tensor, convert_depth_points.tensor, 1e-4)
    assert torch.allclose(depth_point_tensor, convert_depth_points.tensor,
                          1e-4)

    # test DEPTH to CAM and LIDAR
    depth_points = DepthPoints(
        points_np,
        points_dim=7,
        attribute_dims=dict(color=[3, 4, 5], height=6))

    convert_cam_points = depth_points.convert_to(Coord3DMode.CAM)
    expected_tensor = torch.tensor([[
        -5.2422e+00, 2.9757e-01, -4.0021e+01, 6.6660e-01, 1.9560e-01,
        4.9740e-01, 9.4090e-01
    ],
                                    [
                                        -2.6675e+01, -9.1435e-01, -5.5950e+00,
                                        1.5020e-01, 3.7070e-01, 1.0860e-01,
                                        6.2970e-01
                                    ],
                                    [
                                        -5.8098e+00, 2.0089e-01, -3.5409e+01,
                                        6.5650e-01, 6.2480e-01, 6.9540e-01,
                                        2.5380e-01
                                    ],
                                    [
                                        -3.1309e+01, -1.9461e-01, -1.0901e+00,
                                        2.8030e-01, 2.5800e-02, 4.8960e-01,
                                        3.2690e-01
                                    ]])

    cam_point_tensor = Coord3DMode.convert_point(depth_points.tensor,
                                                 Coord3DMode.DEPTH,
                                                 Coord3DMode.CAM)
    assert torch.allclose(expected_tensor, convert_cam_points.tensor, 1e-4)
    assert torch.allclose(cam_point_tensor, convert_cam_points.tensor, 1e-4)

    convert_lidar_points = depth_points.convert_to(Coord3DMode.LIDAR)
    expected_tensor = torch.tensor([[
        4.0021e+01, 5.2422e+00, 2.9757e-01, 6.6660e-01, 1.9560e-01, 4.9740e-01,
        9.4090e-01
    ],
                                    [
                                        5.5950e+00, 2.6675e+01, -9.1435e-01,
                                        1.5020e-01, 3.7070e-01, 1.0860e-01,
                                        6.2970e-01
                                    ],
                                    [
                                        3.5409e+01, 5.8098e+00, 2.0089e-01,
                                        6.5650e-01, 6.2480e-01, 6.9540e-01,
                                        2.5380e-01
                                    ],
                                    [
                                        1.0901e+00, 3.1309e+01, -1.9461e-01,
                                        2.8030e-01, 2.5800e-02, 4.8960e-01,
                                        3.2690e-01
                                    ]])

    lidar_point_tensor = Coord3DMode.convert_point(depth_points.tensor,
                                                   Coord3DMode.DEPTH,
                                                   Coord3DMode.LIDAR)
    assert torch.allclose(lidar_point_tensor, convert_lidar_points.tensor,
                          1e-4)
    assert torch.allclose(lidar_point_tensor, convert_lidar_points.tensor,
                          1e-4)


def test_boxes_conversion():
    # test CAM to LIDAR and DEPTH
    cam_boxes = CameraInstance3DBoxes(
        [[1.7802081, 2.516249, -1.7501148, 1.75, 3.39, 1.65, 1.48],
         [8.959413, 2.4567227, -1.6357126, 1.54, 4.01, 1.57, 1.62],
         [28.2967, -0.5557558, -1.303325, 1.47, 2.23, 1.48, -1.57],
         [26.66902, 21.82302, -1.736057, 1.56, 3.48, 1.4, -1.69],
         [31.31978, 8.162144, -1.6217787, 1.74, 3.77, 1.48, 2.79]])
    convert_lidar_boxes = Coord3DMode.convert(cam_boxes, Coord3DMode.CAM,
                                              Coord3DMode.LIDAR)

    expected_tensor = torch.tensor(
        [[-1.7501, -1.7802, -2.5162, 1.6500, 1.7500, 3.3900, 1.4800],
         [-1.6357, -8.9594, -2.4567, 1.5700, 1.5400, 4.0100, 1.6200],
         [-1.3033, -28.2967, 0.5558, 1.4800, 1.4700, 2.2300, -1.5700],
         [-1.7361, -26.6690, -21.8230, 1.4000, 1.5600, 3.4800, -1.6900],
         [-1.6218, -31.3198, -8.1621, 1.4800, 1.7400, 3.7700, 2.7900]])
    assert torch.allclose(expected_tensor, convert_lidar_boxes.tensor, 1e-3)

    convert_depth_boxes = Coord3DMode.convert(cam_boxes, Coord3DMode.CAM,
                                              Coord3DMode.DEPTH)
    expected_tensor = torch.tensor(
        [[1.7802, 1.7501, 2.5162, 1.7500, 1.6500, 3.3900, 1.4800],
         [8.9594, 1.6357, 2.4567, 1.5400, 1.5700, 4.0100, 1.6200],
         [28.2967, 1.3033, -0.5558, 1.4700, 1.4800, 2.2300, -1.5700],
         [26.6690, 1.7361, 21.8230, 1.5600, 1.4000, 3.4800, -1.6900],
         [31.3198, 1.6218, 8.1621, 1.7400, 1.4800, 3.7700, 2.7900]])
    assert torch.allclose(expected_tensor, convert_depth_boxes.tensor, 1e-3)

    # test LIDAR to CAM and DEPTH
    lidar_boxes = LiDARInstance3DBoxes(
        [[1.7802081, 2.516249, -1.7501148, 1.75, 3.39, 1.65, 1.48],
         [8.959413, 2.4567227, -1.6357126, 1.54, 4.01, 1.57, 1.62],
         [28.2967, -0.5557558, -1.303325, 1.47, 2.23, 1.48, -1.57],
         [26.66902, 21.82302, -1.736057, 1.56, 3.48, 1.4, -1.69],
         [31.31978, 8.162144, -1.6217787, 1.74, 3.77, 1.48, 2.79]])
    convert_cam_boxes = Coord3DMode.convert(lidar_boxes, Coord3DMode.LIDAR,
                                            Coord3DMode.CAM)
    expected_tensor = torch.tensor(
        [[-2.5162, 1.7501, 1.7802, 3.3900, 1.6500, 1.7500, 1.4800],
         [-2.4567, 1.6357, 8.9594, 4.0100, 1.5700, 1.5400, 1.6200],
         [0.5558, 1.3033, 28.2967, 2.2300, 1.4800, 1.4700, -1.5700],
         [-21.8230, 1.7361, 26.6690, 3.4800, 1.4000, 1.5600, -1.6900],
         [-8.1621, 1.6218, 31.3198, 3.7700, 1.4800, 1.7400, 2.7900]])
    assert torch.allclose(expected_tensor, convert_cam_boxes.tensor, 1e-3)

    convert_depth_boxes = Coord3DMode.convert(lidar_boxes, Coord3DMode.LIDAR,
                                              Coord3DMode.DEPTH)
    expected_tensor = torch.tensor(
        [[-2.5162, 1.7802, -1.7501, 3.3900, 1.7500, 1.6500, 1.4800],
         [-2.4567, 8.9594, -1.6357, 4.0100, 1.5400, 1.5700, 1.6200],
         [0.5558, 28.2967, -1.3033, 2.2300, 1.4700, 1.4800, -1.5700],
         [-21.8230, 26.6690, -1.7361, 3.4800, 1.5600, 1.4000, -1.6900],
         [-8.1621, 31.3198, -1.6218, 3.7700, 1.7400, 1.4800, 2.7900]])
    assert torch.allclose(expected_tensor, convert_depth_boxes.tensor, 1e-3)

    # test DEPTH to CAM and LIDAR
    depth_boxes = DepthInstance3DBoxes(
        [[1.7802081, 2.516249, -1.7501148, 1.75, 3.39, 1.65, 1.48],
         [8.959413, 2.4567227, -1.6357126, 1.54, 4.01, 1.57, 1.62],
         [28.2967, -0.5557558, -1.303325, 1.47, 2.23, 1.48, -1.57],
         [26.66902, 21.82302, -1.736057, 1.56, 3.48, 1.4, -1.69],
         [31.31978, 8.162144, -1.6217787, 1.74, 3.77, 1.48, 2.79]])
    convert_cam_boxes = Coord3DMode.convert(depth_boxes, Coord3DMode.DEPTH,
                                            Coord3DMode.CAM)
    expected_tensor = torch.tensor(
        [[1.7802, -1.7501, -2.5162, 1.7500, 1.6500, 3.3900, 1.4800],
         [8.9594, -1.6357, -2.4567, 1.5400, 1.5700, 4.0100, 1.6200],
         [28.2967, -1.3033, 0.5558, 1.4700, 1.4800, 2.2300, -1.5700],
         [26.6690, -1.7361, -21.8230, 1.5600, 1.4000, 3.4800, -1.6900],
         [31.3198, -1.6218, -8.1621, 1.7400, 1.4800, 3.7700, 2.7900]])
    assert torch.allclose(expected_tensor, convert_cam_boxes.tensor, 1e-3)

    convert_lidar_boxes = Coord3DMode.convert(depth_boxes, Coord3DMode.DEPTH,
                                              Coord3DMode.LIDAR)
    expected_tensor = torch.tensor(
        [[2.5162, -1.7802, -1.7501, 3.3900, 1.7500, 1.6500, 1.4800],
         [2.4567, -8.9594, -1.6357, 4.0100, 1.5400, 1.5700, 1.6200],
         [-0.5558, -28.2967, -1.3033, 2.2300, 1.4700, 1.4800, -1.5700],
         [21.8230, -26.6690, -1.7361, 3.4800, 1.5600, 1.4000, -1.6900],
         [8.1621, -31.3198, -1.6218, 3.7700, 1.7400, 1.4800, 2.7900]])
    assert torch.allclose(expected_tensor, convert_lidar_boxes.tensor, 1e-3)
