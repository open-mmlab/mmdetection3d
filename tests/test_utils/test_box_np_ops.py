# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np


def test_camera_to_lidar():
    from mmdet3d.core.bbox.box_np_ops import camera_to_lidar
    points = np.array([[1.84, 1.47, 8.41]])
    rect = np.array([[0.9999128, 0.01009263, -0.00851193, 0.],
                     [-0.01012729, 0.9999406, -0.00403767, 0.],
                     [0.00847068, 0.00412352, 0.9999556, 0.], [0., 0., 0.,
                                                               1.]])
    Trv2c = np.array([[0.00692796, -0.9999722, -0.00275783, -0.02457729],
                      [-0.00116298, 0.00274984, -0.9999955, -0.06127237],
                      [0.9999753, 0.00693114, -0.0011439, -0.3321029],
                      [0., 0., 0., 1.]])
    points_lidar = camera_to_lidar(points, rect, Trv2c)
    expected_points = np.array([[8.73138192, -1.85591746, -1.59969933]])
    assert np.allclose(points_lidar, expected_points)


def test_box_camera_to_lidar():
    from mmdet3d.core.bbox.box_np_ops import box_camera_to_lidar
    box = np.array([[1.84, 1.47, 8.41, 1.2, 1.89, 0.48, -0.01]])
    rect = np.array([[0.9999128, 0.01009263, -0.00851193, 0.],
                     [-0.01012729, 0.9999406, -0.00403767, 0.],
                     [0.00847068, 0.00412352, 0.9999556, 0.], [0., 0., 0.,
                                                               1.]])
    Trv2c = np.array([[0.00692796, -0.9999722, -0.00275783, -0.02457729],
                      [-0.00116298, 0.00274984, -0.9999955, -0.06127237],
                      [0.9999753, 0.00693114, -0.0011439, -0.3321029],
                      [0., 0., 0., 1.]])
    box_lidar = box_camera_to_lidar(box, rect, Trv2c)
    expected_box = np.array([[
        8.73138192, -1.85591746, -1.59969933, 1.2, 0.48, 1.89, 0.01 - np.pi / 2
    ]])
    assert np.allclose(box_lidar, expected_box)


def test_corners_nd():
    from mmdet3d.core.bbox.box_np_ops import corners_nd
    dims = np.array([[0.47, 0.98]])
    corners = corners_nd(dims)
    expected_corners = np.array([[[-0.235, -0.49], [-0.235, 0.49],
                                  [0.235, 0.49], [0.235, -0.49]]])
    assert np.allclose(corners, expected_corners)


def test_center_to_corner_box2d():
    from mmdet3d.core.bbox.box_np_ops import center_to_corner_box2d
    center = np.array([[9.348705, -3.6271024]])
    dims = np.array([[0.47, 0.98]])
    angles = np.array([3.14])
    corner = center_to_corner_box2d(center, dims, angles)
    expected_corner = np.array([[[9.584485, -3.1374772], [9.582925, -4.117476],
                                 [9.112926, -4.1167274],
                                 [9.114486, -3.1367288]]])
    assert np.allclose(corner, expected_corner)

    center = np.array([[-0.0, 0.0]])
    dims = np.array([[4.0, 8.0]])
    angles = np.array([-0.785398])  # -45 degrees
    corner = center_to_corner_box2d(center, dims, angles)
    expected_corner = np.array([[[-4.24264, -1.41421], [1.41421, 4.24264],
                                 [4.24264, 1.41421], [-1.41421, -4.24264]]])
    assert np.allclose(corner, expected_corner)


def test_points_in_convex_polygon_jit():
    from mmdet3d.core.bbox.box_np_ops import points_in_convex_polygon_jit
    points = np.array([[0.4, 0.4], [0.5, 0.5], [0.6, 0.6]])
    polygons = np.array([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.5], [0.0, 0.0]],
                         [[1.0, 0.0], [1.0, 1.0], [0.5, 1.0], [0.0, 1.0]],
                         [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]])
    res = points_in_convex_polygon_jit(points, polygons)
    expected_res = np.array([[1, 0, 1], [0, 0, 0], [0, 1, 0]]).astype(np.bool)
    assert np.allclose(res, expected_res)

    polygons = np.array([[[0.0, 0.0], [0.0, 1.0], [0.5, 0.5], [1.0, 0.0]],
                         [[0.0, 1.0], [1.0, 1.0], [1.0, 0.5], [1.0, 0.0]],
                         [[1.0, 0.0], [0.0, -1.0], [-1.0, 0.0], [0.0, 1.1]]])
    res = points_in_convex_polygon_jit(points, polygons, clockwise=True)
    expected_res = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 0]]).astype(np.bool)
    assert np.allclose(res, expected_res)
