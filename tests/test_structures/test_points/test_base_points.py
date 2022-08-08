# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmdet3d.structures.points import BasePoints


def test_base_points():
    # test empty initialization
    empty_boxes = []
    points = BasePoints(empty_boxes)
    assert points.tensor.shape[0] == 0
    assert points.tensor.shape[1] == 3

    # Test init with origin
    points_np = np.array([[-5.24223238e+00, 4.00209696e+01, 2.97570381e-01],
                          [-2.66751588e+01, 5.59499564e+00, -9.14345860e-01],
                          [-5.80979675e+00, 3.54092357e+01, 2.00889888e-01],
                          [-3.13086877e+01, 1.09007628e+00, -1.94612112e-01]],
                         dtype=np.float32)
    base_points = BasePoints(points_np, points_dim=3)
    assert base_points.tensor.shape[0] == 4

    # Test init with color and height
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
    base_points = BasePoints(
        points_np,
        points_dim=7,
        attribute_dims=dict(color=[3, 4, 5], height=6))
    expected_tensor = torch.tensor([[
        -5.24223238e+00, 4.00209696e+01, 2.97570381e-01, 0.6666, 0.1956,
        0.4974, 0.9409
    ],
                                    [
                                        -2.66751588e+01, 5.59499564e+00,
                                        -9.14345860e-01, 0.1502, 0.3707,
                                        0.1086, 0.6297
                                    ],
                                    [
                                        -5.80979675e+00, 3.54092357e+01,
                                        2.00889888e-01, 0.6565, 0.6248, 0.6954,
                                        0.2538
                                    ],
                                    [
                                        -3.13086877e+01, 1.09007628e+00,
                                        -1.94612112e-01, 0.2803, 0.0258,
                                        0.4896, 0.3269
                                    ]])

    assert torch.allclose(expected_tensor, base_points.tensor)
    assert torch.allclose(expected_tensor[:, :2], base_points.bev)
    assert torch.allclose(expected_tensor[:, :3], base_points.coord)
    assert torch.allclose(expected_tensor[:, 3:6], base_points.color)
    assert torch.allclose(expected_tensor[:, 6], base_points.height)

    # test points clone
    new_base_points = base_points.clone()
    assert torch.allclose(new_base_points.tensor, base_points.tensor)

    # test points shuffle
    new_base_points.shuffle()
    assert new_base_points.tensor.shape == torch.Size([4, 7])

    # test points rotation
    rot_mat = torch.tensor([[0.93629336, -0.27509585, 0.21835066],
                            [0.28962948, 0.95642509, -0.03695701],
                            [-0.19866933, 0.0978434, 0.97517033]])

    base_points.rotate(rot_mat)
    expected_tensor = torch.tensor([[
        6.6239e+00, 3.9748e+01, -2.3335e+00, 6.6660e-01, 1.9560e-01,
        4.9740e-01, 9.4090e-01
    ],
                                    [
                                        -2.3174e+01, 1.2600e+01, -6.9230e+00,
                                        1.5020e-01, 3.7070e-01, 1.0860e-01,
                                        6.2970e-01
                                    ],
                                    [
                                        4.7760e+00, 3.5484e+01, -2.3813e+00,
                                        6.5650e-01, 6.2480e-01, 6.9540e-01,
                                        2.5380e-01
                                    ],
                                    [
                                        -2.8960e+01, 9.6364e+00, -7.0663e+00,
                                        2.8030e-01, 2.5800e-02, 4.8960e-01,
                                        3.2690e-01
                                    ]])
    assert torch.allclose(expected_tensor, base_points.tensor, 1e-3)

    new_base_points = base_points.clone()
    new_base_points.rotate(0.1, axis=2)
    expected_tensor = torch.tensor([[
        2.6226e+00, 4.0211e+01, -2.3335e+00, 6.6660e-01, 1.9560e-01,
        4.9740e-01, 9.4090e-01
    ],
                                    [
                                        -2.4316e+01, 1.0224e+01, -6.9230e+00,
                                        1.5020e-01, 3.7070e-01, 1.0860e-01,
                                        6.2970e-01
                                    ],
                                    [
                                        1.2096e+00, 3.5784e+01, -2.3813e+00,
                                        6.5650e-01, 6.2480e-01, 6.9540e-01,
                                        2.5380e-01
                                    ],
                                    [
                                        -2.9777e+01, 6.6971e+00, -7.0663e+00,
                                        2.8030e-01, 2.5800e-02, 4.8960e-01,
                                        3.2690e-01
                                    ]])
    assert torch.allclose(expected_tensor, new_base_points.tensor, 1e-3)

    # test points translation
    translation_vector = torch.tensor([0.93629336, -0.27509585, 0.21835066])
    base_points.translate(translation_vector)
    expected_tensor = torch.tensor([[
        7.5602e+00, 3.9473e+01, -2.1152e+00, 6.6660e-01, 1.9560e-01,
        4.9740e-01, 9.4090e-01
    ],
                                    [
                                        -2.2237e+01, 1.2325e+01, -6.7046e+00,
                                        1.5020e-01, 3.7070e-01, 1.0860e-01,
                                        6.2970e-01
                                    ],
                                    [
                                        5.7123e+00, 3.5209e+01, -2.1629e+00,
                                        6.5650e-01, 6.2480e-01, 6.9540e-01,
                                        2.5380e-01
                                    ],
                                    [
                                        -2.8023e+01, 9.3613e+00, -6.8480e+00,
                                        2.8030e-01, 2.5800e-02, 4.8960e-01,
                                        3.2690e-01
                                    ]])
    assert torch.allclose(expected_tensor, base_points.tensor, 1e-4)

    # test points filter
    point_range = [-10, -40, -10, 10, 40, 10]
    in_range_flags = base_points.in_range_3d(point_range)
    expected_flags = torch.tensor([True, False, True, False])
    assert torch.all(in_range_flags == expected_flags)

    # test points scale
    base_points.scale(1.2)
    expected_tensor = torch.tensor([[
        9.0722e+00, 4.7368e+01, -2.5382e+00, 6.6660e-01, 1.9560e-01,
        4.9740e-01, 9.4090e-01
    ],
                                    [
                                        -2.6685e+01, 1.4790e+01, -8.0455e+00,
                                        1.5020e-01, 3.7070e-01, 1.0860e-01,
                                        6.2970e-01
                                    ],
                                    [
                                        6.8547e+00, 4.2251e+01, -2.5955e+00,
                                        6.5650e-01, 6.2480e-01, 6.9540e-01,
                                        2.5380e-01
                                    ],
                                    [
                                        -3.3628e+01, 1.1234e+01, -8.2176e+00,
                                        2.8030e-01, 2.5800e-02, 4.8960e-01,
                                        3.2690e-01
                                    ]])
    assert torch.allclose(expected_tensor, base_points.tensor, 1e-3)

    # test get_item
    expected_tensor = torch.tensor(
        [[-26.6848, 14.7898, -8.0455, 0.1502, 0.3707, 0.1086, 0.6297]])
    assert torch.allclose(expected_tensor, base_points[1].tensor, 1e-4)
    expected_tensor = torch.tensor(
        [[-26.6848, 14.7898, -8.0455, 0.1502, 0.3707, 0.1086, 0.6297],
         [6.8547, 42.2509, -2.5955, 0.6565, 0.6248, 0.6954, 0.2538]])
    assert torch.allclose(expected_tensor, base_points[1:3].tensor, 1e-4)
    mask = torch.tensor([True, False, True, False])
    expected_tensor = torch.tensor(
        [[9.0722, 47.3678, -2.5382, 0.6666, 0.1956, 0.4974, 0.9409],
         [6.8547, 42.2509, -2.5955, 0.6565, 0.6248, 0.6954, 0.2538]])
    assert torch.allclose(expected_tensor, base_points[mask].tensor, 1e-4)
    expected_tensor = torch.tensor([[0.6666], [0.1502], [0.6565], [0.2803]])
    assert torch.allclose(expected_tensor, base_points[:, 3].tensor, 1e-4)

    # test length
    assert len(base_points) == 4

    # test repr
    expected_repr = 'BasePoints(\n    '\
        'tensor([[ 9.0722e+00,  4.7368e+01, -2.5382e+00,  '\
        '6.6660e-01,  1.9560e-01,\n          4.9740e-01,  '\
        '9.4090e-01],\n        '\
        '[-2.6685e+01,  1.4790e+01, -8.0455e+00,  1.5020e-01,  '\
        '3.7070e-01,\n          '\
        '1.0860e-01,  6.2970e-01],\n        '\
        '[ 6.8547e+00,  4.2251e+01, -2.5955e+00,  6.5650e-01,  '\
        '6.2480e-01,\n          '\
        '6.9540e-01,  2.5380e-01],\n        '\
        '[-3.3628e+01,  1.1234e+01, -8.2176e+00,  2.8030e-01,  '\
        '2.5800e-02,\n          '\
        '4.8960e-01,  3.2690e-01]]))'
    assert expected_repr == str(base_points)

    # test concatenate
    base_points_clone = base_points.clone()
    cat_points = BasePoints.cat([base_points, base_points_clone])
    assert torch.allclose(cat_points.tensor[:len(base_points)],
                          base_points.tensor)

    # test iteration
    for i, point in enumerate(base_points):
        assert torch.allclose(point, base_points.tensor[i])

    # test new_point
    new_points = base_points.new_point([[1, 2, 3, 4, 5, 6, 7]])
    assert torch.allclose(
        new_points.tensor,
        torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=base_points.tensor.dtype))

    # test BasePoint indexing
    base_points = BasePoints(
        points_np,
        points_dim=7,
        attribute_dims=dict(height=3, color=[4, 5, 6]))
    assert torch.all(base_points[:, 3:].tensor == torch.tensor(points_np[:,
                                                                         3:]))

    # test set and get function for BasePoint color and height
    base_points = BasePoints(points_np[:, :3])
    assert base_points.attribute_dims is None
    base_points.height = points_np[:, 3]
    assert base_points.attribute_dims == dict(height=3)
    base_points.color = points_np[:, 4:]
    assert base_points.attribute_dims == dict(height=3, color=[4, 5, 6])
    assert torch.allclose(base_points.height,
                          torch.tensor([0.6666, 0.1502, 0.6565, 0.2803]))
    assert torch.allclose(
        base_points.color,
        torch.tensor([[0.1956, 0.4974, 0.9409], [0.3707, 0.1086, 0.6297],
                      [0.6248, 0.6954, 0.2538], [0.0258, 0.4896, 0.3269]]))
    # values to be set should have correct shape (e.g. number of points)
    with pytest.raises(ValueError):
        base_points.coord = np.random.rand(5, 3)
    with pytest.raises(ValueError):
        base_points.height = np.random.rand(3)
    with pytest.raises(ValueError):
        base_points.color = np.random.rand(4, 2)
    base_points.coord = points_np[:, [1, 2, 3]]
    base_points.height = points_np[:, 0]
    base_points.color = points_np[:, [4, 5, 6]]
    assert np.allclose(base_points.coord, points_np[:, 1:4])
    assert np.allclose(base_points.height, points_np[:, 0])
    assert np.allclose(base_points.color, points_np[:, 4:])
