# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmdet3d.core import array_converter, draw_heatmap_gaussian, points_img2cam


def test_gaussian():
    heatmap = torch.zeros((128, 128))
    ct_int = torch.tensor([64, 64], dtype=torch.int32)
    radius = 2
    draw_heatmap_gaussian(heatmap, ct_int, radius)
    assert torch.isclose(torch.sum(heatmap), torch.tensor(4.3505), atol=1e-3)


def test_array_converter():
    # to torch
    @array_converter(to_torch=True, apply_to=('array_a', 'array_b'))
    def test_func_1(array_a, array_b, container):
        container.append(array_a)
        container.append(array_b)
        return array_a.clone(), array_b.clone()

    np_array_a = np.array([0.0])
    np_array_b = np.array([0.0])
    container = []
    new_array_a, new_array_b = test_func_1(np_array_a, np_array_b, container)

    assert isinstance(new_array_a, np.ndarray)
    assert isinstance(new_array_b, np.ndarray)
    assert isinstance(container[0], torch.Tensor)
    assert isinstance(container[1], torch.Tensor)

    # one to torch and one not
    @array_converter(to_torch=True, apply_to=('array_a', ))
    def test_func_2(array_a, array_b):
        return torch.cat([array_a, array_b])

    with pytest.raises(TypeError):
        _ = test_func_2(np_array_a, np_array_b)

    # wrong template_arg_name_
    @array_converter(
        to_torch=True, apply_to=('array_a', ), template_arg_name_='array_c')
    def test_func_3(array_a, array_b):
        return torch.cat([array_a, array_b])

    with pytest.raises(ValueError):
        _ = test_func_3(np_array_a, np_array_b)

    # wrong apply_to
    @array_converter(to_torch=True, apply_to=('array_a', 'array_c'))
    def test_func_4(array_a, array_b):
        return torch.cat([array_a, array_b])

    with pytest.raises(ValueError):
        _ = test_func_4(np_array_a, np_array_b)

    # to numpy
    @array_converter(to_torch=False, apply_to=('array_a', 'array_b'))
    def test_func_5(array_a, array_b, container):
        container.append(array_a)
        container.append(array_b)
        return array_a.copy(), array_b.copy()

    pt_array_a = torch.tensor([0.0])
    pt_array_b = torch.tensor([0.0])
    container = []
    new_array_a, new_array_b = test_func_5(pt_array_a, pt_array_b, container)

    assert isinstance(container[0], np.ndarray)
    assert isinstance(container[1], np.ndarray)
    assert isinstance(new_array_a, torch.Tensor)
    assert isinstance(new_array_b, torch.Tensor)

    # apply_to = None
    @array_converter(to_torch=False)
    def test_func_6(array_a, array_b, container):
        container.append(array_a)
        container.append(array_b)
        return array_a.clone(), array_b.clone()

    container = []
    new_array_a, new_array_b = test_func_6(pt_array_a, pt_array_b, container)

    assert isinstance(container[0], torch.Tensor)
    assert isinstance(container[1], torch.Tensor)
    assert isinstance(new_array_a, torch.Tensor)
    assert isinstance(new_array_b, torch.Tensor)

    # with default arg
    @array_converter(to_torch=True, apply_to=('array_a', 'array_b'))
    def test_func_7(array_a, container, array_b=np.array([2.])):
        container.append(array_a)
        container.append(array_b)
        return array_a.clone(), array_b.clone()

    container = []
    new_array_a, new_array_b = test_func_7(np_array_a, container)

    assert isinstance(container[0], torch.Tensor)
    assert isinstance(container[1], torch.Tensor)
    assert isinstance(new_array_a, np.ndarray)
    assert isinstance(new_array_b, np.ndarray)
    assert np.allclose(new_array_b, np.array([2.]), 1e-3)

    # override default arg

    container = []
    new_array_a, new_array_b = test_func_7(np_array_a, container,
                                           np.array([4.]))

    assert isinstance(container[0], torch.Tensor)
    assert isinstance(container[1], torch.Tensor)
    assert isinstance(new_array_a, np.ndarray)
    assert np.allclose(new_array_b, np.array([4.]), 1e-3)

    # list arg
    @array_converter(to_torch=True, apply_to=('array_a', 'array_b'))
    def test_func_8(container, array_a, array_b=[2.]):
        container.append(array_a)
        container.append(array_b)
        return array_a.clone(), array_b.clone()

    container = []
    new_array_a, new_array_b = test_func_8(container, [3.])

    assert isinstance(container[0], torch.Tensor)
    assert isinstance(container[1], torch.Tensor)
    assert np.allclose(new_array_a, np.array([3.]), 1e-3)
    assert np.allclose(new_array_b, np.array([2.]), 1e-3)

    # number arg
    @array_converter(to_torch=True, apply_to=('array_a', 'array_b'))
    def test_func_9(container, array_a, array_b=1):
        container.append(array_a)
        container.append(array_b)
        return array_a.clone(), array_b.clone()

    container = []
    new_array_a, new_array_b = test_func_9(container, np_array_a)

    assert isinstance(container[0], torch.FloatTensor)
    assert isinstance(container[1], torch.FloatTensor)
    assert np.allclose(new_array_a, np_array_a, 1e-3)
    assert np.allclose(new_array_b, np.array(1.0), 1e-3)

    # feed kwargs
    container = []
    kwargs = {'array_a': [5.], 'array_b': [6.]}
    new_array_a, new_array_b = test_func_8(container, **kwargs)

    assert isinstance(container[0], torch.Tensor)
    assert isinstance(container[1], torch.Tensor)
    assert np.allclose(new_array_a, np.array([5.]), 1e-3)
    assert np.allclose(new_array_b, np.array([6.]), 1e-3)

    # feed args and kwargs
    container = []
    kwargs = {'array_b': [7.]}
    args = (container, [8.])
    new_array_a, new_array_b = test_func_8(*args, **kwargs)

    assert isinstance(container[0], torch.Tensor)
    assert isinstance(container[1], torch.Tensor)
    assert np.allclose(new_array_a, np.array([8.]), 1e-3)
    assert np.allclose(new_array_b, np.array([7.]), 1e-3)

    # wrong template arg type
    with pytest.raises(TypeError):
        new_array_a, new_array_b = test_func_9(container, 3 + 4j)

    with pytest.raises(TypeError):
        new_array_a, new_array_b = test_func_9(container, {})

    # invalid template arg list
    with pytest.raises(TypeError):
        new_array_a, new_array_b = test_func_9(container,
                                               [True, np.array([3.0])])


def test_points_img2cam():
    points = torch.tensor([[0.5764, 0.9109, 0.7576], [0.6656, 0.5498, 0.9813]])
    cam2img = torch.tensor([[700., 0., 450., 0.], [0., 700., 200., 0.],
                            [0., 0., 1., 0.]])
    xyzs = points_img2cam(points, cam2img)
    expected_xyzs = torch.tensor([[-0.4864, -0.2155, 0.7576],
                                  [-0.6299, -0.2796, 0.9813]])
    assert torch.allclose(xyzs, expected_xyzs, atol=1e-3)
