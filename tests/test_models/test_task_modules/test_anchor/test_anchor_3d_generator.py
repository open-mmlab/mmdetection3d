# Copyright (c) OpenMMLab. All rights reserved.
"""
CommandLine:
    pytest tests/test_utils/test_anchor.py
    xdoctest tests/test_utils/test_anchor.py zero

"""
import torch
from mmengine import DefaultScope

from mmdet3d.registry import TASK_UTILS


def test_anchor_3d_range_generator():

    import mmdet3d.models.task_modules

    assert hasattr(mmdet3d.models.task_modules, 'Anchor3DRangeGenerator')
    DefaultScope.get_instance(
        'test_ancho3drange_generator', scope_name='mmdet3d')

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    anchor_generator_cfg = dict(
        type='Anchor3DRangeGenerator',
        ranges=[
            [0, -39.68, -0.6, 70.4, 39.68, -0.6],
            [0, -39.68, -0.6, 70.4, 39.68, -0.6],
            [0, -39.68, -1.78, 70.4, 39.68, -1.78],
        ],
        sizes=[[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]],
        rotations=[0, 1.57],
        reshape_out=False)

    anchor_generator = TASK_UTILS.build(anchor_generator_cfg)
    repr_str = repr(anchor_generator)
    expected_repr_str = 'Anchor3DRangeGenerator(anchor_range=' \
                        '[[0, -39.68, -0.6, 70.4, 39.68, -0.6], ' \
                        '[0, -39.68, -0.6, 70.4, 39.68, -0.6], ' \
                        '[0, -39.68, -1.78, 70.4, 39.68, -1.78]],' \
                        '\nscales=[1],\nsizes=[[0.8, 0.6, 1.73], ' \
                        '[1.76, 0.6, 1.73], [3.9, 1.6, 1.56]],' \
                        '\nrotations=[0, 1.57],\nreshape_out=False,' \
                        '\nsize_per_range=True)'
    assert repr_str == expected_repr_str
    featmap_size = (8, 8)
    mr_anchors = anchor_generator.single_level_grid_anchors(
        featmap_size, 1.1, device=device)
    assert mr_anchors.shape == torch.Size([1, 8, 8, 3, 2, 7])


def test_aligned_anchor_generator():

    import mmdet3d.models.task_modules

    assert hasattr(mmdet3d.models.task_modules,
                   'AlignedAnchor3DRangeGenerator')
    DefaultScope.get_instance(
        'test_aligned_ancho3drange_generator', scope_name='mmdet3d')

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    anchor_generator_cfg = dict(
        type='AlignedAnchor3DRangeGenerator',
        ranges=[[-51.2, -51.2, -1.80, 51.2, 51.2, -1.80]],
        scales=[1, 2, 4],
        sizes=[
            [2.5981, 0.8660, 1.],  # 1.5/sqrt(3)
            [1.7321, 0.5774, 1.],  # 1/sqrt(3)
            [1., 1., 1.],
            [0.4, 0.4, 1],
        ],
        custom_values=[0, 0],
        rotations=[0, 1.57],
        size_per_range=False,
        reshape_out=True)

    featmap_sizes = [(16, 16), (8, 8), (4, 4)]
    anchor_generator = TASK_UTILS.build(anchor_generator_cfg)
    assert anchor_generator.num_base_anchors == 8

    # check base anchors
    expected_grid_anchors = [
        torch.tensor([[
            -48.0000, -48.0000, -1.8000, 2.5981, 0.8660, 1.0000, 0.0000,
            0.0000, 0.0000
        ],
                      [
                          -48.0000, -48.0000, -1.8000, 0.4000, 0.4000, 1.0000,
                          1.5700, 0.0000, 0.0000
                      ],
                      [
                          -41.6000, -48.0000, -1.8000, 0.4000, 0.4000, 1.0000,
                          0.0000, 0.0000, 0.0000
                      ],
                      [
                          -35.2000, -48.0000, -1.8000, 1.0000, 1.0000, 1.0000,
                          1.5700, 0.0000, 0.0000
                      ],
                      [
                          -28.8000, -48.0000, -1.8000, 1.0000, 1.0000, 1.0000,
                          0.0000, 0.0000, 0.0000
                      ],
                      [
                          -22.4000, -48.0000, -1.8000, 1.7321, 0.5774, 1.0000,
                          1.5700, 0.0000, 0.0000
                      ],
                      [
                          -16.0000, -48.0000, -1.8000, 1.7321, 0.5774, 1.0000,
                          0.0000, 0.0000, 0.0000
                      ],
                      [
                          -9.6000, -48.0000, -1.8000, 2.5981, 0.8660, 1.0000,
                          1.5700, 0.0000, 0.0000
                      ]],
                     device=device),
        torch.tensor([[
            -44.8000, -44.8000, -1.8000, 5.1962, 1.7320, 2.0000, 0.0000,
            0.0000, 0.0000
        ],
                      [
                          -44.8000, -44.8000, -1.8000, 0.8000, 0.8000, 2.0000,
                          1.5700, 0.0000, 0.0000
                      ],
                      [
                          -32.0000, -44.8000, -1.8000, 0.8000, 0.8000, 2.0000,
                          0.0000, 0.0000, 0.0000
                      ],
                      [
                          -19.2000, -44.8000, -1.8000, 2.0000, 2.0000, 2.0000,
                          1.5700, 0.0000, 0.0000
                      ],
                      [
                          -6.4000, -44.8000, -1.8000, 2.0000, 2.0000, 2.0000,
                          0.0000, 0.0000, 0.0000
                      ],
                      [
                          6.4000, -44.8000, -1.8000, 3.4642, 1.1548, 2.0000,
                          1.5700, 0.0000, 0.0000
                      ],
                      [
                          19.2000, -44.8000, -1.8000, 3.4642, 1.1548, 2.0000,
                          0.0000, 0.0000, 0.0000
                      ],
                      [
                          32.0000, -44.8000, -1.8000, 5.1962, 1.7320, 2.0000,
                          1.5700, 0.0000, 0.0000
                      ]],
                     device=device),
        torch.tensor([[
            -38.4000, -38.4000, -1.8000, 10.3924, 3.4640, 4.0000, 0.0000,
            0.0000, 0.0000
        ],
                      [
                          -38.4000, -38.4000, -1.8000, 1.6000, 1.6000, 4.0000,
                          1.5700, 0.0000, 0.0000
                      ],
                      [
                          -12.8000, -38.4000, -1.8000, 1.6000, 1.6000, 4.0000,
                          0.0000, 0.0000, 0.0000
                      ],
                      [
                          12.8000, -38.4000, -1.8000, 4.0000, 4.0000, 4.0000,
                          1.5700, 0.0000, 0.0000
                      ],
                      [
                          38.4000, -38.4000, -1.8000, 4.0000, 4.0000, 4.0000,
                          0.0000, 0.0000, 0.0000
                      ],
                      [
                          -38.4000, -12.8000, -1.8000, 6.9284, 2.3096, 4.0000,
                          1.5700, 0.0000, 0.0000
                      ],
                      [
                          -12.8000, -12.8000, -1.8000, 6.9284, 2.3096, 4.0000,
                          0.0000, 0.0000, 0.0000
                      ],
                      [
                          12.8000, -12.8000, -1.8000, 10.3924, 3.4640, 4.0000,
                          1.5700, 0.0000, 0.0000
                      ]],
                     device=device)
    ]
    multi_level_anchors = anchor_generator.grid_anchors(
        featmap_sizes, device=device)
    expected_multi_level_shapes = [
        torch.Size([2048, 9]),
        torch.Size([512, 9]),
        torch.Size([128, 9])
    ]
    for i, single_level_anchor in enumerate(multi_level_anchors):
        assert single_level_anchor.shape == expected_multi_level_shapes[i]
        # set [:56:7] thus it could cover 8 (len(size) * len(rotations))
        # anchors on 8 location
        assert single_level_anchor[:56:7].allclose(expected_grid_anchors[i])


def test_aligned_anchor_generator_per_cls():

    import mmdet3d.models.task_modules

    assert hasattr(mmdet3d.models.task_modules,
                   'AlignedAnchor3DRangeGeneratorPerCls')
    DefaultScope.get_instance(
        'test_ancho3drange_generator_percls', scope_name='mmdet3d')

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    anchor_generator_cfg = dict(
        type='AlignedAnchor3DRangeGeneratorPerCls',
        ranges=[[-100, -100, -1.80, 100, 100, -1.80],
                [-100, -100, -1.30, 100, 100, -1.30]],
        sizes=[[1.76, 0.63, 1.44], [2.35, 0.96, 1.59]],
        custom_values=[0, 0],
        rotations=[0, 1.57],
        reshape_out=False)

    featmap_sizes = [(100, 100), (50, 50)]
    anchor_generator = TASK_UTILS.build(anchor_generator_cfg)

    # check base anchors
    expected_grid_anchors = [[
        torch.tensor([[
            -99.0000, -99.0000, -1.8000, 1.7600, 0.6300, 1.4400, 0.0000,
            0.0000, 0.0000
        ],
                      [
                          -99.0000, -99.0000, -1.8000, 1.7600, 0.6300, 1.4400,
                          1.5700, 0.0000, 0.0000
                      ]],
                     device=device),
        torch.tensor([[
            -98.0000, -98.0000, -1.3000, 2.3500, 0.9600, 1.5900, 0.0000,
            0.0000, 0.0000
        ],
                      [
                          -98.0000, -98.0000, -1.3000, 2.3500, 0.9600, 1.5900,
                          1.5700, 0.0000, 0.0000
                      ]],
                     device=device)
    ]]
    multi_level_anchors = anchor_generator.grid_anchors(
        featmap_sizes, device=device)
    expected_multi_level_shapes = [[
        torch.Size([20000, 9]), torch.Size([5000, 9])
    ]]
    for i, single_level_anchor in enumerate(multi_level_anchors):
        assert len(single_level_anchor) == len(expected_multi_level_shapes[i])
        # set [:2*interval:interval] thus it could cover
        # 2 (len(size) * len(rotations)) anchors on 2 location
        # Note that len(size) for each class is always 1 in this case
        for j in range(len(single_level_anchor)):
            interval = int(expected_multi_level_shapes[i][j][0] / 2)
            assert single_level_anchor[j][:2 * interval:interval].allclose(
                expected_grid_anchors[i][j])
