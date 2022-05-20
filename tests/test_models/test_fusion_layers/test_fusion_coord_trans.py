# Copyright (c) OpenMMLab. All rights reserved.
"""Tests coords transformation in fusion modules.

CommandLine:
    pytest tests/test_models/test_fusion/test_fusion_coord_trans.py
"""

import torch

from mmdet3d.models.fusion_layers import apply_3d_transformation


def test_coords_transformation():
    """Test the transformation of 3d coords."""

    # H+R+S+T, not reverse, depth
    img_meta = {
        'pcd_scale_factor':
        1.2311e+00,
        'pcd_rotation': [[8.660254e-01, 0.5, 0], [-0.5, 8.660254e-01, 0],
                         [0, 0, 1.0e+00]],
        'pcd_trans': [1.111e-02, -8.88e-03, 0.0],
        'pcd_horizontal_flip':
        True,
        'transformation_3d_flow': ['HF', 'R', 'S', 'T']
    }

    pcd = torch.tensor([[-5.2422e+00, -2.9757e-01, 4.0021e+01],
                        [-9.1435e-01, 2.6675e+01, -5.5950e+00],
                        [2.0089e-01, 5.8098e+00, -3.5409e+01],
                        [-1.9461e-01, 3.1309e+01, -1.0901e+00]])

    pcd_transformed = apply_3d_transformation(
        pcd, 'DEPTH', img_meta, reverse=False)

    expected_tensor = torch.tensor(
        [[5.78332345e+00, 2.900697e+00, 4.92698531e+01],
         [-1.5433839e+01, 2.8993850e+01, -6.8880045e+00],
         [-3.77929405e+00, 6.061661e+00, -4.35920199e+01],
         [-1.9053658e+01, 3.3491436e+01, -1.34202211e+00]])

    assert torch.allclose(expected_tensor, pcd_transformed, 1e-4)

    # H+R+S+T, reverse, depth
    img_meta = {
        'pcd_scale_factor':
        7.07106781e-01,
        'pcd_rotation': [[7.07106781e-01, 7.07106781e-01, 0.0],
                         [-7.07106781e-01, 7.07106781e-01, 0.0],
                         [0.0, 0.0, 1.0e+00]],
        'pcd_trans': [0.0, 0.0, 0.0],
        'pcd_horizontal_flip':
        False,
        'transformation_3d_flow': ['HF', 'R', 'S', 'T']
    }

    pcd = torch.tensor([[-5.2422e+00, -2.9757e-01, 4.0021e+01],
                        [-9.1435e+01, 2.6675e+01, -5.5950e+00],
                        [6.061661e+00, -0.0, -1.0e+02]])

    pcd_transformed = apply_3d_transformation(
        pcd, 'DEPTH', img_meta, reverse=True)

    expected_tensor = torch.tensor(
        [[-5.53977e+00, 4.94463e+00, 5.65982409e+01],
         [-6.476e+01, 1.1811e+02, -7.91252488e+00],
         [6.061661e+00, -6.061661e+00, -1.41421356e+02]])
    assert torch.allclose(expected_tensor, pcd_transformed, 1e-4)

    # H+R+S+T, not reverse, camera
    img_meta = {
        'pcd_scale_factor':
        1.0 / 7.07106781e-01,
        'pcd_rotation': [[7.07106781e-01, 0.0, 7.07106781e-01],
                         [0.0, 1.0e+00, 0.0],
                         [-7.07106781e-01, 0.0, 7.07106781e-01]],
        'pcd_trans': [1.0e+00, -1.0e+00, 0.0],
        'pcd_horizontal_flip':
        True,
        'transformation_3d_flow': ['HF', 'S', 'R', 'T']
    }

    pcd = torch.tensor([[-5.2422e+00, 4.0021e+01, -2.9757e-01],
                        [-9.1435e+01, -5.5950e+00, 2.6675e+01],
                        [6.061661e+00, -1.0e+02, -0.0]])

    pcd_transformed = apply_3d_transformation(
        pcd, 'CAMERA', img_meta, reverse=False)

    expected_tensor = torch.tensor(
        [[6.53977e+00, 5.55982409e+01, 4.94463e+00],
         [6.576e+01, -8.91252488e+00, 1.1811e+02],
         [-5.061661e+00, -1.42421356e+02, -6.061661e+00]])

    assert torch.allclose(expected_tensor, pcd_transformed, 1e-4)

    # V, reverse, camera
    img_meta = {'pcd_vertical_flip': True, 'transformation_3d_flow': ['VF']}

    pcd_transformed = apply_3d_transformation(
        pcd, 'CAMERA', img_meta, reverse=True)

    expected_tensor = torch.tensor([[-5.2422e+00, 4.0021e+01, 2.9757e-01],
                                    [-9.1435e+01, -5.5950e+00, -2.6675e+01],
                                    [6.061661e+00, -1.0e+02, 0.0]])

    assert torch.allclose(expected_tensor, pcd_transformed, 1e-4)

    # V+H, not reverse, depth
    img_meta = {
        'pcd_vertical_flip': True,
        'pcd_horizontal_flip': True,
        'transformation_3d_flow': ['VF', 'HF']
    }

    pcd_transformed = apply_3d_transformation(
        pcd, 'DEPTH', img_meta, reverse=False)

    expected_tensor = torch.tensor([[5.2422e+00, -4.0021e+01, -2.9757e-01],
                                    [9.1435e+01, 5.5950e+00, 2.6675e+01],
                                    [-6.061661e+00, 1.0e+02, 0.0]])
    assert torch.allclose(expected_tensor, pcd_transformed, 1e-4)

    # V+H, reverse, lidar
    img_meta = {
        'pcd_vertical_flip': True,
        'pcd_horizontal_flip': True,
        'transformation_3d_flow': ['VF', 'HF']
    }

    pcd_transformed = apply_3d_transformation(
        pcd, 'LIDAR', img_meta, reverse=True)

    expected_tensor = torch.tensor([[5.2422e+00, -4.0021e+01, -2.9757e-01],
                                    [9.1435e+01, 5.5950e+00, 2.6675e+01],
                                    [-6.061661e+00, 1.0e+02, 0.0]])
    assert torch.allclose(expected_tensor, pcd_transformed, 1e-4)
