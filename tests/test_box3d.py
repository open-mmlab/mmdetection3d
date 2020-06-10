import numpy as np
import pytest
import torch

from mmdet3d.core.bbox import (Box3DMode, CameraInstance3DBoxes,
                               DepthInstance3DBoxes, LiDARInstance3DBoxes)


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
        gravity_center_box, origin=[0.5, 0.5, 0.5])
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
    np_boxes = np.array(
        [[1.7802081, 2.516249, -1.7501148, 1.75, 3.39, 1.65, 1.48],
         [8.959413, 2.4567227, -1.6357126, 1.54, 4.01, 1.57, 1.62]],
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
    boxes_2 = LiDARInstance3DBoxes(th_boxes)
    assert torch.allclose(boxes_2.tensor, th_boxes)

    # test clone/to/device
    boxes_2 = boxes_2.clone()
    boxes_1 = boxes_1.to(boxes_2.device)

    # test box concatenation
    expected_tensor = torch.tensor(
        [[1.7802081, 2.516249, -1.7501148, 1.75, 3.39, 1.65, 1.48],
         [8.959413, 2.4567227, -1.6357126, 1.54, 4.01, 1.57, 1.62],
         [28.2967, -0.5557558, -1.303325, 1.47, 2.23, 1.48, -1.57],
         [26.66902, 21.82302, -1.736057, 1.56, 3.48, 1.4, -1.69],
         [31.31978, 8.162144, -1.6217787, 1.74, 3.77, 1.48, 2.79]])
    boxes = LiDARInstance3DBoxes.cat([boxes_1, boxes_2])
    assert torch.allclose(boxes.tensor, expected_tensor)
    # concatenate empty list
    empty_boxes = LiDARInstance3DBoxes.cat([])
    assert empty_boxes.tensor.shape[0] == 0
    assert empty_boxes.tensor.shape[-1] == 7

    # test box flip
    expected_tensor = torch.tensor(
        [[1.7802081, -2.516249, -1.7501148, 1.75, 3.39, 1.65, 1.6615927],
         [8.959413, -2.4567227, -1.6357126, 1.54, 4.01, 1.57, 1.5215927],
         [28.2967, 0.5557558, -1.303325, 1.47, 2.23, 1.48, 4.7115927],
         [26.66902, -21.82302, -1.736057, 1.56, 3.48, 1.4, 4.8315926],
         [31.31978, -8.162144, -1.6217787, 1.74, 3.77, 1.48, 0.35159278]])
    boxes.flip('horizontal')
    assert torch.allclose(boxes.tensor, expected_tensor)

    expected_tensor = torch.tensor(
        [[-1.7802, -2.5162, -1.7501, 1.7500, 3.3900, 1.6500, -1.6616],
         [-8.9594, -2.4567, -1.6357, 1.5400, 4.0100, 1.5700, -1.5216],
         [-28.2967, 0.5558, -1.3033, 1.4700, 2.2300, 1.4800, -4.7116],
         [-26.6690, -21.8230, -1.7361, 1.5600, 3.4800, 1.4000, -4.8316],
         [-31.3198, -8.1621, -1.6218, 1.7400, 3.7700, 1.4800, -0.3516]])
    boxes_flip_vert = boxes.clone()
    boxes_flip_vert.flip('vertical')
    assert torch.allclose(boxes_flip_vert.tensor, expected_tensor, 1e-4)

    # test box rotation
    expected_tensor = torch.tensor(
        [[1.0385344, -2.9020846, -1.7501148, 1.75, 3.39, 1.65, 1.9336663],
         [7.969653, -4.774011, -1.6357126, 1.54, 4.01, 1.57, 1.7936664],
         [27.405172, -7.0688415, -1.303325, 1.47, 2.23, 1.48, 4.9836664],
         [19.823532, -28.187025, -1.736057, 1.56, 3.48, 1.4, 5.1036663],
         [27.974297, -16.27845, -1.6217787, 1.74, 3.77, 1.48, 0.6236664]])
    boxes.rotate(0.27207362796436096)
    assert torch.allclose(boxes.tensor, expected_tensor)

    # test box scaling
    expected_tensor = torch.tensor([[
        1.0443488, -2.9183323, -1.7599131, 1.7597977, 3.4089797, 1.6592377,
        1.9336663
    ],
                                    [
                                        8.014273, -4.8007393, -1.6448704,
                                        1.5486219, 4.0324507, 1.57879,
                                        1.7936664
                                    ],
                                    [
                                        27.558605, -7.1084175, -1.310622,
                                        1.4782301, 2.242485, 1.488286,
                                        4.9836664
                                    ],
                                    [
                                        19.934517, -28.344835, -1.7457767,
                                        1.5687338, 3.4994833, 1.4078381,
                                        5.1036663
                                    ],
                                    [
                                        28.130915, -16.369587, -1.6308585,
                                        1.7497417, 3.791107, 1.488286,
                                        0.6236664
                                    ]])
    boxes.scale(1.00559866335275)
    assert torch.allclose(boxes.tensor, expected_tensor)

    # test box translation
    expected_tensor = torch.tensor([[
        1.1281544, -3.0507944, -1.9169292, 1.7597977, 3.4089797, 1.6592377,
        1.9336663
    ],
                                    [
                                        8.098079, -4.9332013, -1.8018866,
                                        1.5486219, 4.0324507, 1.57879,
                                        1.7936664
                                    ],
                                    [
                                        27.64241, -7.2408795, -1.4676381,
                                        1.4782301, 2.242485, 1.488286,
                                        4.9836664
                                    ],
                                    [
                                        20.018322, -28.477297, -1.9027928,
                                        1.5687338, 3.4994833, 1.4078381,
                                        5.1036663
                                    ],
                                    [
                                        28.21472, -16.502048, -1.7878747,
                                        1.7497417, 3.791107, 1.488286,
                                        0.6236664
                                    ]])
    boxes.translate([0.0838056, -0.13246193, -0.15701613])
    assert torch.allclose(boxes.tensor, expected_tensor)

    # test bbox in_range_bev
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
        4.9836664
    ],
                                    [
                                        20.018322, -28.477297, -1.9027928,
                                        1.5687338, 3.4994833, 1.4078381,
                                        5.1036663
                                    ],
                                    [
                                        28.21472, -16.502048, -1.7878747,
                                        1.7497417, 3.791107, 1.488286,
                                        0.6236664
                                    ]])
    assert len(index_boxes) == 3
    assert torch.allclose(index_boxes.tensor, expected_tensor)

    index_boxes = boxes[2]
    expected_tensor = torch.tensor([[
        27.64241, -7.2408795, -1.4676381, 1.4782301, 2.242485, 1.488286,
        4.9836664
    ]])
    assert len(index_boxes) == 1
    assert torch.allclose(index_boxes.tensor, expected_tensor)

    index_boxes = boxes[[2, 4]]
    expected_tensor = torch.tensor([[
        27.64241, -7.2408795, -1.4676381, 1.4782301, 2.242485, 1.488286,
        4.9836664
    ],
                                    [
                                        28.21472, -16.502048, -1.7878747,
                                        1.7497417, 3.791107, 1.488286,
                                        0.6236664
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
    expected_tesor = boxes.tensor.clone()
    assert torch.allclose(expected_tesor, boxes.tensor)

    boxes.flip()
    boxes.flip()
    boxes.limit_yaw()
    assert torch.allclose(expected_tesor, boxes.tensor)

    # test nearest_bev
    expected_tensor = torch.tensor([[-0.5763, -3.9307, 2.8326, -2.1709],
                                    [6.0819, -5.7075, 10.1143, -4.1589],
                                    [26.5212, -7.9800, 28.7637, -6.5018],
                                    [18.2686, -29.2617, 21.7681, -27.6929],
                                    [27.3398, -18.3976, 29.0896, -14.6065]])
    # the pytorch print loses some precision
    assert torch.allclose(
        boxes.nearest_bev, expected_tensor, rtol=1e-4, atol=1e-7)

    # obtained by the print of the original implementation
    expected_tensor = torch.tensor([[[2.4093e+00, -4.4784e+00, -1.9169e+00],
                                     [2.4093e+00, -4.4784e+00, -2.5769e-01],
                                     [-7.7767e-01, -3.2684e+00, -2.5769e-01],
                                     [-7.7767e-01, -3.2684e+00, -1.9169e+00],
                                     [3.0340e+00, -2.8332e+00, -1.9169e+00],
                                     [3.0340e+00, -2.8332e+00, -2.5769e-01],
                                     [-1.5301e-01, -1.6232e+00, -2.5769e-01],
                                     [-1.5301e-01, -1.6232e+00, -1.9169e+00]],
                                    [[9.8933e+00, -6.1340e+00, -1.8019e+00],
                                     [9.8933e+00, -6.1340e+00, -2.2310e-01],
                                     [5.9606e+00, -5.2427e+00, -2.2310e-01],
                                     [5.9606e+00, -5.2427e+00, -1.8019e+00],
                                     [1.0236e+01, -4.6237e+00, -1.8019e+00],
                                     [1.0236e+01, -4.6237e+00, -2.2310e-01],
                                     [6.3029e+00, -3.7324e+00, -2.2310e-01],
                                     [6.3029e+00, -3.7324e+00, -1.8019e+00]],
                                    [[2.8525e+01, -8.2534e+00, -1.4676e+00],
                                     [2.8525e+01, -8.2534e+00, 2.0648e-02],
                                     [2.6364e+01, -7.6525e+00, 2.0648e-02],
                                     [2.6364e+01, -7.6525e+00, -1.4676e+00],
                                     [2.8921e+01, -6.8292e+00, -1.4676e+00],
                                     [2.8921e+01, -6.8292e+00, 2.0648e-02],
                                     [2.6760e+01, -6.2283e+00, 2.0648e-02],
                                     [2.6760e+01, -6.2283e+00, -1.4676e+00]],
                                    [[2.1337e+01, -2.9870e+01, -1.9028e+00],
                                     [2.1337e+01, -2.9870e+01, -4.9495e-01],
                                     [1.8102e+01, -2.8535e+01, -4.9495e-01],
                                     [1.8102e+01, -2.8535e+01, -1.9028e+00],
                                     [2.1935e+01, -2.8420e+01, -1.9028e+00],
                                     [2.1935e+01, -2.8420e+01, -4.9495e-01],
                                     [1.8700e+01, -2.7085e+01, -4.9495e-01],
                                     [1.8700e+01, -2.7085e+01, -1.9028e+00]],
                                    [[2.6398e+01, -1.7530e+01, -1.7879e+00],
                                     [2.6398e+01, -1.7530e+01, -2.9959e-01],
                                     [2.8612e+01, -1.4452e+01, -2.9959e-01],
                                     [2.8612e+01, -1.4452e+01, -1.7879e+00],
                                     [2.7818e+01, -1.8552e+01, -1.7879e+00],
                                     [2.7818e+01, -1.8552e+01, -2.9959e-01],
                                     [3.0032e+01, -1.5474e+01, -2.9959e-01],
                                     [3.0032e+01, -1.5474e+01, -1.7879e+00]]])
    # the pytorch print loses some precision
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

    ComandLine:
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

    expected_tensor = torch.tensor(
        [[
            2.16902434e+01, -4.06038554e-02, -1.61906639e+00, 1.65999997e+00,
            3.20000005e+00, 1.61000001e+00, -1.53999996e+00
        ],
         [
             7.05006905e+00, -6.57459601e+00, -1.60107949e+00, 2.27999997e+00,
             1.27799997e+01, 3.66000009e+00, 1.54999995e+00
         ],
         [
             2.24698818e+01, -6.69203759e+00, -1.50118145e+00, 2.31999993e+00,
             1.47299995e+01, 3.64000010e+00, 1.59000003e+00
         ],
         [
             3.48291965e+01, -7.09058388e+00, -1.36622983e+00, 2.31999993e+00,
             1.00400000e+01, 3.60999990e+00, 1.61000001e+00
         ],
         [
             4.62394617e+01, -7.75838800e+00, -1.32405020e+00, 2.33999991e+00,
             1.28299999e+01, 3.63000011e+00, 1.63999999e+00
         ]],
        dtype=torch.float32)

    rt_mat = rect @ Trv2c
    # test coversion with Box type
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
    np_boxes = np.array(
        [[1.7802081, 2.516249, -1.7501148, 1.75, 3.39, 1.65, 1.48],
         [8.959413, 2.4567227, -1.6357126, 1.54, 4.01, 1.57, 1.62]],
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
    cam_th_boxes = Box3DMode.convert(th_boxes, Box3DMode.LIDAR, Box3DMode.CAM)
    boxes_2 = CameraInstance3DBoxes(cam_th_boxes)
    assert torch.allclose(boxes_2.tensor, cam_th_boxes)

    # test clone/to/device
    boxes_2 = boxes_2.clone()
    boxes_1 = boxes_1.to(boxes_2.device)

    # test box concatenation
    expected_tensor = Box3DMode.convert(
        torch.tensor(
            [[1.7802081, 2.516249, -1.7501148, 1.75, 3.39, 1.65, 1.48],
             [8.959413, 2.4567227, -1.6357126, 1.54, 4.01, 1.57, 1.62],
             [28.2967, -0.5557558, -1.303325, 1.47, 2.23, 1.48, -1.57],
             [26.66902, 21.82302, -1.736057, 1.56, 3.48, 1.4, -1.69],
             [31.31978, 8.162144, -1.6217787, 1.74, 3.77, 1.48, 2.79]]),
        Box3DMode.LIDAR, Box3DMode.CAM)
    boxes = CameraInstance3DBoxes.cat([boxes_1, boxes_2])
    assert torch.allclose(boxes.tensor, expected_tensor)

    # test box flip
    expected_tensor = Box3DMode.convert(
        torch.tensor(
            [[1.7802081, -2.516249, -1.7501148, 1.75, 3.39, 1.65, 1.6615927],
             [8.959413, -2.4567227, -1.6357126, 1.54, 4.01, 1.57, 1.5215927],
             [28.2967, 0.5557558, -1.303325, 1.47, 2.23, 1.48, 4.7115927],
             [26.66902, -21.82302, -1.736057, 1.56, 3.48, 1.4, 4.8315926],
             [31.31978, -8.162144, -1.6217787, 1.74, 3.77, 1.48, 0.35159278]]),
        Box3DMode.LIDAR, Box3DMode.CAM)
    boxes.flip('horizontal')
    assert torch.allclose(boxes.tensor, expected_tensor)

    expected_tensor = torch.tensor(
        [[2.5162, 1.7501, -1.7802, 3.3900, 1.6500, 1.7500, -1.6616],
         [2.4567, 1.6357, -8.9594, 4.0100, 1.5700, 1.5400, -1.5216],
         [-0.5558, 1.3033, -28.2967, 2.2300, 1.4800, 1.4700, -4.7116],
         [21.8230, 1.7361, -26.6690, 3.4800, 1.4000, 1.5600, -4.8316],
         [8.1621, 1.6218, -31.3198, 3.7700, 1.4800, 1.7400, -0.3516]])
    boxes_flip_vert = boxes.clone()
    boxes_flip_vert.flip('vertical')
    assert torch.allclose(boxes_flip_vert.tensor, expected_tensor, 1e-4)

    # test box rotation
    expected_tensor = Box3DMode.convert(
        torch.tensor(
            [[1.0385344, -2.9020846, -1.7501148, 1.75, 3.39, 1.65, 1.9336663],
             [7.969653, -4.774011, -1.6357126, 1.54, 4.01, 1.57, 1.7936664],
             [27.405172, -7.0688415, -1.303325, 1.47, 2.23, 1.48, 4.9836664],
             [19.823532, -28.187025, -1.736057, 1.56, 3.48, 1.4, 5.1036663],
             [27.974297, -16.27845, -1.6217787, 1.74, 3.77, 1.48, 0.6236664]]),
        Box3DMode.LIDAR, Box3DMode.CAM)
    boxes.rotate(torch.tensor(0.27207362796436096))
    assert torch.allclose(boxes.tensor, expected_tensor)

    # test box scaling
    expected_tensor = Box3DMode.convert(
        torch.tensor([[
            1.0443488, -2.9183323, -1.7599131, 1.7597977, 3.4089797, 1.6592377,
            1.9336663
        ],
                      [
                          8.014273, -4.8007393, -1.6448704, 1.5486219,
                          4.0324507, 1.57879, 1.7936664
                      ],
                      [
                          27.558605, -7.1084175, -1.310622, 1.4782301,
                          2.242485, 1.488286, 4.9836664
                      ],
                      [
                          19.934517, -28.344835, -1.7457767, 1.5687338,
                          3.4994833, 1.4078381, 5.1036663
                      ],
                      [
                          28.130915, -16.369587, -1.6308585, 1.7497417,
                          3.791107, 1.488286, 0.6236664
                      ]]), Box3DMode.LIDAR, Box3DMode.CAM)
    boxes.scale(1.00559866335275)
    assert torch.allclose(boxes.tensor, expected_tensor)

    # test box translation
    expected_tensor = Box3DMode.convert(
        torch.tensor([[
            1.1281544, -3.0507944, -1.9169292, 1.7597977, 3.4089797, 1.6592377,
            1.9336663
        ],
                      [
                          8.098079, -4.9332013, -1.8018866, 1.5486219,
                          4.0324507, 1.57879, 1.7936664
                      ],
                      [
                          27.64241, -7.2408795, -1.4676381, 1.4782301,
                          2.242485, 1.488286, 4.9836664
                      ],
                      [
                          20.018322, -28.477297, -1.9027928, 1.5687338,
                          3.4994833, 1.4078381, 5.1036663
                      ],
                      [
                          28.21472, -16.502048, -1.7878747, 1.7497417,
                          3.791107, 1.488286, 0.6236664
                      ]]), Box3DMode.LIDAR, Box3DMode.CAM)
    boxes.translate(torch.tensor([0.13246193, 0.15701613, 0.0838056]))
    assert torch.allclose(boxes.tensor, expected_tensor)

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
    expected_tesor = boxes.tensor.clone()
    assert torch.allclose(expected_tesor, boxes.tensor)

    boxes.flip()
    boxes.flip()
    boxes.limit_yaw()
    assert torch.allclose(expected_tesor, boxes.tensor)

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
    # the pytorch print loses some precision
    assert torch.allclose(
        boxes.nearest_bev, expected_tensor, rtol=1e-4, atol=1e-7)

    # obtained by the print of the original implementation
    expected_tensor = torch.tensor([[[3.2684e+00, 2.5769e-01, -7.7767e-01],
                                     [1.6232e+00, 2.5769e-01, -1.5301e-01],
                                     [1.6232e+00, 1.9169e+00, -1.5301e-01],
                                     [3.2684e+00, 1.9169e+00, -7.7767e-01],
                                     [4.4784e+00, 2.5769e-01, 2.4093e+00],
                                     [2.8332e+00, 2.5769e-01, 3.0340e+00],
                                     [2.8332e+00, 1.9169e+00, 3.0340e+00],
                                     [4.4784e+00, 1.9169e+00, 2.4093e+00]],
                                    [[5.2427e+00, 2.2310e-01, 5.9606e+00],
                                     [3.7324e+00, 2.2310e-01, 6.3029e+00],
                                     [3.7324e+00, 1.8019e+00, 6.3029e+00],
                                     [5.2427e+00, 1.8019e+00, 5.9606e+00],
                                     [6.1340e+00, 2.2310e-01, 9.8933e+00],
                                     [4.6237e+00, 2.2310e-01, 1.0236e+01],
                                     [4.6237e+00, 1.8019e+00, 1.0236e+01],
                                     [6.1340e+00, 1.8019e+00, 9.8933e+00]],
                                    [[7.6525e+00, -2.0648e-02, 2.6364e+01],
                                     [6.2283e+00, -2.0648e-02, 2.6760e+01],
                                     [6.2283e+00, 1.4676e+00, 2.6760e+01],
                                     [7.6525e+00, 1.4676e+00, 2.6364e+01],
                                     [8.2534e+00, -2.0648e-02, 2.8525e+01],
                                     [6.8292e+00, -2.0648e-02, 2.8921e+01],
                                     [6.8292e+00, 1.4676e+00, 2.8921e+01],
                                     [8.2534e+00, 1.4676e+00, 2.8525e+01]],
                                    [[2.8535e+01, 4.9495e-01, 1.8102e+01],
                                     [2.7085e+01, 4.9495e-01, 1.8700e+01],
                                     [2.7085e+01, 1.9028e+00, 1.8700e+01],
                                     [2.8535e+01, 1.9028e+00, 1.8102e+01],
                                     [2.9870e+01, 4.9495e-01, 2.1337e+01],
                                     [2.8420e+01, 4.9495e-01, 2.1935e+01],
                                     [2.8420e+01, 1.9028e+00, 2.1935e+01],
                                     [2.9870e+01, 1.9028e+00, 2.1337e+01]],
                                    [[1.4452e+01, 2.9959e-01, 2.8612e+01],
                                     [1.5474e+01, 2.9959e-01, 3.0032e+01],
                                     [1.5474e+01, 1.7879e+00, 3.0032e+01],
                                     [1.4452e+01, 1.7879e+00, 2.8612e+01],
                                     [1.7530e+01, 2.9959e-01, 2.6398e+01],
                                     [1.8552e+01, 2.9959e-01, 2.7818e+01],
                                     [1.8552e+01, 1.7879e+00, 2.7818e+01],
                                     [1.7530e+01, 1.7879e+00, 2.6398e+01]]])

    # the pytorch print loses some precision
    assert torch.allclose(boxes.corners, expected_tensor, rtol=1e-4, atol=1e-7)


def test_boxes3d_overlaps():
    """Test the iou calculation of boxes in different modes.

    ComandLine:
        xdoctest tests/test_box3d.py::test_boxes3d_overlaps zero
    """
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')

    # Test LiDAR boxes 3D overlaps
    boxes1_tensor = torch.tensor(
        [[1.8, -2.5, -1.8, 1.75, 3.39, 1.65, 1.6615927],
         [8.9, -2.5, -1.6, 1.54, 4.01, 1.57, 1.5215927],
         [28.3, 0.5, -1.3, 1.47, 2.23, 1.48, 4.7115927],
         [31.3, -8.2, -1.6, 1.74, 3.77, 1.48, 0.35]],
        device='cuda')
    boxes1 = LiDARInstance3DBoxes(boxes1_tensor)

    boxes2_tensor = torch.tensor([[1.2, -3.0, -1.9, 1.8, 3.4, 1.7, 1.9],
                                  [8.1, -2.9, -1.8, 1.5, 4.1, 1.6, 1.8],
                                  [31.3, -8.2, -1.6, 1.74, 3.77, 1.48, 0.35],
                                  [20.1, -28.5, -1.9, 1.6, 3.5, 1.4, 5.1]],
                                 device='cuda')
    boxes2 = LiDARInstance3DBoxes(boxes2_tensor)

    expected_tensor = torch.tensor(
        [[0.3710, 0.0000, 0.0000, 0.0000], [0.0000, 0.3322, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 1.0000, 0.0000]],
        device='cuda')
    overlaps_3d = boxes1.overlaps(boxes1, boxes2)
    assert torch.allclose(expected_tensor, overlaps_3d, rtol=1e-4, atol=1e-7)

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
        expected_tensor, cam_overlaps_3d, rtol=1e-4, atol=1e-7)
    assert torch.allclose(cam_overlaps_3d, overlaps_3d)

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
         [2.3262, 3.3065, --0.44255, 0.8234, 0.5325, 1.0099, 2.9971],
         [2.4593, 2.5870, -0.4321, 0.8597, 0.6193, 1.0204, 3.0693],
         [1.4856, 2.5299, -0.5570, 0.9385, 2.1404, 0.8954, 3.0601]])
    boxes = DepthInstance3DBoxes.cat([boxes_1, boxes_2])
    assert torch.allclose(boxes.tensor, expected_tensor)
    # concatenate empty list
    empty_boxes = DepthInstance3DBoxes.cat([])
    assert empty_boxes.tensor.shape[0] == 0
    assert empty_boxes.tensor.shape[-1] == 7

    # test box flip
    expected_tensor = torch.tensor(
        [[-1.4856, 2.5299, -0.5570, 0.9385, 2.1404, 0.8954, 0.0815],
         [-2.3262, 3.3065, 0.4426, 0.8234, 0.5325, 1.0099, 0.1445],
         [-2.4593, 2.5870, -0.4321, 0.8597, 0.6193, 1.0204, 0.0723],
         [-1.4856, 2.5299, -0.5570, 0.9385, 2.1404, 0.8954, 0.0815]])
    boxes.flip(bev_direction='horizontal')
    assert torch.allclose(boxes.tensor, expected_tensor, 1e-3)
    expected_tensor = torch.tensor(
        [[-1.4856, -2.5299, -0.5570, 0.9385, 2.1404, 0.8954, -0.0815],
         [-2.3262, -3.3065, 0.4426, 0.8234, 0.5325, 1.0099, -0.1445],
         [-2.4593, -2.5870, -0.4321, 0.8597, 0.6193, 1.0204, -0.0723],
         [-1.4856, -2.5299, -0.5570, 0.9385, 2.1404, 0.8954, -0.0815]])
    boxes.flip(bev_direction='vertical')
    assert torch.allclose(boxes.tensor, expected_tensor, 1e-3)

    # test box rotation
    boxes_rot = boxes.clone()
    expected_tensor = torch.tensor(
        [[-1.6004, -2.4589, -0.5570, 0.9385, 2.1404, 0.8954, -0.0355],
         [-2.4758, -3.1960, 0.4426, 0.8234, 0.5325, 1.0099, -0.0985],
         [-2.5757, -2.4712, -0.4321, 0.8597, 0.6193, 1.0204, -0.0263],
         [-1.6004, -2.4589, -0.5570, 0.9385, 2.1404, 0.8954, -0.0355]])
    boxes_rot.rotate(-0.04599790655000615)
    assert torch.allclose(boxes_rot.tensor, expected_tensor, 1e-3)

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
    torch.allclose(boxes.corners, expected_tensor)
