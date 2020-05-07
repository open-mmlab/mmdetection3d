import numpy as np
import torch

from mmdet3d.core.bbox import Box3DMode, LiDARInstance3DBoxes


def test_lidar_boxes3d():
    # Test init with numpy array
    np_boxes = np.array(
        [[1.7802081, 2.516249, -1.7501148, 1.75, 3.39, 1.65, 1.48],
         [8.959413, 2.4567227, -1.6357126, 1.54, 4.01, 1.57, 1.62]],
        dtype=np.float32)
    boxes_1 = LiDARInstance3DBoxes(np_boxes)
    assert torch.allclose(boxes_1.tensor, torch.from_numpy(np_boxes))

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

    # test box flip
    expected_tensor = torch.tensor(
        [[1.7802081, -2.516249, -1.7501148, 1.75, 3.39, 1.65, 1.6615927],
         [8.959413, -2.4567227, -1.6357126, 1.54, 4.01, 1.57, 1.5215927],
         [28.2967, 0.5557558, -1.303325, 1.47, 2.23, 1.48, 4.7115927],
         [26.66902, -21.82302, -1.736057, 1.56, 3.48, 1.4, 4.8315926],
         [31.31978, -8.162144, -1.6217787, 1.74, 3.77, 1.48, 0.35159278]])
    boxes.flip()
    assert torch.allclose(boxes.tensor, expected_tensor)

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
        boxes.nearset_bev, expected_tensor, rtol=1e-4, atol=1e-7)

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
