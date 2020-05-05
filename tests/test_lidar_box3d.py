import numpy as np
import torch

from mmdet3d.core.bbox import LiDARInstance3DBoxes


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
