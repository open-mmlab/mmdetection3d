# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet3d.registry import TASK_UTILS
from mmdet3d.structures import CameraInstance3DBoxes


def test_smoke_bbox_coder():
    bbox_coder_cfg = dict(
        type='SMOKECoder',
        base_depth=(28.01, 16.32),
        base_dims=((3.88, 1.63, 1.53), (1.78, 1.70, 0.58), (0.88, 1.73, 0.67)),
        code_size=7)

    bbox_coder = TASK_UTILS.build(bbox_coder_cfg)
    regression = torch.rand([200, 8])
    points = torch.rand([200, 2])
    labels = torch.ones([2, 100])
    cam2imgs = torch.rand([2, 4, 4])
    trans_mats = torch.rand([2, 3, 3])

    img_metas = [dict(box_type_3d=CameraInstance3DBoxes) for i in range(2)]
    locations, dimensions, orientations = bbox_coder.decode(
        regression, points, labels, cam2imgs, trans_mats)
    assert locations.shape == torch.Size([200, 3])
    assert dimensions.shape == torch.Size([200, 3])
    assert orientations.shape == torch.Size([200, 1])
    bboxes = bbox_coder.encode(locations, dimensions, orientations, img_metas)
    assert bboxes.tensor.shape == torch.Size([200, 7])

    # specically designed to test orientation decode function's
    # special cases.
    ori_vector = torch.tensor([[-0.9, -0.01], [-0.9, 0.01]])
    locations = torch.tensor([[15., 2., 1.], [15., 2., -1.]])
    orientations = bbox_coder._decode_orientation(ori_vector, locations)
    assert orientations.shape == torch.Size([2, 1])
