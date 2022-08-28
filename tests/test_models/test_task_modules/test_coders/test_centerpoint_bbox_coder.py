# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet3d.registry import TASK_UTILS


def test_centerpoint_bbox_coder():
    bbox_coder_cfg = dict(
        type='CenterPointBBoxCoder',
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        max_num=500,
        score_threshold=0.1,
        pc_range=[-51.2, -51.2],
        out_size_factor=4,
        voxel_size=[0.2, 0.2])

    bbox_coder = TASK_UTILS.build(bbox_coder_cfg)

    batch_dim = torch.rand([2, 3, 128, 128])
    batch_hei = torch.rand([2, 1, 128, 128])
    batch_hm = torch.rand([2, 2, 128, 128])
    batch_reg = torch.rand([2, 2, 128, 128])
    batch_rotc = torch.rand([2, 1, 128, 128])
    batch_rots = torch.rand([2, 1, 128, 128])
    batch_vel = torch.rand([2, 2, 128, 128])

    temp = bbox_coder.decode(batch_hm, batch_rots, batch_rotc, batch_hei,
                             batch_dim, batch_vel, batch_reg, 5)
    for i in range(len(temp)):
        assert temp[i]['bboxes'].shape == torch.Size([500, 9])
        assert temp[i]['scores'].shape == torch.Size([500])
        assert temp[i]['labels'].shape == torch.Size([500])
