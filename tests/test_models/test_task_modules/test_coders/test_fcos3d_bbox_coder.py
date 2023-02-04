# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import Scale
from torch import nn as nn

from mmdet3d.registry import TASK_UTILS


def test_fcos3d_bbox_coder():
    # test a config without priors
    bbox_coder_cfg = dict(
        type='FCOS3DBBoxCoder',
        base_depths=None,
        base_dims=None,
        code_size=7,
        norm_on_bbox=True)
    bbox_coder = TASK_UTILS.build(bbox_coder_cfg)

    # test decode
    # [2, 7, 1, 1]
    batch_bbox = torch.tensor([[[[0.3130]], [[0.7094]], [[0.8743]], [[0.0570]],
                                [[0.5579]], [[0.1593]], [[0.4553]]],
                               [[[0.7758]], [[0.2298]], [[0.3925]], [[0.6307]],
                                [[0.4377]], [[0.3339]], [[0.1966]]]])
    batch_scale = nn.ModuleList([Scale(1.0) for _ in range(3)])
    stride = 2
    training = False
    cls_score = torch.randn([2, 2, 1, 1]).sigmoid()
    decode_bbox = bbox_coder.decode(batch_bbox, batch_scale, stride, training,
                                    cls_score)

    expected_bbox = torch.tensor([[[[0.6261]], [[1.4188]], [[2.3971]],
                                   [[1.0586]], [[1.7470]], [[1.1727]],
                                   [[0.4553]]],
                                  [[[1.5516]], [[0.4596]], [[1.4806]],
                                   [[1.8790]], [[1.5492]], [[1.3965]],
                                   [[0.1966]]]])
    assert torch.allclose(decode_bbox, expected_bbox, atol=1e-3)

    # test a config with priors
    prior_bbox_coder_cfg = dict(
        type='FCOS3DBBoxCoder',
        base_depths=((28., 13.), (25., 12.)),
        base_dims=((2., 3., 1.), (1., 2., 3.)),
        code_size=7,
        norm_on_bbox=True)
    prior_bbox_coder = TASK_UTILS.build(prior_bbox_coder_cfg)

    # test decode
    batch_bbox = torch.tensor([[[[0.3130]], [[0.7094]], [[0.8743]], [[0.0570]],
                                [[0.5579]], [[0.1593]], [[0.4553]]],
                               [[[0.7758]], [[0.2298]], [[0.3925]], [[0.6307]],
                                [[0.4377]], [[0.3339]], [[0.1966]]]])
    batch_scale = nn.ModuleList([Scale(1.0) for _ in range(3)])
    stride = 2
    training = False
    cls_score = torch.tensor([[[[0.5811]], [[0.6198]]], [[[0.4889]],
                                                         [[0.8142]]]])
    decode_bbox = prior_bbox_coder.decode(batch_bbox, batch_scale, stride,
                                          training, cls_score)
    expected_bbox = torch.tensor([[[[0.6260]], [[1.4188]], [[35.4916]],
                                   [[1.0587]], [[3.4940]], [[3.5181]],
                                   [[0.4553]]],
                                  [[[1.5516]], [[0.4596]], [[29.7100]],
                                   [[1.8789]], [[3.0983]], [[4.1892]],
                                   [[0.1966]]]])
    assert torch.allclose(decode_bbox, expected_bbox, atol=1e-3)

    # test decode_yaw
    decode_bbox = decode_bbox.permute(0, 2, 3, 1).view(-1, 7)
    batch_centers2d = torch.tensor([[100., 150.], [200., 100.]])
    batch_dir_cls = torch.tensor([0., 1.])
    dir_offset = 0.7854
    cam2img = torch.tensor([[700., 0., 450., 0.], [0., 700., 200., 0.],
                            [0., 0., 1., 0.], [0., 0., 0., 1.]])
    decode_bbox = prior_bbox_coder.decode_yaw(decode_bbox, batch_centers2d,
                                              batch_dir_cls, dir_offset,
                                              cam2img)
    expected_bbox = torch.tensor(
        [[0.6260, 1.4188, 35.4916, 1.0587, 3.4940, 3.5181, 3.1332],
         [1.5516, 0.4596, 29.7100, 1.8789, 3.0983, 4.1892, 6.1368]])
    assert torch.allclose(decode_bbox, expected_bbox, atol=1e-3)
