# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmdet3d.core.bbox.assigners import MaxIoUAssigner
from mmdet3d.core.bbox.samplers import IoUNegPiecewiseSampler


def test_iou_piecewise_sampler():
    if not torch.cuda.is_available():
        pytest.skip()
    assigner = MaxIoUAssigner(
        pos_iou_thr=0.55,
        neg_iou_thr=0.55,
        min_pos_iou=0.55,
        ignore_iof_thr=-1,
        iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'))
    bboxes = torch.tensor(
        [[32, 32, 16, 8, 38, 42, -0.3], [32, 32, 16, 8, 38, 42, -0.3],
         [32, 32, 16, 8, 38, 42, -0.3], [32, 32, 16, 8, 38, 42, -0.3],
         [0, 0, 0, 10, 10, 10, 0.2], [10, 10, 10, 20, 20, 15, 0.6],
         [5, 5, 5, 15, 15, 15, 0.7], [5, 5, 5, 15, 15, 15, 0.7],
         [5, 5, 5, 15, 15, 15, 0.7], [32, 32, 16, 8, 38, 42, -0.3],
         [32, 32, 16, 8, 38, 42, -0.3], [32, 32, 16, 8, 38, 42, -0.3]],
        dtype=torch.float32).cuda()
    gt_bboxes = torch.tensor(
        [[0, 0, 0, 10, 10, 9, 0.2], [5, 10, 10, 20, 20, 15, 0.6]],
        dtype=torch.float32).cuda()
    gt_labels = torch.tensor([1, 1], dtype=torch.int64).cuda()
    assign_result = assigner.assign(bboxes, gt_bboxes, gt_labels=gt_labels)

    sampler = IoUNegPiecewiseSampler(
        num=10,
        pos_fraction=0.55,
        neg_piece_fractions=[0.8, 0.2],
        neg_iou_piece_thrs=[0.55, 0.1],
        neg_pos_ub=-1,
        add_gt_as_proposals=False)

    sample_result = sampler.sample(assign_result, bboxes, gt_bboxes, gt_labels)

    assert sample_result.pos_inds == 4
    assert len(sample_result.pos_bboxes) == len(sample_result.pos_inds)
    assert len(sample_result.neg_bboxes) == len(sample_result.neg_inds)
