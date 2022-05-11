from unittest import TestCase

import numpy as np
import pytest
import torch
# TODO: will use real PixelData once it is added in mmengine
from mmengine.data import BaseDataElement as PixelData
from mmengine.data import InstanceData

from mmdet3d.core import Det3DDataSample


def _equal(a, b):
    if isinstance(a, (torch.Tensor, np.ndarray)):
        return (a == b).all()
    else:
        return a == b


class TestDetDataSample(TestCase):

    def test_init(self):
        meta_info = dict(
            img_size=[256, 256],
            scale_factor=np.array([1.5, 1.5]),
            img_shape=torch.rand(4))

        det3d_data_sample = Det3DDataSample(metainfo=meta_info)
        assert 'img_size' in det3d_data_sample
        assert det3d_data_sample.img_size == [256, 256]
        assert det3d_data_sample.get('img_size') == [256, 256]

    def test_setter(self):
        det3d_data_sample = Det3DDataSample()
        # test gt_instances_3d
        gt_instances_3d_data = dict(
            bboxes=torch.rand(4, 4),
            labels=torch.rand(4),
            masks=np.random.rand(4, 2, 2))
        gt_instances_3d = InstanceData(**gt_instances_3d_data)
        det3d_data_sample.gt_instances_3d = gt_instances_3d
        assert 'gt_instances_3d' in det3d_data_sample
        assert _equal(det3d_data_sample.gt_instances_3d.bboxes,
                      gt_instances_3d_data['bboxes'])
        assert _equal(det3d_data_sample.gt_instances_3d.labels,
                      gt_instances_3d_data['labels'])
        assert _equal(det3d_data_sample.gt_instances_3d.masks,
                      gt_instances_3d_data['masks'])

        # test pred_instances
        pred_instances_3d_data = dict(
            bboxes=torch.rand(2, 4),
            labels=torch.rand(2),
            masks=np.random.rand(2, 2, 2))
        pred_instances_3d = InstanceData(**pred_instances_3d_data)
        det3d_data_sample.pred_instances_3d = pred_instances_3d
        assert 'pred_instances_3d' in det3d_data_sample
        assert _equal(det3d_data_sample.pred_instances_3d.bboxes,
                      pred_instances_3d_data['bboxes'])
        assert _equal(det3d_data_sample.pred_instances_3d.labels,
                      pred_instances_3d_data['labels'])
        assert _equal(det3d_data_sample.pred_instances_3d.masks,
                      pred_instances_3d_data['masks'])

        # test proposals
        proposals_data = dict(bboxes=torch.rand(4, 4), labels=torch.rand(4))
        proposals = InstanceData(**proposals_data)
        det3d_data_sample.proposals = proposals
        assert 'proposals' in det3d_data_sample
        assert _equal(det3d_data_sample.proposals.bboxes,
                      proposals_data['bboxes'])
        assert _equal(det3d_data_sample.proposals.labels,
                      proposals_data['labels'])

        # test ignored_instances
        ignored_instances_data = dict(
            bboxes=torch.rand(4, 4), labels=torch.rand(4))
        ignored_instances = InstanceData(**ignored_instances_data)
        det3d_data_sample.ignored_instances = ignored_instances
        assert 'ignored_instances' in det3d_data_sample
        assert _equal(det3d_data_sample.ignored_instances.bboxes,
                      ignored_instances_data['bboxes'])
        assert _equal(det3d_data_sample.ignored_instances.labels,
                      ignored_instances_data['labels'])

        # test gt_panoptic_seg
        gt_pts_panoptic_seg_data = dict(panoptic_seg=torch.rand(5, 4))
        gt_pts_panoptic_seg = PixelData(**gt_pts_panoptic_seg_data)
        det3d_data_sample.gt_pts_panoptic_seg = gt_pts_panoptic_seg
        assert 'gt_pts_panoptic_seg' in det3d_data_sample
        assert _equal(det3d_data_sample.gt_pts_panoptic_seg.panoptic_seg,
                      gt_pts_panoptic_seg_data['panoptic_seg'])

        # test pred_panoptic_seg
        pred_pts_panoptic_seg_data = dict(panoptic_seg=torch.rand(5, 4))
        pred_pts_panoptic_seg = PixelData(**pred_pts_panoptic_seg_data)
        det3d_data_sample.pred_pts_panoptic_seg = pred_pts_panoptic_seg
        assert 'pred_pts_panoptic_seg' in det3d_data_sample
        assert _equal(det3d_data_sample.pred_pts_panoptic_seg.panoptic_seg,
                      pred_pts_panoptic_seg_data['panoptic_seg'])

        # test gt_sem_seg
        gt_pts_sem_seg_data = dict(segm_seg=torch.rand(5, 4, 2))
        gt_pts_sem_seg = PixelData(**gt_pts_sem_seg_data)
        det3d_data_sample.gt_pts_sem_seg = gt_pts_sem_seg
        assert 'gt_pts_sem_seg' in det3d_data_sample
        assert _equal(det3d_data_sample.gt_pts_sem_seg.segm_seg,
                      gt_pts_sem_seg_data['segm_seg'])

        # test pred_segm_seg
        pred_pts_sem_seg_data = dict(segm_seg=torch.rand(5, 4, 2))
        pred_pts_sem_seg = PixelData(**pred_pts_sem_seg_data)
        det3d_data_sample.pred_pts_sem_seg = pred_pts_sem_seg
        assert 'pred_pts_sem_seg' in det3d_data_sample
        assert _equal(det3d_data_sample.pred_pts_sem_seg.segm_seg,
                      pred_pts_sem_seg_data['segm_seg'])

        # test type error
        with pytest.raises(AssertionError):
            det3d_data_sample.pred_instances_3d = torch.rand(2, 4)

        with pytest.raises(AssertionError):
            det3d_data_sample.pred_pts_panoptic_seg = torch.rand(2, 4)

        with pytest.raises(AssertionError):
            det3d_data_sample.pred_pts_sem_seg = torch.rand(2, 4)

    def test_deleter(self):
        gt_instances_3d_data = dict(
            bboxes=torch.rand(4, 4),
            labels=torch.rand(4),
            masks=np.random.rand(4, 2, 2))

        det3d_data_sample = Det3DDataSample()
        gt_instances_3d = InstanceData(data=gt_instances_3d_data)
        det3d_data_sample.gt_instances_3d = gt_instances_3d
        assert 'gt_instances_3d' in det3d_data_sample
        del det3d_data_sample.gt_instances_3d
        assert 'gt_instances_3d' not in det3d_data_sample

        pred_pts_panoptic_seg_data = torch.rand(5, 4)
        pred_pts_panoptic_seg_data = PixelData(data=pred_pts_panoptic_seg_data)
        det3d_data_sample.pred_pts_panoptic_seg_data = \
            pred_pts_panoptic_seg_data
        assert 'pred_pts_panoptic_seg_data' in det3d_data_sample
        del det3d_data_sample.pred_pts_panoptic_seg_data
        assert 'pred_pts_panoptic_seg_data' not in det3d_data_sample

        pred_pts_sem_seg_data = dict(segm_seg=torch.rand(5, 4, 2))
        pred_pts_sem_seg = PixelData(**pred_pts_sem_seg_data)
        det3d_data_sample.pred_pts_sem_seg = pred_pts_sem_seg
        assert 'pred_pts_sem_seg' in det3d_data_sample
        del det3d_data_sample.pred_pts_sem_seg
        assert 'pred_pts_sem_seg' not in det3d_data_sample
