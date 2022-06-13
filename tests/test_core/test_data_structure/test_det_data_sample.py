from unittest import TestCase

import numpy as np
import pytest
import torch
from mmengine.data import InstanceData, PixelData

from mmdet3d.core.data_structures import Det3DDataSample


def _equal(a, b):
    if isinstance(a, (torch.Tensor, np.ndarray)):
        return (a == b).all()
    else:
        return a == b


class TestDet3DataSample(TestCase):

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
            bboxes_3d=torch.rand(4, 4), labels_3d=torch.rand(4))
        gt_instances_3d = InstanceData(**gt_instances_3d_data)
        det3d_data_sample.gt_instances_3d = gt_instances_3d
        assert 'gt_instances_3d' in det3d_data_sample
        assert _equal(det3d_data_sample.gt_instances_3d.bboxes_3d,
                      gt_instances_3d_data['bboxes_3d'])
        assert _equal(det3d_data_sample.gt_instances_3d.labels_3d,
                      gt_instances_3d_data['labels_3d'])

        # test pred_instances_3d
        pred_instances_3d_data = dict(
            bboxes_3d=torch.rand(2, 4),
            labels_3d=torch.rand(2),
            scores_3d=torch.rand(2))
        pred_instances_3d = InstanceData(**pred_instances_3d_data)
        det3d_data_sample.pred_instances_3d = pred_instances_3d
        assert 'pred_instances_3d' in det3d_data_sample
        assert _equal(det3d_data_sample.pred_instances_3d.bboxes_3d,
                      pred_instances_3d_data['bboxes_3d'])
        assert _equal(det3d_data_sample.pred_instances_3d.labels_3d,
                      pred_instances_3d_data['labels_3d'])
        assert _equal(det3d_data_sample.pred_instances_3d.scores_3d,
                      pred_instances_3d_data['scores_3d'])

        # test pts_pred_instances_3d
        pts_pred_instances_3d_data = dict(
            bboxes_3d=torch.rand(2, 4),
            labels_3d=torch.rand(2),
            scores_3d=torch.rand(2))
        pts_pred_instances_3d = InstanceData(**pts_pred_instances_3d_data)
        det3d_data_sample.pts_pred_instances_3d = pts_pred_instances_3d
        assert 'pts_pred_instances_3d' in det3d_data_sample
        assert _equal(det3d_data_sample.pts_pred_instances_3d.bboxes_3d,
                      pts_pred_instances_3d_data['bboxes_3d'])
        assert _equal(det3d_data_sample.pts_pred_instances_3d.labels_3d,
                      pts_pred_instances_3d_data['labels_3d'])
        assert _equal(det3d_data_sample.pts_pred_instances_3d.scores_3d,
                      pts_pred_instances_3d_data['scores_3d'])

        # test img_pred_instances_3d
        img_pred_instances_3d_data = dict(
            bboxes_3d=torch.rand(2, 4),
            labels_3d=torch.rand(2),
            scores_3d=torch.rand(2))
        img_pred_instances_3d = InstanceData(**img_pred_instances_3d_data)
        det3d_data_sample.img_pred_instances_3d = img_pred_instances_3d
        assert 'img_pred_instances_3d' in det3d_data_sample
        assert _equal(det3d_data_sample.img_pred_instances_3d.bboxes_3d,
                      img_pred_instances_3d_data['bboxes_3d'])
        assert _equal(det3d_data_sample.img_pred_instances_3d.labels_3d,
                      img_pred_instances_3d_data['labels_3d'])
        assert _equal(det3d_data_sample.img_pred_instances_3d.scores_3d,
                      img_pred_instances_3d_data['scores_3d'])

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
        tmp_instances_3d_data = dict(
            bboxes_3d=torch.rand(4, 4), labels_3d=torch.rand(4))

        det3d_data_sample = Det3DDataSample()
        gt_instances_3d = InstanceData(data=tmp_instances_3d_data)
        det3d_data_sample.gt_instances_3d = gt_instances_3d
        assert 'gt_instances_3d' in det3d_data_sample
        del det3d_data_sample.gt_instances_3d
        assert 'gt_instances_3d' not in det3d_data_sample

        pred_instances_3d = InstanceData(data=tmp_instances_3d_data)
        det3d_data_sample.pred_instances_3d = pred_instances_3d
        assert 'pred_instances_3d' in det3d_data_sample
        del det3d_data_sample.pred_instances_3d
        assert 'pred_instances_3d' not in det3d_data_sample

        pts_pred_instances_3d = InstanceData(data=tmp_instances_3d_data)
        det3d_data_sample.pts_pred_instances_3d = pts_pred_instances_3d
        assert 'pts_pred_instances_3d' in det3d_data_sample
        del det3d_data_sample.pts_pred_instances_3d
        assert 'pts_pred_instances_3d' not in det3d_data_sample

        img_pred_instances_3d = InstanceData(data=tmp_instances_3d_data)
        det3d_data_sample.img_pred_instances_3d = img_pred_instances_3d
        assert 'img_pred_instances_3d' in det3d_data_sample
        del det3d_data_sample.img_pred_instances_3d
        assert 'img_pred_instances_3d' not in det3d_data_sample

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
