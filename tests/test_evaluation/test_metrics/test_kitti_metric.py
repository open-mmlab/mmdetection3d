import numpy as np
import pytest
import torch
from mmengine.structures import InstanceData

from mmdet3d.evaluation.metrics import KittiMetric
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes

data_root = 'tests/data/kitti'


def _init_evaluate_input():
    metainfo = dict(sample_idx=0)
    predictions = Det3DDataSample()
    pred_instances_3d = InstanceData()
    pred_instances_3d.bboxes_3d = LiDARInstance3DBoxes(
        torch.tensor(
            [[8.7314, -1.8559, -1.5997, 0.4800, 1.2000, 1.8900, 0.0100]]))
    pred_instances_3d.scores_3d = torch.Tensor([0.9])
    pred_instances_3d.labels_3d = torch.Tensor([0])

    predictions.pred_instances_3d = pred_instances_3d
    predictions.pred_instances = InstanceData()
    predictions.set_metainfo(metainfo)
    predictions = predictions.to_dict()
    return {}, [predictions]


def _init_multi_modal_evaluate_input():
    metainfo = dict(sample_idx=0)
    predictions = Det3DDataSample()
    pred_instances_3d = InstanceData()
    pred_instances = InstanceData()
    pred_instances.bboxes = torch.tensor([[712.4, 143, 810.7, 307.92]])
    pred_instances.scores = torch.Tensor([0.9])
    pred_instances.labels = torch.Tensor([0])
    pred_instances_3d.bboxes_3d = LiDARInstance3DBoxes(
        torch.tensor(
            [[8.7314, -1.8559, -1.5997, 0.4800, 1.2000, 1.8900, 0.0100]]))

    pred_instances_3d.scores_3d = torch.Tensor([0.9])
    pred_instances_3d.labels_3d = torch.Tensor([0])

    predictions.pred_instances_3d = pred_instances_3d
    predictions.pred_instances = pred_instances
    predictions.set_metainfo(metainfo)
    predictions = predictions.to_dict()
    return {}, [predictions]


def test_multi_modal_kitti_metric():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    kittimetric = KittiMetric(
        data_root + '/kitti_infos_train.pkl', metric=['mAP'])
    kittimetric.dataset_meta = dict(classes=['Pedestrian', 'Cyclist', 'Car'])
    data_batch, predictions = _init_multi_modal_evaluate_input()
    kittimetric.process(data_batch, predictions)
    ap_dict = kittimetric.compute_metrics(kittimetric.results)
    assert np.isclose(ap_dict['pred_instances_3d/KITTI/Overall_3D_AP11_easy'],
                      3.0303030303030307)
    assert np.isclose(ap_dict['pred_instances_3d/KITTI/Overall_BEV_AP11_easy'],
                      3.0303030303030307)
    assert np.isclose(ap_dict['pred_instances_3d/KITTI/Overall_2D_AP11_easy'],
                      3.0303030303030307)
    assert np.isclose(ap_dict['pred_instances/KITTI/Overall_2D_AP11_easy'],
                      3.0303030303030307)
    assert np.isclose(ap_dict['pred_instances/KITTI/Overall_2D_AP11_moderate'],
                      3.0303030303030307)
    assert np.isclose(ap_dict['pred_instances/KITTI/Overall_2D_AP11_hard'],
                      3.0303030303030307)


def test_kitti_metric_mAP():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    kittimetric = KittiMetric(
        data_root + '/kitti_infos_train.pkl', metric=['mAP'])
    kittimetric.dataset_meta = dict(classes=['Pedestrian', 'Cyclist', 'Car'])
    data_batch, predictions = _init_evaluate_input()
    kittimetric.process(data_batch, predictions)
    ap_dict = kittimetric.compute_metrics(kittimetric.results)
    assert np.isclose(ap_dict['pred_instances_3d/KITTI/Overall_3D_AP11_easy'],
                      3.0303030303030307)
    assert np.isclose(
        ap_dict['pred_instances_3d/KITTI/Overall_3D_AP11_moderate'],
        3.0303030303030307)
    assert np.isclose(ap_dict['pred_instances_3d/KITTI/Overall_3D_AP11_hard'],
                      3.0303030303030307)
