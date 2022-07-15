# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmengine.data import InstanceData

from mmdet3d.core import Det3DDataSample
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.registry import MODELS
from tests.utils.model_utils import _get_detector_cfg, _setup_seed


def test_voxel_net():
    import mmdet3d.models
    assert hasattr(mmdet3d.models, 'VoxelNet')
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    _setup_seed(0)
    voxel_net_cfg = _get_detector_cfg(
        'pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py')
    model = MODELS.build(voxel_net_cfg).cuda()
    input_dict0 = dict(points=torch.rand([2010, 4], device='cuda'))
    input_dict1 = dict(points=torch.rand([2020, 4], device='cuda'))
    gt_instance_3d_0 = InstanceData()
    gt_instance_3d_0.bboxes_3d = LiDARInstance3DBoxes(
        torch.rand([20, 7], device='cuda'))
    gt_instance_3d_0.labels_3d = torch.randint(0, 3, [20], device='cuda')
    data_sample_0 = Det3DDataSample(
        metainfo=dict(box_type_3d=LiDARInstance3DBoxes))
    data_sample_0.gt_instances_3d = gt_instance_3d_0

    gt_instance_3d_1 = InstanceData()
    gt_instance_3d_1.bboxes_3d = LiDARInstance3DBoxes(
        torch.rand([50, 7], device='cuda'))
    gt_instance_3d_1.labels_3d = torch.randint(0, 3, [50], device='cuda')
    data_sample_1 = Det3DDataSample(
        metainfo=dict(box_type_3d=LiDARInstance3DBoxes))
    data_sample_1.gt_instances_3d = gt_instance_3d_1
    data = [dict(inputs=input_dict0, data_sample=data_sample_0)]

    # test simple_test
    # TODO FIX this UT
    pytest.skip('FIX this @shenkun')

    with torch.no_grad():
        results = model.forward(data, return_loss=False)
    bboxes_3d = results[0].pred_instances_3d['bboxes_3d']
    scores_3d = results[0].pred_instances_3d['scores_3d']
    labels_3d = results[0].pred_instances_3d['labels_3d']
    assert bboxes_3d.tensor.shape == (50, 7)
    assert scores_3d.shape == torch.Size([50])
    assert labels_3d.shape == torch.Size([50])

    # test forward_train
    data = [
        dict(inputs=input_dict0, data_sample=data_sample_0),
        dict(inputs=input_dict1, data_sample=data_sample_1)
    ]
    losses = model.forward(data, return_loss=True)
    assert losses['log_vars']['loss_cls'] >= 0
    assert losses['log_vars']['loss_bbox'] >= 0
    assert losses['log_vars']['loss_dir'] >= 0
    assert losses['log_vars']['loss'] >= 0

    # test_aug_test
    metainfo = {
        'pcd_scale_factor': 1,
        'pcd_horizontal_flip': 1,
        'pcd_vertical_flip': 1,
        'box_type_3d': LiDARInstance3DBoxes
    }
    data_sample_0.set_metainfo(metainfo)
    data_sample_1.set_metainfo(metainfo)
    data = [
        dict(inputs=input_dict0, data_sample=data_sample_0),
        dict(inputs=input_dict1, data_sample=data_sample_1)
    ]
    model.forward(data, return_loss=False)


def test_sassd():
    # TODO fix this unitest
    pytest.skip('FIX this')

    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    _setup_seed(0)
    sassd_cfg = _get_detector_cfg('sassd/sassd_6x8_80e_kitti-3d-3class.py')

    self = build_detector(sassd_cfg).cuda()
    points_0 = torch.rand([2010, 4], device='cuda')
    points_1 = torch.rand([2020, 4], device='cuda')
    points = [points_0, points_1]
    gt_bbox_0 = LiDARInstance3DBoxes(torch.rand([10, 7], device='cuda'))
    gt_bbox_1 = LiDARInstance3DBoxes(torch.rand([10, 7], device='cuda'))
    gt_bboxes = [gt_bbox_0, gt_bbox_1]
    gt_labels_0 = torch.randint(0, 3, [10], device='cuda')
    gt_labels_1 = torch.randint(0, 3, [10], device='cuda')
    gt_labels = [gt_labels_0, gt_labels_1]
    img_meta_0 = dict(box_type_3d=LiDARInstance3DBoxes)
    img_meta_1 = dict(box_type_3d=LiDARInstance3DBoxes)
    img_metas = [img_meta_0, img_meta_1]

    # test forward_train
    losses = self.forward_train(points, img_metas, gt_bboxes, gt_labels)
    assert losses['loss_cls'][0] >= 0
    assert losses['loss_bbox'][0] >= 0
    assert losses['loss_dir'][0] >= 0
    assert losses['aux_loss_cls'][0] >= 0
    assert losses['aux_loss_reg'][0] >= 0

    # test simple_test
    with torch.no_grad():
        results = self.simple_test(points, img_metas)
    boxes_3d = results[0]['boxes_3d']
    scores_3d = results[0]['scores_3d']
    labels_3d = results[0]['labels_3d']
    assert boxes_3d.tensor.shape == (50, 7)
    assert scores_3d.shape == torch.Size([50])
    assert labels_3d.shape == torch.Size([50])
