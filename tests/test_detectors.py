import copy
import numpy as np
import pytest
import random
import torch
from os.path import dirname, exists, join

from mmdet3d.core.bbox import DepthInstance3DBoxes, LiDARInstance3DBoxes
from mmdet3d.models.builder import build_detector


def _setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _get_config_directory():
    """Find the predefined detector config directory."""
    try:
        # Assume we are running in the source mmdetection repo
        repo_dpath = dirname(dirname(__file__))
    except NameError:
        # For IPython development when this __file__ is not defined
        import mmdet
        repo_dpath = dirname(dirname(mmdet.__file__))
    config_dpath = join(repo_dpath, 'configs')
    if not exists(config_dpath):
        raise Exception('Cannot find config path')
    return config_dpath


def _get_config_module(fname):
    """Load a configuration as a python module."""
    from mmcv import Config
    config_dpath = _get_config_directory()
    config_fpath = join(config_dpath, fname)
    config_mod = Config.fromfile(config_fpath)
    return config_mod


def _get_model_cfg(fname):
    """Grab configs necessary to create a model.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)

    return model


def _get_detector_cfg(fname):
    """Grab configs necessary to create a detector.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    import mmcv
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    train_cfg = mmcv.Config(copy.deepcopy(config.train_cfg))
    test_cfg = mmcv.Config(copy.deepcopy(config.test_cfg))

    model.update(train_cfg=train_cfg)
    model.update(test_cfg=test_cfg)
    return model


def test_get_dynamic_voxelnet():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')

    dynamic_voxelnet_cfg = _get_model_cfg(
        'dynamic_voxelization/dv_second_secfpn_6x8_80e_kitti-3d-car.py')
    self = build_detector(dynamic_voxelnet_cfg).cuda()
    points_0 = torch.rand([2010, 4], device='cuda')
    points_1 = torch.rand([2020, 4], device='cuda')
    points = [points_0, points_1]
    feats = self.extract_feat(points, None)
    assert feats[0].shape == torch.Size([2, 512, 200, 176])


def test_voxel_net():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    _setup_seed(0)
    voxel_net_cfg = _get_detector_cfg(
        'second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py')

    self = build_detector(voxel_net_cfg).cuda()
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

    # test simple_test
    results = self.simple_test(points, img_metas)
    boxes_3d = results[0]['boxes_3d']
    scores_3d = results[0]['scores_3d']
    labels_3d = results[0]['labels_3d']
    assert boxes_3d.tensor.shape == (50, 7)
    assert scores_3d.shape == torch.Size([50])
    assert labels_3d.shape == torch.Size([50])


def test_vote_net():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')

    _setup_seed(0)
    vote_net_cfg = _get_detector_cfg(
        'votenet/votenet_16x8_sunrgbd-3d-10class.py')
    self = build_detector(vote_net_cfg).cuda()
    points_0 = torch.rand([2000, 4], device='cuda')
    points_1 = torch.rand([2000, 4], device='cuda')
    points = [points_0, points_1]
    img_meta_0 = dict(box_type_3d=DepthInstance3DBoxes)
    img_meta_1 = dict(box_type_3d=DepthInstance3DBoxes)
    img_metas = [img_meta_0, img_meta_1]
    gt_bbox_0 = DepthInstance3DBoxes(torch.rand([10, 7], device='cuda'))
    gt_bbox_1 = DepthInstance3DBoxes(torch.rand([10, 7], device='cuda'))
    gt_bboxes = [gt_bbox_0, gt_bbox_1]
    gt_labels_0 = torch.randint(0, 10, [10], device='cuda')
    gt_labels_1 = torch.randint(0, 10, [10], device='cuda')
    gt_labels = [gt_labels_0, gt_labels_1]

    # test forward_train
    losses = self.forward_train(points, img_metas, gt_bboxes, gt_labels)
    assert losses['vote_loss'] >= 0
    assert losses['objectness_loss'] >= 0
    assert losses['semantic_loss'] >= 0
    assert losses['center_loss'] >= 0
    assert losses['dir_class_loss'] >= 0
    assert losses['dir_res_loss'] >= 0
    assert losses['size_class_loss'] >= 0
    assert losses['size_res_loss'] >= 0

    # test simple_test
    results = self.simple_test(points, img_metas)
    boxes_3d = results[0]['boxes_3d']
    scores_3d = results[0]['scores_3d']
    labels_3d = results[0]['labels_3d']
    assert boxes_3d.tensor.shape[0] >= 0
    assert boxes_3d.tensor.shape[1] == 7
    assert scores_3d.shape[0] >= 0
    assert labels_3d.shape[0] >= 0


def test_parta2():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    _setup_seed(0)
    parta2 = _get_detector_cfg(
        'parta2/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class.py')
    self = build_detector(parta2).cuda()
    points_0 = torch.rand([1000, 4], device='cuda')
    points_1 = torch.rand([1000, 4], device='cuda')
    points = [points_0, points_1]
    img_meta_0 = dict(box_type_3d=LiDARInstance3DBoxes)
    img_meta_1 = dict(box_type_3d=LiDARInstance3DBoxes)
    img_metas = [img_meta_0, img_meta_1]
    gt_bbox_0 = LiDARInstance3DBoxes(torch.rand([10, 7], device='cuda'))
    gt_bbox_1 = LiDARInstance3DBoxes(torch.rand([10, 7], device='cuda'))
    gt_bboxes = [gt_bbox_0, gt_bbox_1]
    gt_labels_0 = torch.randint(0, 3, [10], device='cuda')
    gt_labels_1 = torch.randint(0, 3, [10], device='cuda')
    gt_labels = [gt_labels_0, gt_labels_1]

    # test_forward_train
    losses = self.forward_train(points, img_metas, gt_bboxes, gt_labels)
    assert losses['loss_rpn_cls'][0] >= 0
    assert losses['loss_rpn_bbox'][0] >= 0
    assert losses['loss_rpn_dir'][0] >= 0
    assert losses['loss_seg'] >= 0
    assert losses['loss_part'] >= 0
    assert losses['loss_cls'] >= 0
    assert losses['loss_bbox'] >= 0
    assert losses['loss_corner'] >= 0

    # test_simple_test
    results = self.simple_test(points, img_metas)
    boxes_3d = results[0]['boxes_3d']
    scores_3d = results[0]['scores_3d']
    labels_3d = results[0]['labels_3d']
    assert boxes_3d.tensor.shape[0] >= 0
    assert boxes_3d.tensor.shape[1] == 7
    assert scores_3d.shape[0] >= 0
    assert labels_3d.shape[0] >= 0


def test_centerpoint():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    centerpoint = _get_detector_cfg(
        'centerpoint/centerpoint_0075voxel_second_secfpn_'
        'dcn_4x8_cyclic_flip-tta_20e_nus.py')
    self = build_detector(centerpoint).cuda()
    points_0 = torch.rand([1000, 5], device='cuda')
    points_1 = torch.rand([1000, 5], device='cuda')
    points = [points_0, points_1]
    img_meta_0 = dict(
        box_type_3d=LiDARInstance3DBoxes,
        flip=True,
        pcd_horizontal_flip=True,
        pcd_vertical_flip=False)
    img_meta_1 = dict(
        box_type_3d=LiDARInstance3DBoxes,
        flip=True,
        pcd_horizontal_flip=False,
        pcd_vertical_flip=True)
    img_metas = [img_meta_0, img_meta_1]
    gt_bbox_0 = LiDARInstance3DBoxes(
        torch.rand([10, 9], device='cuda'), box_dim=9)
    gt_bbox_1 = LiDARInstance3DBoxes(
        torch.rand([10, 9], device='cuda'), box_dim=9)
    gt_bboxes = [gt_bbox_0, gt_bbox_1]
    gt_labels_0 = torch.randint(0, 3, [10], device='cuda')
    gt_labels_1 = torch.randint(0, 3, [10], device='cuda')
    gt_labels = [gt_labels_0, gt_labels_1]

    # test_forward_train
    losses = self.forward_train(points, img_metas, gt_bboxes, gt_labels)
    for key, value in losses.items():
        assert value >= 0

    # test_simple_test
    results = self.simple_test(points, img_metas)
    boxes_3d_0 = results[0]['pts_bbox']['boxes_3d']
    scores_3d_0 = results[0]['pts_bbox']['scores_3d']
    labels_3d_0 = results[0]['pts_bbox']['labels_3d']
    assert boxes_3d_0.tensor.shape[0] >= 0
    assert boxes_3d_0.tensor.shape[1] == 9
    assert scores_3d_0.shape[0] >= 0
    assert labels_3d_0.shape[0] >= 0
    boxes_3d_1 = results[1]['pts_bbox']['boxes_3d']
    scores_3d_1 = results[1]['pts_bbox']['scores_3d']
    labels_3d_1 = results[1]['pts_bbox']['labels_3d']
    assert boxes_3d_1.tensor.shape[0] >= 0
    assert boxes_3d_1.tensor.shape[1] == 9
    assert scores_3d_1.shape[0] >= 0
    assert labels_3d_1.shape[0] >= 0

    # test_aug_test
    points = [[torch.rand([1000, 5], device='cuda')]]
    img_metas = [[
        dict(
            box_type_3d=LiDARInstance3DBoxes,
            pcd_scale_factor=1.0,
            flip=True,
            pcd_horizontal_flip=True,
            pcd_vertical_flip=False)
    ]]
    results = self.aug_test(points, img_metas)
    boxes_3d_0 = results[0]['pts_bbox']['boxes_3d']
    scores_3d_0 = results[0]['pts_bbox']['scores_3d']
    labels_3d_0 = results[0]['pts_bbox']['labels_3d']
    assert boxes_3d_0.tensor.shape[0] >= 0
    assert boxes_3d_0.tensor.shape[1] == 9
    assert scores_3d_0.shape[0] >= 0
    assert labels_3d_0.shape[0] >= 0
