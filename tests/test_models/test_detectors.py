# Copyright (c) OpenMMLab. All rights reserved.
import copy
import random
from os.path import dirname, exists, join

import numpy as np
import pytest
import torch

from mmdet3d.core.bbox import (CameraInstance3DBoxes, DepthInstance3DBoxes,
                               LiDARInstance3DBoxes)
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
        # Assume we are running in the source mmdetection3d repo
        repo_dpath = dirname(dirname(dirname(__file__)))
    except NameError:
        # For IPython development when this __file__ is not defined
        import mmdet3d
        repo_dpath = dirname(dirname(mmdet3d.__file__))
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
    train_cfg = mmcv.Config(copy.deepcopy(config.model.train_cfg))
    test_cfg = mmcv.Config(copy.deepcopy(config.model.test_cfg))

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
    with torch.no_grad():
        results = self.simple_test(points, img_metas)
    boxes_3d = results[0]['boxes_3d']
    scores_3d = results[0]['scores_3d']
    labels_3d = results[0]['labels_3d']
    assert boxes_3d.tensor.shape == (50, 7)
    assert scores_3d.shape == torch.Size([50])
    assert labels_3d.shape == torch.Size([50])


def test_3dssd():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    _setup_seed(0)
    ssd3d_cfg = _get_detector_cfg('3dssd/3dssd_4x4_kitti-3d-car.py')
    self = build_detector(ssd3d_cfg).cuda()
    points_0 = torch.rand([2000, 4], device='cuda')
    points_1 = torch.rand([2000, 4], device='cuda')
    points = [points_0, points_1]
    img_meta_0 = dict(box_type_3d=DepthInstance3DBoxes)
    img_meta_1 = dict(box_type_3d=DepthInstance3DBoxes)
    img_metas = [img_meta_0, img_meta_1]
    gt_bbox_0 = DepthInstance3DBoxes(torch.rand([10, 7], device='cuda'))
    gt_bbox_1 = DepthInstance3DBoxes(torch.rand([10, 7], device='cuda'))
    gt_bboxes = [gt_bbox_0, gt_bbox_1]
    gt_labels_0 = torch.zeros([10], device='cuda').long()
    gt_labels_1 = torch.zeros([10], device='cuda').long()
    gt_labels = [gt_labels_0, gt_labels_1]

    # test forward_train
    losses = self.forward_train(points, img_metas, gt_bboxes, gt_labels)
    assert losses['vote_loss'] >= 0
    assert losses['centerness_loss'] >= 0
    assert losses['center_loss'] >= 0
    assert losses['dir_class_loss'] >= 0
    assert losses['dir_res_loss'] >= 0
    assert losses['corner_loss'] >= 0
    assert losses['size_res_loss'] >= 0

    # test simple_test
    with torch.no_grad():
        results = self.simple_test(points, img_metas)
    boxes_3d = results[0]['boxes_3d']
    scores_3d = results[0]['scores_3d']
    labels_3d = results[0]['labels_3d']
    assert boxes_3d.tensor.shape[0] >= 0
    assert boxes_3d.tensor.shape[1] == 7
    assert scores_3d.shape[0] >= 0
    assert labels_3d.shape[0] >= 0


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
    with torch.no_grad():
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
    with torch.no_grad():
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
    with torch.no_grad():
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
    with torch.no_grad():
        results = self.aug_test(points, img_metas)
    boxes_3d_0 = results[0]['pts_bbox']['boxes_3d']
    scores_3d_0 = results[0]['pts_bbox']['scores_3d']
    labels_3d_0 = results[0]['pts_bbox']['labels_3d']
    assert boxes_3d_0.tensor.shape[0] >= 0
    assert boxes_3d_0.tensor.shape[1] == 9
    assert scores_3d_0.shape[0] >= 0
    assert labels_3d_0.shape[0] >= 0


def test_fcos3d():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')

    _setup_seed(0)
    fcos3d_cfg = _get_detector_cfg(
        'fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d.py')
    self = build_detector(fcos3d_cfg).cuda()
    imgs = torch.rand([1, 3, 928, 1600], dtype=torch.float32).cuda()
    gt_bboxes = [torch.rand([3, 4], dtype=torch.float32).cuda()]
    gt_bboxes_3d = CameraInstance3DBoxes(
        torch.rand([3, 9], device='cuda'), box_dim=9)
    gt_labels = [torch.randint(0, 10, [3], device='cuda')]
    gt_labels_3d = gt_labels
    centers2d = [torch.rand([3, 2], dtype=torch.float32).cuda()]
    depths = [torch.rand([3], dtype=torch.float32).cuda()]
    attr_labels = [torch.randint(0, 9, [3], device='cuda')]
    img_metas = [
        dict(
            cam2img=[[1260.8474446004698, 0.0, 807.968244525554],
                     [0.0, 1260.8474446004698, 495.3344268742088],
                     [0.0, 0.0, 1.0]],
            scale_factor=np.array([1., 1., 1., 1.], dtype=np.float32),
            box_type_3d=CameraInstance3DBoxes)
    ]

    # test forward_train
    losses = self.forward_train(imgs, img_metas, gt_bboxes, gt_labels,
                                gt_bboxes_3d, gt_labels_3d, centers2d, depths,
                                attr_labels)
    assert losses['loss_cls'] >= 0
    assert losses['loss_offset'] >= 0
    assert losses['loss_depth'] >= 0
    assert losses['loss_size'] >= 0
    assert losses['loss_rotsin'] >= 0
    assert losses['loss_centerness'] >= 0
    assert losses['loss_velo'] >= 0
    assert losses['loss_dir'] >= 0
    assert losses['loss_attr'] >= 0

    # test simple_test
    with torch.no_grad():
        results = self.simple_test(imgs, img_metas)
    boxes_3d = results[0]['img_bbox']['boxes_3d']
    scores_3d = results[0]['img_bbox']['scores_3d']
    labels_3d = results[0]['img_bbox']['labels_3d']
    attrs_3d = results[0]['img_bbox']['attrs_3d']
    assert boxes_3d.tensor.shape[0] >= 0
    assert boxes_3d.tensor.shape[1] == 9
    assert scores_3d.shape[0] >= 0
    assert labels_3d.shape[0] >= 0
    assert attrs_3d.shape[0] >= 0


def test_groupfree3dnet():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')

    _setup_seed(0)
    groupfree3d_cfg = _get_detector_cfg(
        'groupfree3d/groupfree3d_8x4_scannet-3d-18class-L6-O256.py')
    self = build_detector(groupfree3d_cfg).cuda()

    points_0 = torch.rand([50000, 3], device='cuda')
    points_1 = torch.rand([50000, 3], device='cuda')
    points = [points_0, points_1]
    img_meta_0 = dict(box_type_3d=DepthInstance3DBoxes)
    img_meta_1 = dict(box_type_3d=DepthInstance3DBoxes)
    img_metas = [img_meta_0, img_meta_1]
    gt_bbox_0 = DepthInstance3DBoxes(torch.rand([10, 7], device='cuda'))
    gt_bbox_1 = DepthInstance3DBoxes(torch.rand([10, 7], device='cuda'))
    gt_bboxes = [gt_bbox_0, gt_bbox_1]
    gt_labels_0 = torch.randint(0, 18, [10], device='cuda')
    gt_labels_1 = torch.randint(0, 18, [10], device='cuda')
    gt_labels = [gt_labels_0, gt_labels_1]
    pts_instance_mask_1 = torch.randint(0, 10, [50000], device='cuda')
    pts_instance_mask_2 = torch.randint(0, 10, [50000], device='cuda')
    pts_instance_mask = [pts_instance_mask_1, pts_instance_mask_2]
    pts_semantic_mask_1 = torch.randint(0, 19, [50000], device='cuda')
    pts_semantic_mask_2 = torch.randint(0, 19, [50000], device='cuda')
    pts_semantic_mask = [pts_semantic_mask_1, pts_semantic_mask_2]

    # test forward_train
    losses = self.forward_train(points, img_metas, gt_bboxes, gt_labels,
                                pts_semantic_mask, pts_instance_mask)

    assert losses['sampling_objectness_loss'] >= 0
    assert losses['s5.objectness_loss'] >= 0
    assert losses['s5.semantic_loss'] >= 0
    assert losses['s5.center_loss'] >= 0
    assert losses['s5.dir_class_loss'] >= 0
    assert losses['s5.dir_res_loss'] >= 0
    assert losses['s5.size_class_loss'] >= 0
    assert losses['s5.size_res_loss'] >= 0

    # test simple_test
    with torch.no_grad():
        results = self.simple_test(points, img_metas)
    boxes_3d = results[0]['boxes_3d']
    scores_3d = results[0]['scores_3d']
    labels_3d = results[0]['labels_3d']
    assert boxes_3d.tensor.shape[0] >= 0
    assert boxes_3d.tensor.shape[1] == 7
    assert scores_3d.shape[0] >= 0
    assert labels_3d.shape[0] >= 0


def test_imvoxelnet():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')

    imvoxelnet_cfg = _get_detector_cfg(
        'imvoxelnet/imvoxelnet_4x8_kitti-3d-car.py')
    self = build_detector(imvoxelnet_cfg).cuda()
    imgs = torch.rand([1, 3, 384, 1280], dtype=torch.float32).cuda()
    gt_bboxes_3d = [LiDARInstance3DBoxes(torch.rand([3, 7], device='cuda'))]
    gt_labels_3d = [torch.zeros([3], dtype=torch.long, device='cuda')]
    img_metas = [
        dict(
            box_type_3d=LiDARInstance3DBoxes,
            lidar2img=np.array([[6.0e+02, -7.2e+02, -1.2e+00, -1.2e+02],
                                [1.8e+02, 7.6e+00, -7.1e+02, -1.0e+02],
                                [9.9e-01, 1.2e-04, 1.0e-02, -2.6e-01],
                                [0.0e+00, 0.0e+00, 0.0e+00, 1.0e+00]],
                               dtype=np.float32),
            img_shape=(384, 1272, 3))
    ]

    # test forward_train
    losses = self.forward_train(imgs, img_metas, gt_bboxes_3d, gt_labels_3d)
    assert losses['loss_cls'][0] >= 0
    assert losses['loss_bbox'][0] >= 0
    assert losses['loss_dir'][0] >= 0

    # test simple_test
    with torch.no_grad():
        results = self.simple_test(imgs, img_metas)
    boxes_3d = results[0]['boxes_3d']
    scores_3d = results[0]['scores_3d']
    labels_3d = results[0]['labels_3d']
    assert boxes_3d.tensor.shape[0] >= 0
    assert boxes_3d.tensor.shape[1] == 7
    assert scores_3d.shape[0] >= 0
    assert labels_3d.shape[0] >= 0


def test_point_rcnn():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    pointrcnn_cfg = _get_detector_cfg(
        'point_rcnn/point_rcnn_2x8_kitti-3d-3classes.py')
    self = build_detector(pointrcnn_cfg).cuda()
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
    assert losses['bbox_loss'] >= 0
    assert losses['semantic_loss'] >= 0
    assert losses['loss_cls'] >= 0
    assert losses['loss_bbox'] >= 0
    assert losses['loss_corner'] >= 0


def test_smoke():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')

    _setup_seed(0)
    smoke_cfg = _get_detector_cfg(
        'smoke/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d.py')
    self = build_detector(smoke_cfg).cuda()
    imgs = torch.rand([1, 3, 384, 1280], dtype=torch.float32).cuda()
    gt_bboxes = [
        torch.Tensor([[563.63122442, 175.02195182, 614.81298184, 224.97763099],
                      [480.89676358, 179.86272635, 511.53017463, 202.54645962],
                      [541.48322272, 175.73767011, 564.55208966, 193.95009791],
                      [329.51448848, 176.14566789, 354.24670848,
                       213.82599081]]).cuda()
    ]
    gt_bboxes_3d = [
        CameraInstance3DBoxes(
            torch.Tensor([[-0.69, 1.69, 25.01, 3.20, 1.61, 1.66, -1.59],
                          [-7.43, 1.88, 47.55, 3.70, 1.40, 1.51, 1.55],
                          [-4.71, 1.71, 60.52, 4.05, 1.46, 1.66, 1.56],
                          [-12.63, 1.88, 34.09, 1.95, 1.72, 0.50,
                           1.54]]).cuda(),
            box_dim=7)
    ]
    gt_labels = [torch.tensor([0, 0, 0, 1]).cuda()]
    gt_labels_3d = gt_labels
    centers2d = [
        torch.Tensor([[589.6528477, 198.3862263], [496.8143155, 190.75967182],
                      [553.40528354, 184.53785991],
                      [342.23690317, 194.44298819]]).cuda()
    ]
    # depths is actually not used in smoke head loss computation
    depths = [torch.rand([3], dtype=torch.float32).cuda()]
    attr_labels = None
    img_metas = [
        dict(
            cam2img=[[721.5377, 0., 609.5593, 0.], [0., 721.5377, 172.854, 0.],
                     [0., 0., 1., 0.], [0., 0., 0., 1.]],
            scale_factor=np.array([1., 1., 1., 1.], dtype=np.float32),
            pad_shape=[384, 1280],
            trans_mat=np.array([[0.25, 0., 0.], [0., 0.25, 0], [0., 0., 1.]],
                               dtype=np.float32),
            affine_aug=False,
            box_type_3d=CameraInstance3DBoxes)
    ]

    # test forward_train
    losses = self.forward_train(imgs, img_metas, gt_bboxes, gt_labels,
                                gt_bboxes_3d, gt_labels_3d, centers2d, depths,
                                attr_labels)

    assert losses['loss_cls'] >= 0
    assert losses['loss_bbox'] >= 0

    # test simple_test
    with torch.no_grad():
        results = self.simple_test(imgs, img_metas)
    boxes_3d = results[0]['img_bbox']['boxes_3d']
    scores_3d = results[0]['img_bbox']['scores_3d']
    labels_3d = results[0]['img_bbox']['labels_3d']
    assert boxes_3d.tensor.shape[0] >= 0
    assert boxes_3d.tensor.shape[1] == 7
    assert scores_3d.shape[0] >= 0
    assert labels_3d.shape[0] >= 0


def test_sassd():
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
