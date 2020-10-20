import copy
import numpy as np
import pytest
import random
import torch
from os.path import dirname, exists, join

from mmdet3d.core.bbox import (Box3DMode, DepthInstance3DBoxes,
                               LiDARInstance3DBoxes)
from mmdet3d.models.builder import build_head
from mmdet.apis import set_random_seed


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


def _get_head_cfg(fname):
    """Grab configs necessary to create a bbox_head.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    import mmcv
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    train_cfg = mmcv.Config(copy.deepcopy(config.train_cfg))
    test_cfg = mmcv.Config(copy.deepcopy(config.test_cfg))

    bbox_head = model.bbox_head
    bbox_head.update(train_cfg=train_cfg)
    bbox_head.update(test_cfg=test_cfg)
    return bbox_head


def _get_rpn_head_cfg(fname):
    """Grab configs necessary to create a rpn_head.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    import mmcv
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    train_cfg = mmcv.Config(copy.deepcopy(config.train_cfg))
    test_cfg = mmcv.Config(copy.deepcopy(config.test_cfg))

    rpn_head = model.rpn_head
    rpn_head.update(train_cfg=train_cfg.rpn)
    rpn_head.update(test_cfg=test_cfg.rpn)
    return rpn_head, train_cfg.rpn_proposal


def _get_roi_head_cfg(fname):
    """Grab configs necessary to create a roi_head.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    import mmcv
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    train_cfg = mmcv.Config(copy.deepcopy(config.train_cfg))
    test_cfg = mmcv.Config(copy.deepcopy(config.test_cfg))

    roi_head = model.roi_head
    roi_head.update(train_cfg=train_cfg.rcnn)
    roi_head.update(test_cfg=test_cfg.rcnn)
    return roi_head


def _get_pts_bbox_head_cfg(fname):
    """Grab configs necessary to create a pts_bbox_head.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    import mmcv
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    train_cfg = mmcv.Config(copy.deepcopy(config.train_cfg.pts))
    test_cfg = mmcv.Config(copy.deepcopy(config.test_cfg.pts))

    pts_bbox_head = model.pts_bbox_head
    pts_bbox_head.update(train_cfg=train_cfg)
    pts_bbox_head.update(test_cfg=test_cfg)
    return pts_bbox_head


def _get_vote_head_cfg(fname):
    """Grab configs necessary to create a vote_head.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    import mmcv
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    train_cfg = mmcv.Config(copy.deepcopy(config.train_cfg))
    test_cfg = mmcv.Config(copy.deepcopy(config.test_cfg))

    vote_head = model.bbox_head
    vote_head.update(train_cfg=train_cfg)
    vote_head.update(test_cfg=test_cfg)
    return vote_head


def _get_parta2_bbox_head_cfg(fname):
    """Grab configs necessary to create a parta2_bbox_head.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)

    vote_head = model.roi_head.bbox_head
    return vote_head


def test_anchor3d_head_loss():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    bbox_head_cfg = _get_head_cfg(
        'second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py')

    from mmdet3d.models.builder import build_head
    self = build_head(bbox_head_cfg)
    self.cuda()
    assert isinstance(self.conv_cls, torch.nn.modules.conv.Conv2d)
    assert self.conv_cls.in_channels == 512
    assert self.conv_cls.out_channels == 18
    assert self.conv_reg.out_channels == 42
    assert self.conv_dir_cls.out_channels == 12

    # test forward
    feats = list()
    feats.append(torch.rand([2, 512, 200, 176], dtype=torch.float32).cuda())
    (cls_score, bbox_pred, dir_cls_preds) = self.forward(feats)
    assert cls_score[0].shape == torch.Size([2, 18, 200, 176])
    assert bbox_pred[0].shape == torch.Size([2, 42, 200, 176])
    assert dir_cls_preds[0].shape == torch.Size([2, 12, 200, 176])

    # test loss
    gt_bboxes = list(
        torch.tensor(
            [[[6.4118, -3.4305, -1.7291, 1.7033, 3.4693, 1.6197, -0.9091]],
             [[16.9107, 9.7925, -1.9201, 1.6097, 3.2786, 1.5307, -2.4056]]],
            dtype=torch.float32).cuda())
    gt_labels = list(torch.tensor([[0], [1]], dtype=torch.int64).cuda())
    input_metas = [{
        'sample_idx': 1234
    }, {
        'sample_idx': 2345
    }]  # fake input_metas

    losses = self.loss(cls_score, bbox_pred, dir_cls_preds, gt_bboxes,
                       gt_labels, input_metas)
    assert losses['loss_cls'][0] > 0
    assert losses['loss_bbox'][0] > 0
    assert losses['loss_dir'][0] > 0

    # test empty ground truth case
    gt_bboxes = list(torch.empty((2, 0, 7)).cuda())
    gt_labels = list(torch.empty((2, 0)).cuda())
    empty_gt_losses = self.loss(cls_score, bbox_pred, dir_cls_preds, gt_bboxes,
                                gt_labels, input_metas)
    assert empty_gt_losses['loss_cls'][0] > 0
    assert empty_gt_losses['loss_bbox'][0] == 0
    assert empty_gt_losses['loss_dir'][0] == 0


def test_anchor3d_head_getboxes():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    bbox_head_cfg = _get_head_cfg(
        'second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py')

    from mmdet3d.models.builder import build_head
    self = build_head(bbox_head_cfg)
    self.cuda()

    feats = list()
    feats.append(torch.rand([2, 512, 200, 176], dtype=torch.float32).cuda())
    # fake input_metas
    input_metas = [{
        'sample_idx': 1234,
        'box_type_3d': LiDARInstance3DBoxes,
        'box_mode_3d': Box3DMode.LIDAR
    }, {
        'sample_idx': 2345,
        'box_type_3d': LiDARInstance3DBoxes,
        'box_mode_3d': Box3DMode.LIDAR
    }]
    (cls_score, bbox_pred, dir_cls_preds) = self.forward(feats)

    # test get_boxes
    cls_score[0] -= 1.5  # too many positive samples may cause cuda oom
    result_list = self.get_bboxes(cls_score, bbox_pred, dir_cls_preds,
                                  input_metas)
    assert (result_list[0][1] > 0.3).all()


def test_parta2_rpnhead_getboxes():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    rpn_head_cfg, proposal_cfg = _get_rpn_head_cfg(
        'parta2/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class.py')

    self = build_head(rpn_head_cfg)
    self.cuda()

    feats = list()
    feats.append(torch.rand([2, 512, 200, 176], dtype=torch.float32).cuda())
    # fake input_metas
    input_metas = [{
        'sample_idx': 1234,
        'box_type_3d': LiDARInstance3DBoxes,
        'box_mode_3d': Box3DMode.LIDAR
    }, {
        'sample_idx': 2345,
        'box_type_3d': LiDARInstance3DBoxes,
        'box_mode_3d': Box3DMode.LIDAR
    }]
    (cls_score, bbox_pred, dir_cls_preds) = self.forward(feats)

    # test get_boxes
    cls_score[0] -= 1.5  # too many positive samples may cause cuda oom
    result_list = self.get_bboxes(cls_score, bbox_pred, dir_cls_preds,
                                  input_metas, proposal_cfg)
    assert result_list[0]['scores_3d'].shape == torch.Size([512])
    assert result_list[0]['labels_3d'].shape == torch.Size([512])
    assert result_list[0]['cls_preds'].shape == torch.Size([512, 3])
    assert result_list[0]['boxes_3d'].tensor.shape == torch.Size([512, 7])


def test_vote_head():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    _setup_seed(0)
    vote_head_cfg = _get_vote_head_cfg(
        'votenet/votenet_8x8_scannet-3d-18class.py')
    self = build_head(vote_head_cfg).cuda()
    fp_xyz = [torch.rand([2, 256, 3], dtype=torch.float32).cuda()]
    fp_features = [torch.rand([2, 256, 256], dtype=torch.float32).cuda()]
    fp_indices = [torch.randint(0, 128, [2, 256]).cuda()]

    input_dict = dict(
        fp_xyz=fp_xyz, fp_features=fp_features, fp_indices=fp_indices)

    # test forward
    ret_dict = self(input_dict, 'vote')
    assert ret_dict['center'].shape == torch.Size([2, 256, 3])
    assert ret_dict['obj_scores'].shape == torch.Size([2, 256, 2])
    assert ret_dict['size_res'].shape == torch.Size([2, 256, 18, 3])
    assert ret_dict['dir_res'].shape == torch.Size([2, 256, 1])

    # test loss
    points = [torch.rand([40000, 4], device='cuda') for i in range(2)]
    gt_bbox1 = LiDARInstance3DBoxes(torch.rand([10, 7], device='cuda'))
    gt_bbox2 = LiDARInstance3DBoxes(torch.rand([10, 7], device='cuda'))
    gt_bboxes = [gt_bbox1, gt_bbox2]
    gt_labels = [torch.randint(0, 18, [10], device='cuda') for i in range(2)]
    pts_semantic_mask = [
        torch.randint(0, 18, [40000], device='cuda') for i in range(2)
    ]
    pts_instance_mask = [
        torch.randint(0, 10, [40000], device='cuda') for i in range(2)
    ]
    losses = self.loss(ret_dict, points, gt_bboxes, gt_labels,
                       pts_semantic_mask, pts_instance_mask)
    assert losses['vote_loss'] >= 0
    assert losses['objectness_loss'] >= 0
    assert losses['semantic_loss'] >= 0
    assert losses['center_loss'] >= 0
    assert losses['dir_class_loss'] >= 0
    assert losses['dir_res_loss'] >= 0
    assert losses['size_class_loss'] >= 0
    assert losses['size_res_loss'] >= 0

    # test multiclass_nms_single
    obj_scores = torch.rand([256], device='cuda')
    sem_scores = torch.rand([256, 18], device='cuda')
    points = torch.rand([40000, 3], device='cuda')
    bbox = torch.rand([256, 7], device='cuda')
    input_meta = dict(box_type_3d=DepthInstance3DBoxes)
    bbox_selected, score_selected, labels = self.multiclass_nms_single(
        obj_scores, sem_scores, bbox, points, input_meta)
    assert bbox_selected.shape[0] >= 0
    assert bbox_selected.shape[1] == 7
    assert score_selected.shape[0] >= 0
    assert labels.shape[0] >= 0

    # test get_boxes
    points = torch.rand([1, 40000, 4], device='cuda')
    seed_points = torch.rand([1, 1024, 3], device='cuda')
    seed_indices = torch.randint(0, 40000, [1, 1024], device='cuda')
    vote_points = torch.rand([1, 1024, 3], device='cuda')
    vote_features = torch.rand([1, 256, 1024], device='cuda')
    aggregated_points = torch.rand([1, 256, 3], device='cuda')
    aggregated_indices = torch.range(0, 256, device='cuda')
    obj_scores = torch.rand([1, 256, 2], device='cuda')
    center = torch.rand([1, 256, 3], device='cuda')
    dir_class = torch.rand([1, 256, 1], device='cuda')
    dir_res_norm = torch.rand([1, 256, 1], device='cuda')
    dir_res = torch.rand([1, 256, 1], device='cuda')
    size_class = torch.rand([1, 256, 18], device='cuda')
    size_res = torch.rand([1, 256, 18, 3], device='cuda')
    sem_scores = torch.rand([1, 256, 18], device='cuda')
    bbox_preds = dict(
        seed_points=seed_points,
        seed_indices=seed_indices,
        vote_points=vote_points,
        vote_features=vote_features,
        aggregated_points=aggregated_points,
        aggregated_indices=aggregated_indices,
        obj_scores=obj_scores,
        center=center,
        dir_class=dir_class,
        dir_res_norm=dir_res_norm,
        dir_res=dir_res,
        size_class=size_class,
        size_res=size_res,
        sem_scores=sem_scores)
    results = self.get_bboxes(points, bbox_preds, [input_meta])
    assert results[0][0].tensor.shape[0] >= 0
    assert results[0][0].tensor.shape[1] == 7
    assert results[0][1].shape[0] >= 0
    assert results[0][2].shape[0] >= 0


def test_parta2_bbox_head():
    parta2_bbox_head_cfg = _get_parta2_bbox_head_cfg(
        './parta2/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class.py')
    self = build_head(parta2_bbox_head_cfg)
    seg_feats = torch.rand([256, 14, 14, 14, 16])
    part_feats = torch.rand([256, 14, 14, 14, 4])

    cls_score, bbox_pred = self.forward(seg_feats, part_feats)
    assert cls_score.shape == (256, 1)
    assert bbox_pred.shape == (256, 7)


def test_part_aggregation_ROI_head():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')

    roi_head_cfg = _get_roi_head_cfg(
        'parta2/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class.py')
    self = build_head(roi_head_cfg).cuda()

    features = np.load('./tests/test_samples/parta2_roihead_inputs.npz')
    seg_features = torch.tensor(
        features['seg_features'], dtype=torch.float32, device='cuda')
    feats_dict = dict(seg_features=seg_features)

    voxels = torch.tensor(
        features['voxels'], dtype=torch.float32, device='cuda')
    num_points = torch.ones([500], device='cuda')
    coors = torch.zeros([500, 4], device='cuda')
    voxel_centers = torch.zeros([500, 3], device='cuda')
    box_type_3d = LiDARInstance3DBoxes
    img_metas = [dict(box_type_3d=box_type_3d)]
    voxels_dict = dict(
        voxels=voxels,
        num_points=num_points,
        coors=coors,
        voxel_centers=voxel_centers)

    pred_bboxes = LiDARInstance3DBoxes(
        torch.tensor(
            [[0.3990, 0.5167, 0.0249, 0.9401, 0.9459, 0.7967, 0.4150],
             [0.8203, 0.2290, 0.9096, 0.1183, 0.0752, 0.4092, 0.9601],
             [0.2093, 0.1940, 0.8909, 0.4387, 0.3570, 0.5454, 0.8299],
             [0.2099, 0.7684, 0.4290, 0.2117, 0.6606, 0.1654, 0.4250],
             [0.9927, 0.6964, 0.2472, 0.7028, 0.7494, 0.9303, 0.0494]],
            dtype=torch.float32,
            device='cuda'))
    pred_scores = torch.tensor([0.9722, 0.7910, 0.4690, 0.3300, 0.3345],
                               dtype=torch.float32,
                               device='cuda')
    pred_labels = torch.tensor([0, 1, 0, 2, 1],
                               dtype=torch.int64,
                               device='cuda')
    pred_clses = torch.tensor(
        [[0.7874, 0.1344, 0.2190], [0.8193, 0.6969, 0.7304],
         [0.2328, 0.9028, 0.3900], [0.6177, 0.5012, 0.2330],
         [0.8985, 0.4894, 0.7152]],
        dtype=torch.float32,
        device='cuda')
    proposal = dict(
        boxes_3d=pred_bboxes,
        scores_3d=pred_scores,
        labels_3d=pred_labels,
        cls_preds=pred_clses)
    proposal_list = [proposal]
    gt_bboxes_3d = [LiDARInstance3DBoxes(torch.rand([5, 7], device='cuda'))]
    gt_labels_3d = [torch.randint(0, 3, [5], device='cuda')]

    losses = self.forward_train(feats_dict, voxels_dict, {}, proposal_list,
                                gt_bboxes_3d, gt_labels_3d)
    assert losses['loss_seg'] >= 0
    assert losses['loss_part'] >= 0
    assert losses['loss_cls'] >= 0
    assert losses['loss_bbox'] >= 0
    assert losses['loss_corner'] >= 0

    bbox_results = self.simple_test(feats_dict, voxels_dict, img_metas,
                                    proposal_list)
    boxes_3d = bbox_results[0]['boxes_3d']
    scores_3d = bbox_results[0]['scores_3d']
    labels_3d = bbox_results[0]['labels_3d']
    assert boxes_3d.tensor.shape == (12, 7)
    assert scores_3d.shape == (12, )
    assert labels_3d.shape == (12, )


def test_free_anchor_3D_head():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    _setup_seed(0)
    pts_bbox_head_cfg = _get_pts_bbox_head_cfg(
        './free_anchor/hv_pointpillars_fpn_sbn-all_'
        'free-anchor_4x8_2x_nus-3d.py')
    self = build_head(pts_bbox_head_cfg)
    cls_scores = [
        torch.rand([4, 80, 200, 200], device='cuda') for i in range(3)
    ]
    bbox_preds = [
        torch.rand([4, 72, 200, 200], device='cuda') for i in range(3)
    ]
    dir_cls_preds = [
        torch.rand([4, 16, 200, 200], device='cuda') for i in range(3)
    ]
    gt_bboxes = [
        LiDARInstance3DBoxes(torch.rand([8, 9], device='cuda'), box_dim=9)
        for i in range(4)
    ]
    gt_labels = [
        torch.randint(0, 10, [8], device='cuda', dtype=torch.long)
        for i in range(4)
    ]
    input_metas = [0]
    losses = self.loss(cls_scores, bbox_preds, dir_cls_preds, gt_bboxes,
                       gt_labels, input_metas, None)
    assert losses['positive_bag_loss'] >= 0
    assert losses['negative_bag_loss'] >= 0


def test_primitive_head():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    _setup_seed(0)

    primitive_head_cfg = dict(
        type='PrimitiveHead',
        num_dims=2,
        num_classes=18,
        primitive_mode='z',
        vote_module_cfg=dict(
            in_channels=256,
            vote_per_seed=1,
            gt_per_seed=1,
            conv_channels=(256, 256),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d'),
            norm_feats=True,
            vote_loss=dict(
                type='ChamferDistance',
                mode='l1',
                reduction='none',
                loss_dst_weight=10.0)),
        vote_aggregation_cfg=dict(
            type='PointSAModule',
            num_point=64,
            radius=0.3,
            num_sample=16,
            mlp_channels=[256, 128, 128, 128],
            use_xyz=True,
            normalize_xyz=True),
        feat_channels=(128, 128),
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        objectness_loss=dict(
            type='CrossEntropyLoss',
            class_weight=[0.4, 0.6],
            reduction='mean',
            loss_weight=1.0),
        center_loss=dict(
            type='ChamferDistance',
            mode='l1',
            reduction='sum',
            loss_src_weight=1.0,
            loss_dst_weight=1.0),
        semantic_reg_loss=dict(
            type='ChamferDistance',
            mode='l1',
            reduction='sum',
            loss_src_weight=1.0,
            loss_dst_weight=1.0),
        semantic_cls_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        train_cfg=dict(
            dist_thresh=0.2,
            var_thresh=1e-2,
            lower_thresh=1e-6,
            num_point=100,
            num_point_line=10,
            line_thresh=0.2))

    self = build_head(primitive_head_cfg).cuda()
    fp_xyz = [torch.rand([2, 64, 3], dtype=torch.float32).cuda()]
    hd_features = torch.rand([2, 256, 64], dtype=torch.float32).cuda()
    fp_indices = [torch.randint(0, 64, [2, 64]).cuda()]
    input_dict = dict(
        fp_xyz_net0=fp_xyz, hd_feature=hd_features, fp_indices_net0=fp_indices)

    # test forward
    ret_dict = self(input_dict, 'vote')
    assert ret_dict['center_z'].shape == torch.Size([2, 64, 3])
    assert ret_dict['size_residuals_z'].shape == torch.Size([2, 64, 2])
    assert ret_dict['sem_cls_scores_z'].shape == torch.Size([2, 64, 18])
    assert ret_dict['aggregated_points_z'].shape == torch.Size([2, 64, 3])

    # test loss
    points = torch.rand([2, 1024, 3], dtype=torch.float32).cuda()
    ret_dict['seed_points'] = fp_xyz[0]
    ret_dict['seed_indices'] = fp_indices[0]

    from mmdet3d.core.bbox import DepthInstance3DBoxes
    gt_bboxes_3d = [
        DepthInstance3DBoxes(torch.rand([4, 7], dtype=torch.float32).cuda()),
        DepthInstance3DBoxes(torch.rand([4, 7], dtype=torch.float32).cuda())
    ]
    gt_labels_3d = torch.randint(0, 18, [2, 4]).cuda()
    gt_labels_3d = [gt_labels_3d[0], gt_labels_3d[1]]
    pts_semantic_mask = torch.randint(0, 19, [2, 1024]).cuda()
    pts_semantic_mask = [pts_semantic_mask[0], pts_semantic_mask[1]]
    pts_instance_mask = torch.randint(0, 4, [2, 1024]).cuda()
    pts_instance_mask = [pts_instance_mask[0], pts_instance_mask[1]]

    loss_input_dict = dict(
        bbox_preds=ret_dict,
        points=points,
        gt_bboxes_3d=gt_bboxes_3d,
        gt_labels_3d=gt_labels_3d,
        pts_semantic_mask=pts_semantic_mask,
        pts_instance_mask=pts_instance_mask)
    losses_dict = self.loss(**loss_input_dict)

    assert losses_dict['flag_loss_z'] >= 0
    assert losses_dict['vote_loss_z'] >= 0
    assert losses_dict['center_loss_z'] >= 0
    assert losses_dict['size_loss_z'] >= 0
    assert losses_dict['sem_loss_z'] >= 0

    # 'Primitive_mode' should be one of ['z', 'xy', 'line']
    with pytest.raises(AssertionError):
        primitive_head_cfg['vote_module_cfg']['in_channels'] = 'xyz'
        build_head(primitive_head_cfg)


def test_h3d_head():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    _setup_seed(0)

    h3d_head_cfg = _get_roi_head_cfg('h3dnet/h3dnet_8x3_scannet-3d-18class.py')

    num_point = 128
    num_proposal = 64
    h3d_head_cfg.primitive_list[0].vote_aggregation_cfg.num_point = num_point
    h3d_head_cfg.primitive_list[1].vote_aggregation_cfg.num_point = num_point
    h3d_head_cfg.primitive_list[2].vote_aggregation_cfg.num_point = num_point
    h3d_head_cfg.bbox_head.num_proposal = num_proposal
    self = build_head(h3d_head_cfg).cuda()

    # prepare roi outputs
    fp_xyz = [torch.rand([1, num_point, 3], dtype=torch.float32).cuda()]
    hd_features = torch.rand([1, 256, num_point], dtype=torch.float32).cuda()
    fp_indices = [torch.randint(0, 128, [1, num_point]).cuda()]
    aggregated_points = torch.rand([1, num_proposal, 3],
                                   dtype=torch.float32).cuda()
    aggregated_features = torch.rand([1, 128, num_proposal],
                                     dtype=torch.float32).cuda()
    proposal_list = torch.cat([
        torch.rand([1, num_proposal, 3], dtype=torch.float32).cuda() * 4 - 2,
        torch.rand([1, num_proposal, 3], dtype=torch.float32).cuda() * 4,
        torch.zeros([1, num_proposal, 1]).cuda()
    ],
                              dim=-1)

    input_dict = dict(
        fp_xyz_net0=fp_xyz,
        hd_feature=hd_features,
        aggregated_points=aggregated_points,
        aggregated_features=aggregated_features,
        seed_points=fp_xyz[0],
        seed_indices=fp_indices[0],
        proposal_list=proposal_list)

    # prepare gt label
    from mmdet3d.core.bbox import DepthInstance3DBoxes
    gt_bboxes_3d = [
        DepthInstance3DBoxes(torch.rand([4, 7], dtype=torch.float32).cuda()),
        DepthInstance3DBoxes(torch.rand([4, 7], dtype=torch.float32).cuda())
    ]
    gt_labels_3d = torch.randint(0, 18, [1, 4]).cuda()
    gt_labels_3d = [gt_labels_3d[0]]
    pts_semantic_mask = torch.randint(0, 19, [1, num_point]).cuda()
    pts_semantic_mask = [pts_semantic_mask[0]]
    pts_instance_mask = torch.randint(0, 4, [1, num_point]).cuda()
    pts_instance_mask = [pts_instance_mask[0]]
    points = torch.rand([1, num_point, 3], dtype=torch.float32).cuda()

    # prepare rpn targets
    vote_targets = torch.rand([1, num_point, 9], dtype=torch.float32).cuda()
    vote_target_masks = torch.rand([1, num_point], dtype=torch.float32).cuda()
    size_class_targets = torch.rand([1, num_proposal],
                                    dtype=torch.float32).cuda().long()
    size_res_targets = torch.rand([1, num_proposal, 3],
                                  dtype=torch.float32).cuda()
    dir_class_targets = torch.rand([1, num_proposal],
                                   dtype=torch.float32).cuda().long()
    dir_res_targets = torch.rand([1, num_proposal], dtype=torch.float32).cuda()
    center_targets = torch.rand([1, 4, 3], dtype=torch.float32).cuda()
    mask_targets = torch.rand([1, num_proposal],
                              dtype=torch.float32).cuda().long()
    valid_gt_masks = torch.rand([1, 4], dtype=torch.float32).cuda()
    objectness_targets = torch.rand([1, num_proposal],
                                    dtype=torch.float32).cuda().long()
    objectness_weights = torch.rand([1, num_proposal],
                                    dtype=torch.float32).cuda()
    box_loss_weights = torch.rand([1, num_proposal],
                                  dtype=torch.float32).cuda()
    valid_gt_weights = torch.rand([1, 4], dtype=torch.float32).cuda()

    targets = (vote_targets, vote_target_masks, size_class_targets,
               size_res_targets, dir_class_targets, dir_res_targets,
               center_targets, mask_targets, valid_gt_masks,
               objectness_targets, objectness_weights, box_loss_weights,
               valid_gt_weights)

    input_dict['targets'] = targets

    # train forward
    ret_dict = self.forward_train(
        input_dict,
        points=points,
        gt_bboxes_3d=gt_bboxes_3d,
        gt_labels_3d=gt_labels_3d,
        pts_semantic_mask=pts_semantic_mask,
        pts_instance_mask=pts_instance_mask,
        img_metas=None)

    assert ret_dict['flag_loss_z'] >= 0
    assert ret_dict['vote_loss_z'] >= 0
    assert ret_dict['center_loss_z'] >= 0
    assert ret_dict['size_loss_z'] >= 0
    assert ret_dict['sem_loss_z'] >= 0
    assert ret_dict['objectness_loss_optimized'] >= 0
    assert ret_dict['primitive_sem_matching_loss'] >= 0


def test_center_head():
    tasks = [
        dict(num_class=1, class_names=['car']),
        dict(num_class=2, class_names=['truck', 'construction_vehicle']),
        dict(num_class=2, class_names=['bus', 'trailer']),
        dict(num_class=1, class_names=['barrier']),
        dict(num_class=2, class_names=['motorcycle', 'bicycle']),
        dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
    ]
    bbox_cfg = dict(
        type='CenterPointBBoxCoder',
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        max_num=500,
        score_threshold=0.1,
        pc_range=[-51.2, -51.2],
        out_size_factor=8,
        voxel_size=[0.2, 0.2])
    train_cfg = dict(
        grid_size=[1024, 1024, 40],
        point_cloud_range=[-51.2, -51.2, -5., 51.2, 51.2, 3.],
        voxel_size=[0.1, 0.1, 0.2],
        out_size_factor=8,
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0],
        min_radius=2)
    test_cfg = dict(
        post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        max_per_img=500,
        max_pool_nms=False,
        min_radius=[4, 12, 10, 1, 0.85, 0.175],
        post_max_size=83,
        score_threshold=0.1,
        pc_range=[-51.2, -51.2],
        out_size_factor=8,
        voxel_size=[0.2, 0.2],
        nms_type='circle')
    center_head_cfg = dict(
        type='CenterHead',
        in_channels=sum([256, 256]),
        tasks=tasks,
        train_cfg=train_cfg,
        test_cfg=test_cfg,
        bbox_coder=bbox_cfg,
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        norm_bbox=True)

    center_head = build_head(center_head_cfg)

    x = torch.rand([2, 512, 128, 128])
    output = center_head([x])
    for i in range(6):
        assert output[i][0]['reg'].shape == torch.Size([2, 2, 128, 128])
        assert output[i][0]['height'].shape == torch.Size([2, 1, 128, 128])
        assert output[i][0]['dim'].shape == torch.Size([2, 3, 128, 128])
        assert output[i][0]['rot'].shape == torch.Size([2, 2, 128, 128])
        assert output[i][0]['vel'].shape == torch.Size([2, 2, 128, 128])
        assert output[i][0]['heatmap'].shape == torch.Size(
            [2, tasks[i]['num_class'], 128, 128])

    # test get_bboxes
    img_metas = [
        dict(box_type_3d=LiDARInstance3DBoxes),
        dict(box_type_3d=LiDARInstance3DBoxes)
    ]
    ret_lists = center_head.get_bboxes(output, img_metas)
    for ret_list in ret_lists:
        assert ret_list[0].tensor.shape[0] <= 500
        assert ret_list[1].shape[0] <= 500
        assert ret_list[2].shape[0] <= 500


def test_dcn_center_head():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and CUDA')
    set_random_seed(0)
    tasks = [
        dict(num_class=1, class_names=['car']),
        dict(num_class=2, class_names=['truck', 'construction_vehicle']),
        dict(num_class=2, class_names=['bus', 'trailer']),
        dict(num_class=1, class_names=['barrier']),
        dict(num_class=2, class_names=['motorcycle', 'bicycle']),
        dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
    ]
    voxel_size = [0.2, 0.2, 8]
    dcn_center_head_cfg = dict(
        type='CenterHead',
        in_channels=sum([128, 128, 128]),
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
        ],
        common_heads={
            'reg': (2, 2),
            'height': (1, 2),
            'dim': (3, 2),
            'rot': (2, 2),
            'vel': (2, 2)
        },
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            pc_range=[-51.2, -51.2],
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            code_size=9),
        seperate_head=dict(
            type='DCNSeperateHead',
            dcn_config=dict(
                type='DCN',
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                groups=4,
                bias=True),
            init_bias=-2.19,
            final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='none', loss_weight=0.25),
        norm_bbox=True)
    # model training and testing settings
    train_cfg = dict(
        grid_size=[512, 512, 1],
        point_cloud_range=[-51.2, -51.2, -5., 51.2, 51.2, 3.],
        voxel_size=voxel_size,
        out_size_factor=4,
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0])

    test_cfg = dict(
        post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        max_per_img=500,
        max_pool_nms=False,
        min_radius=[4, 12, 10, 1, 0.85, 0.175],
        post_max_size=83,
        score_threshold=0.1,
        pc_range=[-51.2, -51.2],
        out_size_factor=4,
        voxel_size=voxel_size[:2],
        nms_type='circle')
    dcn_center_head_cfg.update(train_cfg=train_cfg, test_cfg=test_cfg)

    dcn_center_head = build_head(dcn_center_head_cfg).cuda()

    x = torch.ones([2, 384, 128, 128]).cuda()
    output = dcn_center_head([x])
    for i in range(6):
        assert output[i][0]['reg'].shape == torch.Size([2, 2, 128, 128])
        assert output[i][0]['height'].shape == torch.Size([2, 1, 128, 128])
        assert output[i][0]['dim'].shape == torch.Size([2, 3, 128, 128])
        assert output[i][0]['rot'].shape == torch.Size([2, 2, 128, 128])
        assert output[i][0]['vel'].shape == torch.Size([2, 2, 128, 128])
        assert output[i][0]['heatmap'].shape == torch.Size(
            [2, tasks[i]['num_class'], 128, 128])

    # Test loss.
    gt_bboxes_0 = LiDARInstance3DBoxes(torch.rand([10, 9]).cuda(), box_dim=9)
    gt_bboxes_1 = LiDARInstance3DBoxes(torch.rand([20, 9]).cuda(), box_dim=9)
    gt_labels_0 = torch.randint(1, 11, [10]).cuda()
    gt_labels_1 = torch.randint(1, 11, [20]).cuda()
    gt_bboxes_3d = [gt_bboxes_0, gt_bboxes_1]
    gt_labels_3d = [gt_labels_0, gt_labels_1]
    loss = dcn_center_head.loss(gt_bboxes_3d, gt_labels_3d, output)
    for key, item in loss.items():
        if 'heatmap' in key:
            assert item >= 0
        else:
            assert torch.sum(item) >= 0

    # test get_bboxes
    img_metas = [
        dict(box_type_3d=LiDARInstance3DBoxes),
        dict(box_type_3d=LiDARInstance3DBoxes)
    ]
    ret_lists = dcn_center_head.get_bboxes(output, img_metas)
    for ret_list in ret_lists:
        assert ret_list[0].tensor.shape[0] <= 500
        assert ret_list[1].shape[0] <= 500
        assert ret_list[2].shape[0] <= 500


def test_ssd3d_head():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    _setup_seed(0)
    ssd3d_head_cfg = _get_vote_head_cfg('3dssd/3dssd_kitti-3d-car.py')
    ssd3d_head_cfg.vote_module_cfg.num_points = 64
    self = build_head(ssd3d_head_cfg).cuda()
    sa_xyz = [torch.rand([2, 128, 3], dtype=torch.float32).cuda()]
    sa_features = [torch.rand([2, 256, 128], dtype=torch.float32).cuda()]
    sa_indices = [torch.randint(0, 64, [2, 128]).cuda()]

    input_dict = dict(
        sa_xyz=sa_xyz, sa_features=sa_features, sa_indices=sa_indices)

    # test forward
    ret_dict = self(input_dict, 'spec')
    assert ret_dict['center'].shape == torch.Size([2, 64, 3])
    assert ret_dict['obj_scores'].shape == torch.Size([2, 1, 64])
    assert ret_dict['size'].shape == torch.Size([2, 64, 3])
    assert ret_dict['dir_res'].shape == torch.Size([2, 64, 12])

    # test loss
    points = [torch.rand([4000, 4], device='cuda') for i in range(2)]
    gt_bbox1 = LiDARInstance3DBoxes(torch.rand([5, 7], device='cuda'))
    gt_bbox2 = LiDARInstance3DBoxes(torch.rand([5, 7], device='cuda'))
    gt_bboxes = [gt_bbox1, gt_bbox2]
    gt_labels = [
        torch.zeros([5], dtype=torch.long, device='cuda') for i in range(2)
    ]
    img_metas = [dict(box_type_3d=LiDARInstance3DBoxes) for i in range(2)]
    losses = self.loss(
        ret_dict, points, gt_bboxes, gt_labels, img_metas=img_metas)

    assert losses['centerness_loss'] >= 0
    assert losses['center_loss'] >= 0
    assert losses['dir_class_loss'] >= 0
    assert losses['dir_res_loss'] >= 0
    assert losses['size_res_loss'] >= 0
    assert losses['corner_loss'] >= 0
    assert losses['vote_loss'] >= 0

    # test multiclass_nms_single
    sem_scores = ret_dict['obj_scores'].transpose(1, 2)[0]
    obj_scores = sem_scores.max(-1)[0]
    bbox = self.bbox_coder.decode(ret_dict)[0]
    input_meta = img_metas[0]
    bbox_selected, score_selected, labels = self.multiclass_nms_single(
        obj_scores, sem_scores, bbox, points[0], input_meta)
    assert bbox_selected.shape[0] >= 0
    assert bbox_selected.shape[1] == 7
    assert score_selected.shape[0] >= 0
    assert labels.shape[0] >= 0

    # test get_boxes
    points = torch.stack(points, 0)
    results = self.get_bboxes(points, ret_dict, img_metas)
    assert results[0][0].tensor.shape[0] >= 0
    assert results[0][0].tensor.shape[1] == 7
    assert results[0][1].shape[0] >= 0
    assert results[0][2].shape[0] >= 0


def test_shape_aware_head_loss():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    bbox_head_cfg = _get_pts_bbox_head_cfg(
        'ssn/hv_ssn_secfpn_sbn-all_2x16_2x_lyft-3d.py')
    # modify bn config to avoid bugs caused by syncbn
    for task in bbox_head_cfg['tasks']:
        task['norm_cfg'] = dict(type='BN2d')

    from mmdet3d.models.builder import build_head
    self = build_head(bbox_head_cfg)
    self.cuda()
    assert len(self.heads) == 4
    assert isinstance(self.heads[0].conv_cls, torch.nn.modules.conv.Conv2d)
    assert self.heads[0].conv_cls.in_channels == 64
    assert self.heads[0].conv_cls.out_channels == 36
    assert self.heads[0].conv_reg.out_channels == 28
    assert self.heads[0].conv_dir_cls.out_channels == 8

    # test forward
    feats = list()
    feats.append(torch.rand([2, 384, 200, 200], dtype=torch.float32).cuda())
    (cls_score, bbox_pred, dir_cls_preds) = self.forward(feats)
    assert cls_score[0].shape == torch.Size([2, 420000, 9])
    assert bbox_pred[0].shape == torch.Size([2, 420000, 7])
    assert dir_cls_preds[0].shape == torch.Size([2, 420000, 2])

    # test loss
    gt_bboxes = [
        LiDARInstance3DBoxes(
            torch.tensor(
                [[-14.5695, -6.4169, -2.1054, 1.8830, 4.6720, 1.4840, 1.5587],
                 [25.7215, 3.4581, -1.3456, 1.6720, 4.4090, 1.5830, 1.5301]],
                dtype=torch.float32).cuda()),
        LiDARInstance3DBoxes(
            torch.tensor(
                [[-50.763, -3.5517, -0.99658, 1.7430, 4.4020, 1.6990, 1.7874],
                 [-68.720, 0.033, -0.75276, 1.7860, 4.9100, 1.6610, 1.7525]],
                dtype=torch.float32).cuda())
    ]
    gt_labels = list(torch.tensor([[4, 4], [4, 4]], dtype=torch.int64).cuda())
    input_metas = [{
        'sample_idx': 1234
    }, {
        'sample_idx': 2345
    }]  # fake input_metas

    losses = self.loss(cls_score, bbox_pred, dir_cls_preds, gt_bboxes,
                       gt_labels, input_metas)

    assert losses['loss_cls'][0] > 0
    assert losses['loss_bbox'][0] > 0
    assert losses['loss_dir'][0] > 0

    # test empty ground truth case
    gt_bboxes = list(torch.empty((2, 0, 7)).cuda())
    gt_labels = list(torch.empty((2, 0)).cuda())
    empty_gt_losses = self.loss(cls_score, bbox_pred, dir_cls_preds, gt_bboxes,
                                gt_labels, input_metas)
    assert empty_gt_losses['loss_cls'][0] > 0
    assert empty_gt_losses['loss_bbox'][0] == 0
    assert empty_gt_losses['loss_dir'][0] == 0


def test_shape_aware_head_getboxes():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    bbox_head_cfg = _get_pts_bbox_head_cfg(
        'ssn/hv_ssn_secfpn_sbn-all_2x16_2x_lyft-3d.py')
    # modify bn config to avoid bugs caused by syncbn
    for task in bbox_head_cfg['tasks']:
        task['norm_cfg'] = dict(type='BN2d')

    from mmdet3d.models.builder import build_head
    self = build_head(bbox_head_cfg)
    self.cuda()

    feats = list()
    feats.append(torch.rand([2, 384, 200, 200], dtype=torch.float32).cuda())
    # fake input_metas
    input_metas = [{
        'sample_idx': 1234,
        'box_type_3d': LiDARInstance3DBoxes,
        'box_mode_3d': Box3DMode.LIDAR
    }, {
        'sample_idx': 2345,
        'box_type_3d': LiDARInstance3DBoxes,
        'box_mode_3d': Box3DMode.LIDAR
    }]
    (cls_score, bbox_pred, dir_cls_preds) = self.forward(feats)

    # test get_bboxes
    cls_score[0] -= 1.5  # too many positive samples may cause cuda oom
    result_list = self.get_bboxes(cls_score, bbox_pred, dir_cls_preds,
                                  input_metas)
    assert len(result_list[0][1]) > 0  # ensure not all boxes are filtered
    assert (result_list[0][1] > 0.3).all()
