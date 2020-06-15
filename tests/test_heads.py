import copy
from os.path import dirname, exists, join

import pytest
import torch

from mmdet3d.core.bbox import Box3DMode, LiDARInstance3DBoxes


def _get_config_directory():
    """ Find the predefined detector config directory """
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
    """
    Load a configuration as a python module
    """
    from mmcv import Config
    config_dpath = _get_config_directory()
    config_fpath = join(config_dpath, fname)
    config_mod = Config.fromfile(config_fpath)
    return config_mod


def _get_head_cfg(fname):
    """
    Grab configs necessary to create a bbox_head. These are deep copied to
    allow for safe modification of parameters without influencing other tests.
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
    """
    Grab configs necessary to create a rpn_head. These are deep copied to allow
    for safe modification of parameters without influencing other tests.
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


def test_anchor3d_head_loss():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    bbox_head_cfg = _get_head_cfg(
        'kitti/dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class.py')

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
    assert losses['loss_rpn_cls'][0] > 0
    assert losses['loss_rpn_bbox'][0] > 0
    assert losses['loss_rpn_dir'][0] > 0

    # test empty ground truth case
    gt_bboxes = list(torch.empty((2, 0, 7)).cuda())
    gt_labels = list(torch.empty((2, 0)).cuda())
    empty_gt_losses = self.loss(cls_score, bbox_pred, dir_cls_preds, gt_bboxes,
                                gt_labels, input_metas)
    assert empty_gt_losses['loss_rpn_cls'][0] > 0
    assert empty_gt_losses['loss_rpn_bbox'][0] == 0
    assert empty_gt_losses['loss_rpn_dir'][0] == 0


def test_anchor3d_head_getboxes():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    bbox_head_cfg = _get_head_cfg(
        'kitti/dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class.py')

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
        'kitti/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class.py')

    from mmdet3d.models.builder import build_head
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
    from mmdet3d.models.dense_heads import VoteHead
    bbox_head_cfg = dict(
        num_classes=10,
        bbox_coder=dict(
            type='PartialBinBasedBBoxCoder',
            num_sizes=10,
            num_dir_bins=5,
            with_rot=True,
            mean_sizes=[[2.114256, 1.620300, 0.927272],
                        [0.791118, 1.279516, 0.718182],
                        [0.923508, 1.867419, 0.845495],
                        [0.591958, 0.552978, 0.827272],
                        [0.699104, 0.454178, 0.75625],
                        [0.69519, 1.346299, 0.736364],
                        [0.528526, 1.002642, 1.172878],
                        [0.500618, 0.632163, 0.683424],
                        [0.404671, 1.071108, 1.688889],
                        [0.76584, 1.398258, 0.472728]]),
        vote_moudule_cfg=dict(
            in_channels=64,
            vote_per_seed=1,
            gt_per_seed=3,
            conv_channels=(64, 64),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d'),
            norm_feats=True,
            vote_loss=dict(
                type='ChamferDistance',
                mode='l1',
                reduction='none',
                loss_dst_weight=10.0)),
        vote_aggregation_cfg=dict(
            num_point=256,
            radius=0.3,
            num_sample=16,
            mlp_channels=[64, 32, 32, 32],
            use_xyz=True,
            normalize_xyz=True),
        feat_channels=(64, 64),
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        objectness_loss=dict(
            type='CrossEntropyLoss',
            class_weight=[0.2, 0.8],
            reduction='sum',
            loss_weight=5.0),
        center_loss=dict(
            type='ChamferDistance',
            mode='l2',
            reduction='sum',
            loss_src_weight=10.0,
            loss_dst_weight=10.0),
        dir_class_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        dir_res_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=10.0),
        size_class_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        size_res_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=10.0 / 3.0),
        semantic_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0))

    train_cfg = dict(
        pos_distance_thr=0.3, neg_distance_thr=0.6, sample_mod='vote')

    self = VoteHead(train_cfg=train_cfg, **bbox_head_cfg).cuda()
    fp_xyz = [torch.rand([2, 64, 3], dtype=torch.float32).cuda()]
    fp_features = [torch.rand([2, 64, 64], dtype=torch.float32).cuda()]
    fp_indices = [torch.randint(0, 128, [2, 64]).cuda()]

    input_dict = dict(
        fp_xyz=fp_xyz, fp_features=fp_features, fp_indices=fp_indices)
    # test forward
    ret_dict = self(input_dict, 'vote')
    assert ret_dict['center'].shape == torch.Size([2, 256, 3])
    assert ret_dict['obj_scores'].shape == torch.Size([2, 256, 2])
    assert ret_dict['size_res'].shape == torch.Size([2, 256, 10, 3])
    assert ret_dict['dir_res'].shape == torch.Size([2, 256, 5])
