# Copyright (c) OpenMMLab. All rights reserved.
import copy
import numpy as np
import pytest
import torch
from os.path import dirname, exists, join

from mmdet3d.models.builder import build_segmentor
from mmdet.apis import set_random_seed


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


def _get_segmentor_cfg(fname):
    """Grab configs necessary to create a segmentor.

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


def test_pointnet2_ssg():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')

    set_random_seed(0, True)
    pn2_ssg_cfg = _get_segmentor_cfg(
        'pointnet2/pointnet2_ssg_16x2_cosine_200e_scannet_seg-3d-20class.py')
    pn2_ssg_cfg.test_cfg.num_points = 32
    self = build_segmentor(pn2_ssg_cfg).cuda()
    points = [torch.rand(1024, 6).float().cuda() for _ in range(2)]
    img_metas = [dict(), dict()]
    gt_masks = [torch.randint(0, 20, (1024, )).long().cuda() for _ in range(2)]

    # test forward_train
    losses = self.forward_train(points, img_metas, gt_masks)
    assert losses['decode.loss_sem_seg'].item() >= 0

    # test forward function
    set_random_seed(0, True)
    data_dict = dict(
        points=points, img_metas=img_metas, pts_semantic_mask=gt_masks)
    forward_losses = self.forward(return_loss=True, **data_dict)
    assert np.allclose(losses['decode.loss_sem_seg'].item(),
                       forward_losses['decode.loss_sem_seg'].item())

    # test loss with ignore_index
    ignore_masks = [torch.ones_like(gt_masks[0]) * 20 for _ in range(2)]
    losses = self.forward_train(points, img_metas, ignore_masks)
    assert losses['decode.loss_sem_seg'].item() == 0

    # test simple_test
    self.eval()
    with torch.no_grad():
        scene_points = [
            torch.randn(500, 6).float().cuda() * 3.0,
            torch.randn(200, 6).float().cuda() * 2.5
        ]
        results = self.simple_test(scene_points, img_metas)
        assert results[0]['semantic_mask'].shape == torch.Size([500])
        assert results[1]['semantic_mask'].shape == torch.Size([200])

    # test forward function calling simple_test
    with torch.no_grad():
        data_dict = dict(points=[scene_points], img_metas=[img_metas])
        results = self.forward(return_loss=False, **data_dict)
        assert results[0]['semantic_mask'].shape == torch.Size([500])
        assert results[1]['semantic_mask'].shape == torch.Size([200])

    # test aug_test
    with torch.no_grad():
        scene_points = [
            torch.randn(2, 500, 6).float().cuda() * 3.0,
            torch.randn(2, 200, 6).float().cuda() * 2.5
        ]
        img_metas = [[dict(), dict()], [dict(), dict()]]
        results = self.aug_test(scene_points, img_metas)
        assert results[0]['semantic_mask'].shape == torch.Size([500])
        assert results[1]['semantic_mask'].shape == torch.Size([200])

    # test forward function calling aug_test
    with torch.no_grad():
        data_dict = dict(points=scene_points, img_metas=img_metas)
        results = self.forward(return_loss=False, **data_dict)
        assert results[0]['semantic_mask'].shape == torch.Size([500])
        assert results[1]['semantic_mask'].shape == torch.Size([200])


def test_pointnet2_msg():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')

    set_random_seed(0, True)
    pn2_msg_cfg = _get_segmentor_cfg(
        'pointnet2/pointnet2_msg_16x2_cosine_250e_scannet_seg-3d-20class.py')
    pn2_msg_cfg.test_cfg.num_points = 32
    self = build_segmentor(pn2_msg_cfg).cuda()
    points = [torch.rand(1024, 6).float().cuda() for _ in range(2)]
    img_metas = [dict(), dict()]
    gt_masks = [torch.randint(0, 20, (1024, )).long().cuda() for _ in range(2)]

    # test forward_train
    losses = self.forward_train(points, img_metas, gt_masks)
    assert losses['decode.loss_sem_seg'].item() >= 0

    # test loss with ignore_index
    ignore_masks = [torch.ones_like(gt_masks[0]) * 20 for _ in range(2)]
    losses = self.forward_train(points, img_metas, ignore_masks)
    assert losses['decode.loss_sem_seg'].item() == 0

    # test simple_test
    self.eval()
    with torch.no_grad():
        scene_points = [
            torch.randn(500, 6).float().cuda() * 3.0,
            torch.randn(200, 6).float().cuda() * 2.5
        ]
        results = self.simple_test(scene_points, img_metas)
        assert results[0]['semantic_mask'].shape == torch.Size([500])
        assert results[1]['semantic_mask'].shape == torch.Size([200])

    # test aug_test
    with torch.no_grad():
        scene_points = [
            torch.randn(2, 500, 6).float().cuda() * 3.0,
            torch.randn(2, 200, 6).float().cuda() * 2.5
        ]
        img_metas = [[dict(), dict()], [dict(), dict()]]
        results = self.aug_test(scene_points, img_metas)
        assert results[0]['semantic_mask'].shape == torch.Size([500])
        assert results[1]['semantic_mask'].shape == torch.Size([200])


def test_paconv_ssg():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')

    set_random_seed(0, True)
    paconv_ssg_cfg = _get_segmentor_cfg(
        'paconv/paconv_ssg_8x8_cosine_150e_s3dis_seg-3d-13class.py')
    # for GPU memory consideration
    paconv_ssg_cfg.backbone.num_points = (256, 64, 16, 4)
    paconv_ssg_cfg.test_cfg.num_points = 32
    self = build_segmentor(paconv_ssg_cfg).cuda()
    points = [torch.rand(1024, 9).float().cuda() for _ in range(2)]
    img_metas = [dict(), dict()]
    gt_masks = [torch.randint(0, 13, (1024, )).long().cuda() for _ in range(2)]

    # test forward_train
    losses = self.forward_train(points, img_metas, gt_masks)
    assert losses['decode.loss_sem_seg'].item() >= 0
    assert losses['regularize.loss_regularize'].item() >= 0

    # test forward function
    set_random_seed(0, True)
    data_dict = dict(
        points=points, img_metas=img_metas, pts_semantic_mask=gt_masks)
    forward_losses = self.forward(return_loss=True, **data_dict)
    assert np.allclose(losses['decode.loss_sem_seg'].item(),
                       forward_losses['decode.loss_sem_seg'].item())
    assert np.allclose(losses['regularize.loss_regularize'].item(),
                       forward_losses['regularize.loss_regularize'].item())

    # test loss with ignore_index
    ignore_masks = [torch.ones_like(gt_masks[0]) * 13 for _ in range(2)]
    losses = self.forward_train(points, img_metas, ignore_masks)
    assert losses['decode.loss_sem_seg'].item() == 0

    # test simple_test
    self.eval()
    with torch.no_grad():
        scene_points = [
            torch.randn(200, 6).float().cuda() * 3.0,
            torch.randn(100, 6).float().cuda() * 2.5
        ]
        results = self.simple_test(scene_points, img_metas)
        assert results[0]['semantic_mask'].shape == torch.Size([200])
        assert results[1]['semantic_mask'].shape == torch.Size([100])

    # test forward function calling simple_test
    with torch.no_grad():
        data_dict = dict(points=[scene_points], img_metas=[img_metas])
        results = self.forward(return_loss=False, **data_dict)
        assert results[0]['semantic_mask'].shape == torch.Size([200])
        assert results[1]['semantic_mask'].shape == torch.Size([100])

    # test aug_test
    with torch.no_grad():
        scene_points = [
            torch.randn(2, 200, 6).float().cuda() * 3.0,
            torch.randn(2, 100, 6).float().cuda() * 2.5
        ]
        img_metas = [[dict(), dict()], [dict(), dict()]]
        results = self.aug_test(scene_points, img_metas)
        assert results[0]['semantic_mask'].shape == torch.Size([200])
        assert results[1]['semantic_mask'].shape == torch.Size([100])

    # test forward function calling aug_test
    with torch.no_grad():
        data_dict = dict(points=scene_points, img_metas=img_metas)
        results = self.forward(return_loss=False, **data_dict)
        assert results[0]['semantic_mask'].shape == torch.Size([200])
        assert results[1]['semantic_mask'].shape == torch.Size([100])


def test_paconv_cuda_ssg():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')

    set_random_seed(0, True)
    paconv_cuda_ssg_cfg = _get_segmentor_cfg(
        'paconv/paconv_cuda_ssg_8x8_cosine_200e_s3dis_seg-3d-13class.py')
    # for GPU memory consideration
    paconv_cuda_ssg_cfg.backbone.num_points = (256, 64, 16, 4)
    paconv_cuda_ssg_cfg.test_cfg.num_points = 32
    self = build_segmentor(paconv_cuda_ssg_cfg).cuda()
    points = [torch.rand(1024, 9).float().cuda() for _ in range(2)]
    img_metas = [dict(), dict()]
    gt_masks = [torch.randint(0, 13, (1024, )).long().cuda() for _ in range(2)]

    # test forward_train
    losses = self.forward_train(points, img_metas, gt_masks)
    assert losses['decode.loss_sem_seg'].item() >= 0
    assert losses['regularize.loss_regularize'].item() >= 0

    # test forward function
    set_random_seed(0, True)
    data_dict = dict(
        points=points, img_metas=img_metas, pts_semantic_mask=gt_masks)
    forward_losses = self.forward(return_loss=True, **data_dict)
    assert np.allclose(losses['decode.loss_sem_seg'].item(),
                       forward_losses['decode.loss_sem_seg'].item())
    assert np.allclose(losses['regularize.loss_regularize'].item(),
                       forward_losses['regularize.loss_regularize'].item())

    # test loss with ignore_index
    ignore_masks = [torch.ones_like(gt_masks[0]) * 13 for _ in range(2)]
    losses = self.forward_train(points, img_metas, ignore_masks)
    assert losses['decode.loss_sem_seg'].item() == 0

    # test simple_test
    self.eval()
    with torch.no_grad():
        scene_points = [
            torch.randn(200, 6).float().cuda() * 3.0,
            torch.randn(100, 6).float().cuda() * 2.5
        ]
        results = self.simple_test(scene_points, img_metas)
        assert results[0]['semantic_mask'].shape == torch.Size([200])
        assert results[1]['semantic_mask'].shape == torch.Size([100])

    # test forward function calling simple_test
    with torch.no_grad():
        data_dict = dict(points=[scene_points], img_metas=[img_metas])
        results = self.forward(return_loss=False, **data_dict)
        assert results[0]['semantic_mask'].shape == torch.Size([200])
        assert results[1]['semantic_mask'].shape == torch.Size([100])

    # test aug_test
    with torch.no_grad():
        scene_points = [
            torch.randn(2, 200, 6).float().cuda() * 3.0,
            torch.randn(2, 100, 6).float().cuda() * 2.5
        ]
        img_metas = [[dict(), dict()], [dict(), dict()]]
        results = self.aug_test(scene_points, img_metas)
        assert results[0]['semantic_mask'].shape == torch.Size([200])
        assert results[1]['semantic_mask'].shape == torch.Size([100])

    # test forward function calling aug_test
    with torch.no_grad():
        data_dict = dict(points=scene_points, img_metas=img_metas)
        results = self.forward(return_loss=False, **data_dict)
        assert results[0]['semantic_mask'].shape == torch.Size([200])
        assert results[1]['semantic_mask'].shape == torch.Size([100])
