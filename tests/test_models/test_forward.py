# Copyright (c) OpenMMLab. All rights reserved.
"""Test model forward process.

CommandLine:
    pytest tests/test_models/test_forward.py
    xdoctest tests/test_models/test_forward.py zero
"""
import copy
import numpy as np
import torch
from os.path import dirname, exists, join


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


def _get_detector_cfg(fname):
    """Grab configs necessary to create a detector.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    return model


def _test_two_stage_forward(cfg_file):
    model = _get_detector_cfg(cfg_file)
    model['pretrained'] = None

    from mmdet.models import build_detector
    detector = build_detector(model)

    input_shape = (1, 3, 256, 256)

    # Test forward train with a non-empty truth batch
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[10])
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']
    gt_masks = mm_inputs['gt_masks']
    losses = detector.forward(
        imgs,
        img_metas,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        gt_masks=gt_masks,
        return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = detector._parse_losses(losses)
    loss.requires_grad_(True)
    assert float(loss.item()) > 0
    loss.backward()

    # Test forward train with an empty truth batch
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[0])
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']
    gt_masks = mm_inputs['gt_masks']
    losses = detector.forward(
        imgs,
        img_metas,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        gt_masks=gt_masks,
        return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = detector._parse_losses(losses)
    assert float(loss.item()) > 0
    loss.backward()

    # Test forward test
    with torch.no_grad():
        img_list = [g[None, :] for g in imgs]
        batch_results = []
        for one_img, one_meta in zip(img_list, img_metas):
            result = detector.forward([one_img], [[one_meta]],
                                      return_loss=False)
            batch_results.append(result)


def _test_single_stage_forward(cfg_file):
    model = _get_detector_cfg(cfg_file)
    model['pretrained'] = None

    from mmdet.models import build_detector
    detector = build_detector(model)

    input_shape = (1, 3, 300, 300)
    mm_inputs = _demo_mm_inputs(input_shape)

    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')

    # Test forward train
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']
    losses = detector.forward(
        imgs,
        img_metas,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = detector._parse_losses(losses)
    assert float(loss.item()) > 0

    # Test forward test
    with torch.no_grad():
        img_list = [g[None, :] for g in imgs]
        batch_results = []
        for one_img, one_meta in zip(img_list, img_metas):
            result = detector.forward([one_img], [[one_meta]],
                                      return_loss=False)
            batch_results.append(result)


def _demo_mm_inputs(input_shape=(1, 3, 300, 300),
                    num_items=None, num_classes=10):  # yapf: disable
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions

        num_items (None | List[int]):
            specifies the number of boxes in each batch item

        num_classes (int):
            number of different labels a box might have
    """
    from mmdet.core import BitmapMasks

    (N, C, H, W) = input_shape

    rng = np.random.RandomState(0)

    imgs = rng.rand(*input_shape)

    img_metas = [{
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False,
    } for _ in range(N)]

    gt_bboxes = []
    gt_labels = []
    gt_masks = []

    for batch_idx in range(N):
        if num_items is None:
            num_boxes = rng.randint(1, 10)
        else:
            num_boxes = num_items[batch_idx]

        cx, cy, bw, bh = rng.rand(num_boxes, 4).T

        tl_x = ((cx * W) - (W * bw / 2)).clip(0, W)
        tl_y = ((cy * H) - (H * bh / 2)).clip(0, H)
        br_x = ((cx * W) + (W * bw / 2)).clip(0, W)
        br_y = ((cy * H) + (H * bh / 2)).clip(0, H)

        boxes = np.vstack([tl_x, tl_y, br_x, br_y]).T
        class_idxs = rng.randint(1, num_classes, size=num_boxes)

        gt_bboxes.append(torch.FloatTensor(boxes))
        gt_labels.append(torch.LongTensor(class_idxs))

    mask = np.random.randint(0, 2, (len(boxes), H, W), dtype=np.uint8)
    gt_masks.append(BitmapMasks(mask, H, W))

    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True),
        'img_metas': img_metas,
        'gt_bboxes': gt_bboxes,
        'gt_labels': gt_labels,
        'gt_bboxes_ignore': None,
        'gt_masks': gt_masks,
    }
    return mm_inputs
