from os.path import dirname, exists, join, relpath

from mmdet.core import BitmapMasks, PolygonMasks


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


def test_config_build_detector():
    """
    Test that all detection models defined in the configs can be initialized.
    """
    from mmcv import Config
    from mmdet3d.models import build_detector

    config_dpath = _get_config_directory()
    print('Found config_dpath = {!r}'.format(config_dpath))

    import glob
    config_fpaths = list(glob.glob(join(config_dpath, '**', '*.py')))
    config_fpaths = [p for p in config_fpaths if p.find('_base_') == -1]
    config_names = [relpath(p, config_dpath) for p in config_fpaths]

    print('Using {} config files'.format(len(config_names)))

    for config_fname in config_names:
        config_fpath = join(config_dpath, config_fname)
        config_mod = Config.fromfile(config_fpath)

        config_mod.model
        config_mod.train_cfg
        config_mod.test_cfg
        print('Building detector, config_fpath = {!r}'.format(config_fpath))

        # Remove pretrained keys to allow for testing in an offline environment
        if 'pretrained' in config_mod.model:
            config_mod.model['pretrained'] = None

        detector = build_detector(
            config_mod.model,
            train_cfg=config_mod.train_cfg,
            test_cfg=config_mod.test_cfg)
        assert detector is not None

        if 'roi_head' in config_mod.model.keys():
            # for two stage detector
            # detectors must have bbox head
            assert detector.roi_head.with_bbox and detector.with_bbox
            assert detector.roi_head.with_mask == detector.with_mask

            head_config = config_mod.model['roi_head']
            _check_roi_head(head_config, detector.roi_head)
        # else:
        #     # for single stage detector
        #     # detectors must have bbox head
        #     # assert detector.with_bbox
        #     head_config = config_mod.model['bbox_head']
        #     _check_bbox_head(head_config, detector.bbox_head)


def test_config_build_pipeline():
    """
    Test that all detection models defined in the configs can be initialized.
    """
    from mmcv import Config
    from mmdet3d.datasets.pipelines import Compose

    config_dpath = _get_config_directory()
    print('Found config_dpath = {!r}'.format(config_dpath))

    # Other configs needs database sampler.
    config_names = [
        'nus/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d.py',
    ]

    print('Using {} config files'.format(len(config_names)))

    for config_fname in config_names:
        config_fpath = join(config_dpath, config_fname)
        config_mod = Config.fromfile(config_fpath)

        # build train_pipeline
        train_pipeline = Compose(config_mod.train_pipeline)
        test_pipeline = Compose(config_mod.test_pipeline)
        assert train_pipeline is not None
        assert test_pipeline is not None


def test_config_data_pipeline():
    """
    Test whether the data pipeline is valid and can process corner cases.
    CommandLine:
        xdoctest -m tests/test_config.py test_config_build_data_pipeline
    """
    from mmcv import Config
    from mmdet3d.datasets.pipelines import Compose
    import numpy as np

    config_dpath = _get_config_directory()
    print('Found config_dpath = {!r}'.format(config_dpath))

    # Only tests a representative subset of configurations
    # TODO: test pipelines using Albu, current Albu throw None given empty GT
    config_names = [
        'nus/faster_rcnn_r50_fpn_caffe_2x8_1x_nus.py',
        'nus/retinanet_r50_fpn_caffe_2x8_1x_nus.py',
        'kitti/'
        'faster_rcnn_r50_fpn_caffe_1x_kitti-2d-3class_coco-3x-pretrain.py',
    ]

    def dummy_masks(h, w, num_obj=3, mode='bitmap'):
        assert mode in ('polygon', 'bitmap')
        if mode == 'bitmap':
            masks = np.random.randint(0, 2, (num_obj, h, w), dtype=np.uint8)
            masks = BitmapMasks(masks, h, w)
        else:
            masks = []
            for i in range(num_obj):
                masks.append([])
                masks[-1].append(
                    np.random.uniform(0, min(h - 1, w - 1), (8 + 4 * i, )))
                masks[-1].append(
                    np.random.uniform(0, min(h - 1, w - 1), (10 + 4 * i, )))
            masks = PolygonMasks(masks, h, w)
        return masks

    print('Using {} config files'.format(len(config_names)))

    for config_fname in config_names:
        config_fpath = join(config_dpath, config_fname)
        config_mod = Config.fromfile(config_fpath)

        # remove loading pipeline
        loading_pipeline = config_mod.train_pipeline.pop(0)
        loading_ann_pipeline = config_mod.train_pipeline.pop(0)
        config_mod.test_pipeline.pop(0)

        train_pipeline = Compose(config_mod.train_pipeline)
        test_pipeline = Compose(config_mod.test_pipeline)

        print(
            'Building data pipeline, config_fpath = {!r}'.format(config_fpath))

        print('Test training data pipeline: \n{!r}'.format(train_pipeline))
        img = np.random.randint(0, 255, size=(888, 666, 3), dtype=np.uint8)
        if loading_pipeline.get('to_float32', False):
            img = img.astype(np.float32)
        mode = 'bitmap' if loading_ann_pipeline.get('poly2mask',
                                                    True) else 'polygon'
        results = dict(
            filename='test_img.png',
            img=img,
            img_shape=img.shape,
            ori_shape=img.shape,
            gt_bboxes=np.array([[35.2, 11.7, 39.7, 15.7]], dtype=np.float32),
            gt_labels=np.array([1], dtype=np.int64),
            gt_masks=dummy_masks(img.shape[0], img.shape[1], mode=mode),
        )
        results['bbox_fields'] = ['gt_bboxes']
        results['mask_fields'] = ['gt_masks']
        output_results = train_pipeline(results)
        assert output_results is not None

        print('Test testing data pipeline: \n{!r}'.format(test_pipeline))
        results = dict(
            filename='test_img.png',
            img=img,
            img_shape=img.shape,
            ori_shape=img.shape,
            gt_bboxes=np.array([[35.2, 11.7, 39.7, 15.7]], dtype=np.float32),
            gt_labels=np.array([1], dtype=np.int64),
            gt_masks=dummy_masks(img.shape[0], img.shape[1], mode=mode),
        )
        results['bbox_fields'] = ['gt_bboxes']
        results['mask_fields'] = ['gt_masks']
        output_results = test_pipeline(results)
        assert output_results is not None

        # test empty GT
        print('Test empty GT with training data pipeline: \n{!r}'.format(
            train_pipeline))
        results = dict(
            filename='test_img.png',
            img=img,
            img_shape=img.shape,
            ori_shape=img.shape,
            gt_bboxes=np.zeros((0, 4), dtype=np.float32),
            gt_labels=np.array([], dtype=np.int64),
            gt_masks=dummy_masks(
                img.shape[0], img.shape[1], num_obj=0, mode=mode),
        )
        results['bbox_fields'] = ['gt_bboxes']
        results['mask_fields'] = ['gt_masks']
        output_results = train_pipeline(results)
        assert output_results is not None

        print('Test empty GT with testing data pipeline: \n{!r}'.format(
            test_pipeline))
        results = dict(
            filename='test_img.png',
            img=img,
            img_shape=img.shape,
            ori_shape=img.shape,
            gt_bboxes=np.zeros((0, 4), dtype=np.float32),
            gt_labels=np.array([], dtype=np.int64),
            gt_masks=dummy_masks(
                img.shape[0], img.shape[1], num_obj=0, mode=mode),
        )
        results['bbox_fields'] = ['gt_bboxes']
        results['mask_fields'] = ['gt_masks']
        output_results = test_pipeline(results)
        assert output_results is not None


def _check_roi_head(config, head):
    # check consistency between head_config and roi_head
    assert config['type'] == head.__class__.__name__

    # check roi_align
    bbox_roi_cfg = config.bbox_roi_extractor
    bbox_roi_extractor = head.bbox_roi_extractor
    _check_roi_extractor(bbox_roi_cfg, bbox_roi_extractor)

    # check bbox head infos
    bbox_cfg = config.bbox_head
    bbox_head = head.bbox_head
    _check_bbox_head(bbox_cfg, bbox_head)

    if head.with_mask:
        # check roi_align
        if config.mask_roi_extractor:
            mask_roi_cfg = config.mask_roi_extractor
            mask_roi_extractor = head.mask_roi_extractor
            _check_roi_extractor(mask_roi_cfg, mask_roi_extractor,
                                 bbox_roi_extractor)

        # check mask head infos
        mask_head = head.mask_head
        mask_cfg = config.mask_head
        _check_mask_head(mask_cfg, mask_head)


def _check_roi_extractor(config, roi_extractor, prev_roi_extractor=None):
    import torch.nn as nn
    if isinstance(roi_extractor, nn.ModuleList):
        if prev_roi_extractor:
            prev_roi_extractor = prev_roi_extractor[0]
        roi_extractor = roi_extractor[0]

    assert (len(config.featmap_strides) == len(roi_extractor.roi_layers))
    assert (config.out_channels == roi_extractor.out_channels)
    from torch.nn.modules.utils import _pair
    assert (_pair(
        config.roi_layer.out_size) == roi_extractor.roi_layers[0].out_size)

    if 'use_torchvision' in config.roi_layer:
        assert (config.roi_layer.use_torchvision ==
                roi_extractor.roi_layers[0].use_torchvision)
    elif 'aligned' in config.roi_layer:
        assert (
            config.roi_layer.aligned == roi_extractor.roi_layers[0].aligned)

    if prev_roi_extractor:
        assert (roi_extractor.roi_layers[0].aligned ==
                prev_roi_extractor.roi_layers[0].aligned)
        assert (roi_extractor.roi_layers[0].use_torchvision ==
                prev_roi_extractor.roi_layers[0].use_torchvision)


def _check_mask_head(mask_cfg, mask_head):
    import torch.nn as nn
    if isinstance(mask_cfg, list):
        for single_mask_cfg, single_mask_head in zip(mask_cfg, mask_head):
            _check_mask_head(single_mask_cfg, single_mask_head)
    elif isinstance(mask_head, nn.ModuleList):
        for single_mask_head in mask_head:
            _check_mask_head(mask_cfg, single_mask_head)
    else:
        assert mask_cfg['type'] == mask_head.__class__.__name__
        assert mask_cfg.in_channels == mask_head.in_channels
        assert (
            mask_cfg.conv_out_channels == mask_head.conv_logits.in_channels)
        class_agnostic = mask_cfg.get('class_agnostic', False)
        out_dim = (1 if class_agnostic else mask_cfg.num_classes)
        assert mask_head.conv_logits.out_channels == out_dim


def _check_bbox_head(bbox_cfg, bbox_head):
    import torch.nn as nn
    if isinstance(bbox_cfg, list):
        for single_bbox_cfg, single_bbox_head in zip(bbox_cfg, bbox_head):
            _check_bbox_head(single_bbox_cfg, single_bbox_head)
    elif isinstance(bbox_head, nn.ModuleList):
        for single_bbox_head in bbox_head:
            _check_bbox_head(bbox_cfg, single_bbox_head)
    else:
        assert bbox_cfg['type'] == bbox_head.__class__.__name__
        assert bbox_cfg.in_channels == bbox_head.in_channels
        with_cls = bbox_cfg.get('with_cls', True)
        if with_cls:
            fc_out_channels = bbox_cfg.get('fc_out_channels', 2048)
            assert (fc_out_channels == bbox_head.fc_cls.in_features)
            assert bbox_cfg.num_classes + 1 == bbox_head.fc_cls.out_features

        with_reg = bbox_cfg.get('with_reg', True)
        if with_reg:
            out_dim = (4 if bbox_cfg.reg_class_agnostic else 4 *
                       bbox_cfg.num_classes)
            assert bbox_head.fc_reg.out_features == out_dim
