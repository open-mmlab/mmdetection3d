# Copyright (c) OpenMMLab. All rights reserved.
from os.path import dirname, exists, join, relpath


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


def test_config_build_model():
    """Test that all detection models defined in the configs can be
    initialized."""
    from mmcv import Config

    from mmdet3d.models import build_model

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
        config_mod.model.train_cfg
        config_mod.model.test_cfg
        print('Building detector, config_fpath = {!r}'.format(config_fpath))

        # Remove pretrained keys to allow for testing in an offline environment
        if 'pretrained' in config_mod.model:
            config_mod.model['pretrained'] = None

        detector = build_model(config_mod.model)
        assert detector is not None

        if 'roi_head' in config_mod.model.keys():
            # for two stage detector
            # detectors must have bbox head
            assert detector.roi_head.with_bbox and detector.with_bbox
            assert detector.roi_head.with_mask == detector.with_mask

            head_config = config_mod.model['roi_head']
            if head_config.type == 'PartAggregationROIHead':
                check_parta2_roi_head(head_config, detector.roi_head)
            elif head_config.type == 'H3DRoIHead':
                check_h3d_roi_head(head_config, detector.roi_head)
            elif head_config.type == 'PointRCNNRoIHead':
                check_pointrcnn_roi_head(head_config, detector.roi_head)
            else:
                _check_roi_head(head_config, detector.roi_head)
        # else:
        #     # for single stage detector
        #     # detectors must have bbox head
        #     # assert detector.with_bbox
        #     head_config = config_mod.model['bbox_head']
        #     _check_bbox_head(head_config, detector.bbox_head)


def test_config_build_pipeline():
    """Test that all detection models defined in the configs can be
    initialized."""
    from mmcv import Config

    from mmdet3d.datasets.pipelines import Compose

    config_dpath = _get_config_directory()
    print('Found config_dpath = {!r}'.format(config_dpath))

    # Other configs needs database sampler.
    config_names = [
        'pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d.py',
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
    from torch import nn as nn
    if isinstance(roi_extractor, nn.ModuleList):
        if prev_roi_extractor:
            prev_roi_extractor = prev_roi_extractor[0]
        roi_extractor = roi_extractor[0]

    assert (len(config.featmap_strides) == len(roi_extractor.roi_layers))
    assert (config.out_channels == roi_extractor.out_channels)
    from torch.nn.modules.utils import _pair
    assert (_pair(config.roi_layer.output_size) ==
            roi_extractor.roi_layers[0].output_size)

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
    from torch import nn as nn
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
    from torch import nn as nn
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


def check_parta2_roi_head(config, head):
    assert config['type'] == head.__class__.__name__

    # check seg_roi_extractor
    seg_roi_cfg = config.seg_roi_extractor
    seg_roi_extractor = head.seg_roi_extractor
    _check_parta2_roi_extractor(seg_roi_cfg, seg_roi_extractor)

    # check part_roi_extractor
    part_roi_cfg = config.part_roi_extractor
    part_roi_extractor = head.part_roi_extractor
    _check_parta2_roi_extractor(part_roi_cfg, part_roi_extractor)

    # check bbox head infos
    bbox_cfg = config.bbox_head
    bbox_head = head.bbox_head
    _check_parta2_bbox_head(bbox_cfg, bbox_head)


def _check_parta2_roi_extractor(config, roi_extractor):
    assert config['type'] == roi_extractor.__class__.__name__
    assert (config.roi_layer.out_size == roi_extractor.roi_layer.out_size)
    assert (config.roi_layer.max_pts_per_voxel ==
            roi_extractor.roi_layer.max_pts_per_voxel)


def _check_parta2_bbox_head(bbox_cfg, bbox_head):
    from torch import nn as nn
    if isinstance(bbox_cfg, list):
        for single_bbox_cfg, single_bbox_head in zip(bbox_cfg, bbox_head):
            _check_bbox_head(single_bbox_cfg, single_bbox_head)
    elif isinstance(bbox_head, nn.ModuleList):
        for single_bbox_head in bbox_head:
            _check_bbox_head(bbox_cfg, single_bbox_head)
    else:
        assert bbox_cfg['type'] == bbox_head.__class__.__name__
        assert bbox_cfg.seg_in_channels == bbox_head.seg_conv[0][0].in_channels
        assert bbox_cfg.part_in_channels == bbox_head.part_conv[0][
            0].in_channels


def check_h3d_roi_head(config, head):
    assert config['type'] == head.__class__.__name__

    # check seg_roi_extractor
    primitive_z_cfg = config.primitive_list[0]
    primitive_z_extractor = head.primitive_z
    _check_primitive_extractor(primitive_z_cfg, primitive_z_extractor)

    primitive_xy_cfg = config.primitive_list[1]
    primitive_xy_extractor = head.primitive_xy
    _check_primitive_extractor(primitive_xy_cfg, primitive_xy_extractor)

    primitive_line_cfg = config.primitive_list[2]
    primitive_line_extractor = head.primitive_line
    _check_primitive_extractor(primitive_line_cfg, primitive_line_extractor)

    # check bbox head infos
    bbox_cfg = config.bbox_head
    bbox_head = head.bbox_head
    _check_h3d_bbox_head(bbox_cfg, bbox_head)


def _check_primitive_extractor(config, primitive_extractor):
    assert config['type'] == primitive_extractor.__class__.__name__
    assert (config.num_dims == primitive_extractor.num_dims)
    assert (config.num_classes == primitive_extractor.num_classes)


def _check_h3d_bbox_head(bbox_cfg, bbox_head):
    assert bbox_cfg['type'] == bbox_head.__class__.__name__
    assert bbox_cfg.num_proposal * \
        6 == bbox_head.surface_center_matcher.num_point[0]
    assert bbox_cfg.num_proposal * \
        12 == bbox_head.line_center_matcher.num_point[0]
    assert bbox_cfg.suface_matching_cfg.mlp_channels[-1] * \
        18 == bbox_head.bbox_pred[0].in_channels


def check_pointrcnn_roi_head(config, head):
    assert config['type'] == head.__class__.__name__

    # check point_roi_extractor
    point_roi_cfg = config.point_roi_extractor
    point_roi_extractor = head.point_roi_extractor
    _check_pointrcnn_roi_extractor(point_roi_cfg, point_roi_extractor)
    # check pointrcnn rcnn bboxhead
    bbox_cfg = config.bbox_head
    bbox_head = head.bbox_head
    _check_pointrcnn_bbox_head(bbox_cfg, bbox_head)


def _check_pointrcnn_roi_extractor(config, roi_extractor):
    assert config['type'] == roi_extractor.__class__.__name__
    assert config.roi_layer.num_sampled_points == \
        roi_extractor.roi_layer.num_sampled_points


def _check_pointrcnn_bbox_head(bbox_cfg, bbox_head):
    assert bbox_cfg['type'] == bbox_head.__class__.__name__
    assert bbox_cfg.num_classes == bbox_head.num_classes
    assert bbox_cfg.with_corner_loss == bbox_head.with_corner_loss
