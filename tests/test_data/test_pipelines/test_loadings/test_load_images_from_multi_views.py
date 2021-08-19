# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.parallel import DataContainer

from mmdet3d.datasets.pipelines import (DefaultFormatBundle,
                                        LoadMultiViewImageFromFiles)


def test_load_multi_view_image_from_files():
    multi_view_img_loader = LoadMultiViewImageFromFiles(to_float32=True)

    num_views = 6
    filename = 'tests/data/waymo/kitti_format/training/image_0/0000000.png'
    filenames = [filename for _ in range(num_views)]

    input_dict = dict(img_filename=filenames)
    results = multi_view_img_loader(input_dict)
    img = results['img']
    img0 = img[0]
    img_norm_cfg = results['img_norm_cfg']

    assert isinstance(img, list)
    assert len(img) == num_views
    assert img0.dtype == np.float32
    assert results['filename'] == filenames
    assert results['img_shape'] == results['ori_shape'] == \
        results['pad_shape'] == (1280, 1920, 3, num_views)
    assert results['scale_factor'] == 1.0
    assert np.all(img_norm_cfg['mean'] == np.zeros(3, dtype=np.float32))
    assert np.all(img_norm_cfg['std'] == np.ones(3, dtype=np.float32))
    assert not img_norm_cfg['to_rgb']

    repr_str = repr(multi_view_img_loader)
    expected_str = 'LoadMultiViewImageFromFiles(to_float32=True, ' \
                   "color_type='unchanged')"
    assert repr_str == expected_str

    # test LoadMultiViewImageFromFiles's compatibility with DefaultFormatBundle
    # refer to https://github.com/open-mmlab/mmdetection3d/issues/227
    default_format_bundle = DefaultFormatBundle()
    results = default_format_bundle(results)
    img = results['img']

    assert isinstance(img, DataContainer)
    assert img._data.shape == torch.Size((num_views, 3, 1280, 1920))
