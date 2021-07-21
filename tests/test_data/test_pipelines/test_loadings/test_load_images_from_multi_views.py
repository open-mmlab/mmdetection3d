import numpy as np
import os
import torch
from mmcv.parallel import DataContainer

from mmdet3d.datasets.pipelines import (DefaultFormatBundle,
                                        LoadMultiViewImageFromFiles,
                                        MultiViewPipeline)


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


def test_multi_view_pipeline():
    file_names = ['00000.jpg', '00011.jpg', '00102.jpg']
    input_dict = dict(
        img_prefix=None,
        img_info=[
            dict(
                filename=os.path.join(
                    'tests/data/scannet/posed_images/scene0000_00', file_name))
            for file_name in file_names
        ],
        depth2img=[np.eye(4), np.eye(4) + 0.1,
                   np.eye(4) - 0.1])

    pipeline = MultiViewPipeline(
        transforms=[dict(type='LoadImageFromFile')], n_images=2)
    results = pipeline(input_dict)

    assert len(results['img']) == 2
    assert len(results['depth2img']) == 2
    shape = (968, 1296, 3)
    for img in results['img']:
        assert img.shape == shape
    file_names = set(img_info['filename'] for img_info in results['img_info'])
    assert len(file_names) == 2
    depth2img = set(str(x) for x in results['depth2img'])
    assert len(depth2img) == 2
