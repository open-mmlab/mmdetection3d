import os.path as osp

import mmcv
import numpy as np

from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadPointsFromFile(object):

    def __init__(self, points_dim=4, with_reflectivity=True):
        self.points_dim = points_dim
        self.with_reflectivity = with_reflectivity

    def __call__(self, results):
        if results['pts_prefix'] is not None:
            filename = osp.join(results['pts_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        points = np.fromfile(
            filename, dtype=np.float32).reshape(-1, self.points_dim)
        results['points'] = points
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(points_dim={})'.format(self.points_dim)
        repr_str += '(points_dim={})'.format(self.with_reflectivity)
        return repr_str


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles(object):
    """ Load multi channel images from a list of separate channel files.
    Expects results['filename'] to be a list of filenames
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        if results['img_prefix'] is not None:
            filename = [
                osp.join(results['img_prefix'], fname)
                for fname in results['img_info']['filename']
            ]
        else:
            filename = results['img_info']['filename']
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        return "{} (to_float32={}, color_type='{}')".format(
            self.__class__.__name__, self.to_float32, self.color_type)
