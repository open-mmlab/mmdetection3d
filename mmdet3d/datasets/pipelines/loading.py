import mmcv
import numpy as np

from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles(object):
    """ Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        filename = results['img_filename']
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


@PIPELINES.register_module()
class LoadPointsFromMultiSweeps(object):

    def __init__(self, sweeps_num=10):
        self.sweeps_num = sweeps_num

    def __call__(self, results):
        points = results['points']
        points[:, 3] /= 255
        points[:, 4] = 0
        sweep_points_list = [points]
        ts = results['timestamp']

        for idx, sweep in enumerate(results['sweeps']):
            if idx >= self.sweeps_num:
                break
            points_sweep = np.fromfile(
                sweep['data_path'], dtype=np.float32,
                count=-1).reshape([-1, 5])
            sweep_ts = sweep['timestamp'] / 1e6
            points_sweep[:, 3] /= 255
            points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                'sensor2lidar_rotation'].T
            points_sweep[:, :3] += sweep['sensor2lidar_translation']
            points_sweep[:, 4] = ts - sweep_ts
            sweep_points_list.append(points_sweep)

        points = np.concatenate(sweep_points_list, axis=0)[:, [0, 1, 2, 4]]
        results['points'] = points
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'
