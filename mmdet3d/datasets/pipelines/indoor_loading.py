import os.path as osp

import numpy as np

from mmdet.datasets.registry import PIPELINES


@PIPELINES.register_module
class LoadPointsFromFile(object):
    """Load Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        name (str): scannet or sunrgbd.
        use_color (bool): Whether to use color.
        use_height (bool): Whether to use height.
        color_mean (List[float]): Mean color of the point cloud.
    """

    def __init__(self, use_color, use_height, color_mean):
        self.use_color = use_color
        self.use_height = use_height
        self.color_mean = color_mean

    def __call__(self, results):
        data_path = results.get('data_path', None)
        info = results.get('info', None)
        name = 'scannet' if info.get('image', None) is None else 'sunrgbd'
        if name == 'scannet':
            scan_name = info['point_cloud']['lidar_idx']
            point_cloud = self._get_lidar(scan_name, data_path)
        else:
            point_cloud = np.load(
                osp.join(data_path, 'lidar',
                         '%06d.npz' % info['point_cloud']['lidar_idx']))['pc']

        if not self.use_color:
            if name == 'scannet':
                pcl_color = point_cloud[:, 3:6]
            point_cloud = point_cloud[:, 0:3]
        else:
            if name == 'scannet':
                pcl_color = point_cloud[:, 3:6]
            point_cloud = point_cloud[:, 0:6]
            point_cloud[:, 3:] = (point_cloud[:, 3:] -
                                  np.array(self.color_mean)) / 256.0

        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate(
                [point_cloud, np.expand_dims(height, 1)], 1)
        results['point_cloud'] = point_cloud
        if name == 'scannet':
            results['pcl_color'] = pcl_color
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(use_height={})'.format(self.use_height)
        repr_str += '(use_color={}'.format(self.use_color)
        repr_str += '(mean_color={})'.format(self.color_mean)
        return repr_str

    def _get_lidar(self, scan_name, data_path):
        lidar_file = osp.join(data_path, scan_name + '_vert.npy')
        assert osp.exists(lidar_file)
        return np.load(lidar_file)


@PIPELINES.register_module
class LoadAnnotations3D(object):
    """Load Annotations3D.

    Load sunrgbd and scannet annotations.

    Args:
        name (str): scannet or sunrgbd.
    """

    def __init__(self):
        pass

    def __call__(self, results):
        data_path = results.get('data_path', None)
        info = results.get('info', None)
        if info['annos']['gt_num'] != 0:
            gt_boxes = info['annos']['gt_boxes_upright_depth']
            gt_classes = info['annos']['class'].reshape(-1, 1)
            gt_boxes_mask = np.ones_like(gt_classes)
        else:
            gt_boxes = np.zeros((1, 6), dtype=np.float32)
            gt_classes = np.zeros((1, 1))
            gt_boxes_mask = np.zeros((1, 1))
        name = 'scannet' if info.get('image', None) is None else 'sunrgbd'

        if name == 'scannet':
            scan_name = info['point_cloud']['lidar_idx']
            instance_labels = self._get_instance_label(scan_name, data_path)
            semantic_labels = self._get_semantic_label(scan_name, data_path)
            results['instance_labels'] = instance_labels
            results['semantic_labels'] = semantic_labels

        results['gt_boxes'] = gt_boxes
        results['gt_classes'] = gt_classes
        results['gt_boxes_mask'] = gt_boxes_mask
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

    def _get_instance_label(self, scan_name, data_path):
        ins_file = osp.join(data_path, scan_name + '_ins_label.npy')
        assert osp.exists(ins_file)
        return np.load(ins_file)

    def _get_semantic_label(self, scan_name, data_path):
        sem_file = osp.join(data_path, scan_name + '_sem_label.npy')
        assert osp.exists(sem_file)
        return np.load(sem_file)
