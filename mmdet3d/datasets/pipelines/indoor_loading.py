import os.path as osp

import numpy as np

from mmdet.datasets.registry import PIPELINES


@PIPELINES.register_module
class LoadPointsFromFile(object):
    """Load Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        use_height (bool): Whether to use height.
        color_mean (List[float]): Mean color of the point cloud.
        load_dim (int): The dimension of the loaded points.
            Default: 6.
        use_dim (List[int]): Which dimensions of the points to be used.
            Default: [0, 1, 2].
    """

    def __init__(self, use_height, color_mean, load_dim=6, use_dim=[0, 1, 2]):
        self.use_height = use_height
        self.color_mean = color_mean
        assert max(use_dim) < load_dim
        self.load_dim = load_dim
        self.use_dim = use_dim

    def __call__(self, results):
        pts_filename = results.get('pts_filename', None)
        info = results.get('info', None)
        name = 'scannet' if info.get('image', None) is None else 'sunrgbd'
        assert osp.exists(pts_filename)
        if name == 'scannet':
            points = np.load(pts_filename)
        else:
            points = np.load(pts_filename)['pc']
        points = points.reshape(-1, self.load_dim)
        if self.load_dim >= 6:
            points[:, 3:6] = points[:, 3:6] - np.array(self.color_mean) / 256.0
        points = points[:, self.use_dim]

        if self.use_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate([points, np.expand_dims(height, 1)], 1)
        results['points'] = points
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(use_height={})'.format(self.use_height)
        repr_str += '(mean_color={})'.format(self.color_mean)
        repr_str += '(load_dim={})'.format(self.load_dim)
        repr_str += '(use_dim={})'.format(self.use_dim)
        return repr_str


@PIPELINES.register_module
class LoadAnnotations3D(object):
    """Load Annotations3D.

    Load sunrgbd and scannet annotations.
    """

    def __init__(self):
        pass

    def __call__(self, results):
        ins_labelname = results.get('ins_labelname', None)
        sem_labelname = results.get('sem_labelname', None)
        info = results.get('info', None)
        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth']
            gt_labels = info['annos']['class'].reshape(-1, 1)
            gt_bboxes_3d_mask = np.ones_like(gt_labels)
        else:
            gt_bboxes_3d = np.zeros((1, 6), dtype=np.float32)
            gt_labels = np.zeros((1, 1))
            gt_bboxes_3d_mask = np.zeros((1, 1))

        if ins_labelname is not None and sem_labelname is not None:
            assert osp.exists(ins_labelname)
            assert osp.exists(sem_labelname)
            pts_instance_mask = np.load(ins_labelname)
            pts_semantic_mask = np.load(sem_labelname)
            results['pts_instance_mask'] = pts_instance_mask
            results['pts_semantic_mask'] = pts_semantic_mask

        results['gt_bboxes_3d'] = gt_bboxes_3d
        results['gt_labels'] = gt_labels
        results['gt_bboxes_3d_mask'] = gt_bboxes_3d_mask
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
