import mmcv
import numpy as np

from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class IndoorPointsColorNormalize(object):
    """Indoor Points Color Normalize

    Normalize color of the points.

    Args:
        color_mean (List[float]): Mean color of the point cloud.
    """

    def __init__(self, color_mean):
        self.color_mean = color_mean

    def __call__(self, results):
        points = results['points']
        assert points.shape[1] >= 6,\
            f'Expect points have channel >=6, got {points.shape[1]}'
        points[:, 3:6] = points[:, 3:6] - np.array(self.color_mean) / 256.0
        results['points'] = points
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(color_mean={})'.format(self.color_mean)
        return repr_str


@PIPELINES.register_module()
class IndoorLoadPointsFromFile(object):
    """Indoor Load Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        use_height (bool): Whether to use height.
        load_dim (int): The dimension of the loaded points.
            Default: 6.
        use_dim (List[int]): Which dimensions of the points to be used.
            Default: [0, 1, 2].
    """

    def __init__(self, use_height, load_dim=6, use_dim=[0, 1, 2]):
        self.use_height = use_height
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'

        self.load_dim = load_dim
        self.use_dim = use_dim

    def __call__(self, results):
        pts_filename = results['pts_filename']
        mmcv.check_file_exist(pts_filename)
        points = np.load(pts_filename)
        points = points.reshape(-1, self.load_dim)
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
class IndoorLoadAnnotations3D(object):
    """Indoor Load Annotations3D.

    Load instance mask and semantic mask of points.
    """

    def __init__(self):
        pass

    def __call__(self, results):
        pts_instance_mask_path = results['pts_instance_mask_path']
        pts_semantic_mask_path = results['pts_semantic_mask_path']

        mmcv.check_file_exist(pts_instance_mask_path)
        mmcv.check_file_exist(pts_semantic_mask_path)
        pts_instance_mask = np.load(pts_instance_mask_path).astype(np.int)
        pts_semantic_mask = np.load(pts_semantic_mask_path).astype(np.int)
        results['pts_instance_mask'] = pts_instance_mask
        results['pts_semantic_mask'] = pts_semantic_mask

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
