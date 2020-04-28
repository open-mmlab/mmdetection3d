import numpy as np

from mmdet.datasets.registry import PIPELINES


def points_random_sampling(points,
                           num_samples,
                           replace=None,
                           return_choices=False):
    """Points Random Sampling

    Sample points to a certain number.

    Args:
        points (ndarray): 3D Points.
        num_samples (int): Number of samples to be sampled.
    """
    if replace is None:
        replace = (points.shape[0] < num_samples)
    choices = np.random.choice(points.shape[0], num_samples, replace=replace)
    if return_choices:
        return points[choices], choices
    else:
        return points[choices]


@PIPELINES.register_module
class IndoorSamplePoints(object):
    """Indoor Sample Points

    Sampling data to a certain number.

    Args:
        name (str): Name of the dataset.
        num_points (int): Number of points to be sampled.
    """

    def __init__(self, name, num_points):
        assert name in ['scannet', 'sunrgbd']
        self.name = name
        self.num_points = num_points

    def __call__(self, results):
        points = results.get('points', None)
        pcl_color = results.get('pcl_color', None)
        points, choices = points_random_sampling(
            points, self.num_points, return_choices=True)
        results['points'] = points

        if self.name == 'scannet':
            pcl_color = pcl_color[choices]
            instance_labels = results.get('instance_labels', None)
            semantic_labels = results.get('semantic_labels', None)
            instance_labels = instance_labels[choices]
            semantic_labels = semantic_labels[choices]
            results['instance_labels'] = instance_labels
            results['semantic_labels'] = semantic_labels
            results['pcl_color'] = pcl_color

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(num_points={})'.format(self.num_points)
        return repr_str
