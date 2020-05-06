import numpy as np

from mmdet.datasets.registry import PIPELINES


@PIPELINES.register_module
class PointSample(object):
    """Point Sample.

    Sampling data to a certain number.

    Args:
        name (str): Name of the dataset.
        num_points (int): Number of points to be sampled.
    """

    def __init__(self, num_points):
        self.num_points = num_points

    def points_random_sampling(self,
                               points,
                               num_samples,
                               replace=None,
                               return_choices=False):
        """Points Random Sampling.

        Sample points to a certain number.

        Args:
            points (ndarray): 3D Points.
            num_samples (int): Number of samples to be sampled.
            replace (bool): Whether the sample is with or without replacement.
            return_choices (bool): Whether return choice.

        Returns:
            points (ndarray): 3D Points.
            choices (ndarray): The generated random samples
        """
        if replace is None:
            replace = (points.shape[0] < num_samples)
        choices = np.random.choice(
            points.shape[0], num_samples, replace=replace)
        if return_choices:
            return points[choices], choices
        else:
            return points[choices]

    def __call__(self, results, seed=None):
        points = results.get('points', None)
        pts_color = results.get('pts_color', None)
        points, choices = self.points_random_sampling(
            points, self.num_points, return_choices=True)
        results['points'] = points

        if pts_color is not None:
            pts_color = pts_color[choices]
            instance_labels = results.get('instance_labels', None)
            semantic_labels = results.get('semantic_labels', None)
            instance_labels = instance_labels[choices]
            semantic_labels = semantic_labels[choices]
            results['instance_labels'] = instance_labels
            results['semantic_labels'] = semantic_labels
            results['pts_color'] = pts_color

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(num_points={})'.format(self.num_points)
        return repr_str
