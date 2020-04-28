import numpy as np

from mmdet.datasets.registry import PIPELINES


@PIPELINES.register_module()
class IndoorFlipData(object):
    """Indoor Flip Data

    Flip the points and groundtruth boxes.

    Args:
        name (str): name of the dataset.
    """

    def __init__(self, name):
        assert name in ['scannet', 'sunrgbd']
        self.name = name

    def __call__(self, results):
        point_cloud = results.get('point_cloud', None)
        gt_boxes = results.get('gt_boxes', None)
        if np.random.random() > 0.5:
            # Flipping along the YZ plane
            point_cloud[:, 0] = -1 * point_cloud[:, 0]
            gt_boxes[:, 0] = -1 * gt_boxes[:, 0]
            if self.name == 'sunrgbd':
                gt_boxes[:, 6] = np.pi - gt_boxes[:, 6]
            results['gt_boxes'] = gt_boxes
        if self.name == 'scannet' and np.random.random() > 0.5:
            # Flipping along the XZ plane
            point_cloud[:, 1] = -1 * point_cloud[:, 1]
            gt_boxes[:, 1] = -1 * gt_boxes[:, 1]
            results['gt_boxes'] = gt_boxes
        results['point_cloud'] = point_cloud

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(dataset_name={})'.format(self.name)
        return repr_str
