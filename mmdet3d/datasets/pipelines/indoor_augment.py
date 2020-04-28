import numpy as np

from mmdet.datasets.registry import PIPELINES


def _rotz(t):
    """Rotate About Z.

    Rotation about the z-axis.

    Args:
        t (float): Angle of rotation.

    Returns:
        rot_mat (ndarray): Matrix of rotation.
    """
    c = np.cos(t)
    s = np.sin(t)
    rot_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return rot_mat


def _rotate_aligned_boxes(input_boxes, rot_mat):
    centers, lengths = input_boxes[:, 0:3], input_boxes[:, 3:6]
    new_centers = np.dot(centers, np.transpose(rot_mat))

    dx, dy = lengths[:, 0] / 2.0, lengths[:, 1] / 2.0
    new_x = np.zeros((dx.shape[0], 4))
    new_y = np.zeros((dx.shape[0], 4))

    for i, crnr in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
        crnrs = np.zeros((dx.shape[0], 3))
        crnrs[:, 0] = crnr[0] * dx
        crnrs[:, 1] = crnr[1] * dy
        crnrs = np.dot(crnrs, np.transpose(rot_mat))
        new_x[:, i] = crnrs[:, 0]
        new_y[:, i] = crnrs[:, 1]

    new_dx = 2.0 * np.max(new_x, 1)
    new_dy = 2.0 * np.max(new_y, 1)
    new_lengths = np.stack((new_dx, new_dy, lengths[:, 2]), axis=1)

    return np.concatenate([new_centers, new_lengths], axis=1)


@PIPELINES.register_module()
class IndoorFlipData(object):
    """Indoor Flip Data

    Flip points and groundtruth boxes.

    Args:
        name (str): name of the dataset.
    """

    def __init__(self, name):
        assert name in ['scannet', 'sunrgbd']
        self.name = name

    def __call__(self, results):
        points = results.get('points', None)
        gt_boxes = results.get('gt_boxes', None)
        if np.random.random() > 0.5:
            # Flipping along the YZ plane
            points[:, 0] = -1 * points[:, 0]
            gt_boxes[:, 0] = -1 * gt_boxes[:, 0]
            if self.name == 'sunrgbd':
                gt_boxes[:, 6] = np.pi - gt_boxes[:, 6]
            results['gt_boxes'] = gt_boxes
        if self.name == 'scannet' and np.random.random() > 0.5:
            # Flipping along the XZ plane
            points[:, 1] = -1 * points[:, 1]
            gt_boxes[:, 1] = -1 * gt_boxes[:, 1]
            results['gt_boxes'] = gt_boxes
        results['points'] = points

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(dataset_name={})'.format(self.name)
        return repr_str


@PIPELINES.register_module()
class IndoorRotateData(object):
    """Indoor Rotate Data

    Rotate points and groundtruth boxes.

    Args:
        name (str): name of the dataset.
    """

    def __init__(self, name):
        assert name in ['scannet', 'sunrgbd']
        self.name = name

    def __call__(self, results):
        points = results.get('points', None)
        gt_boxes = results.get('gt_boxes', None)
        rot_angle = (np.random.random() * np.pi /
                     3) - np.pi / 6  # -30 ~ +30 degree
        rot_mat = _rotz(rot_angle)
        points[:, 0:3] = np.dot(points[:, 0:3], np.transpose(rot_mat))
        if self.name == 'scannet':
            gt_boxes = _rotate_aligned_boxes(gt_boxes, rot_mat)
        else:
            gt_boxes[:, 0:3] = np.dot(gt_boxes[:, 0:3], np.transpose(rot_mat))
            gt_boxes[:, 6] -= rot_angle
        results['points'] = points
        results['gt_boxes'] = gt_boxes
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(dataset_name={})'.format(self.name)
        return repr_str


@PIPELINES.register_module()
class IndoorShuffleData(object):
    """Indoor Shuffle Data

    Shuffle points.
    """

    def __init__(self):
        pass

    def __call__(self, results):
        points = results.get('points')
        np.random.shuffle(points)
        results['points'] = points
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
