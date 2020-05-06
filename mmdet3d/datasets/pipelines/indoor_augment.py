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


@PIPELINES.register_module()
class IndoorFlipData(object):
    """Indoor Flip Data.

    Flip point_cloud and groundtruth boxes.

    Args:
        seed (int): Numpy random seed.
    """

    def __init__(self):
        pass

    def __call__(self, results):
        points = results.get('points', None)
        gt_bboxes_3d = results.get('gt_bboxes_3d', None)
        name = 'scannet' if gt_bboxes_3d.shape[1] == 6 else 'sunrgbd'
        if np.random.random() > 0.5:
            # Flipping along the YZ plane
            points[:, 0] = -1 * points[:, 0]
            gt_bboxes_3d[:, 0] = -1 * gt_bboxes_3d[:, 0]
            if name == 'sunrgbd':
                gt_bboxes_3d[:, 6] = np.pi - gt_bboxes_3d[:, 6]
            results['gt_boxes'] = gt_bboxes_3d

        if name == 'scannet' and np.random.random() > 0.5:
            # Flipping along the XZ plane
            points[:, 1] = -1 * points[:, 1]
            gt_bboxes_3d[:, 1] = -1 * gt_bboxes_3d[:, 1]
            results['gt_bboxes_3d'] = gt_bboxes_3d
        results['points'] = points

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


# TODO: merge outdoor indoor transform.
# TODO: try transform noise.
@PIPELINES.register_module()
class IndoorGlobalRotScale(object):
    """Indoor Global Rotate Scale.

    Augment sunrgbd and scannet data with global rotating and scaling.

    Args:
        seed (int): Numpy random seed.
        use_rotate (bool): Whether to use rotate.
        use_color (bool): Whether to use color.
        use_height (bool): Whether to use height.
        rot_range (float): Range of rotation.
        scale_range (float): Range of scale.
        (List[float]): Mean color of the point cloud.
    """

    def __init__(self,
                 use_rotate=True,
                 use_color=False,
                 use_scale=True,
                 use_height=True,
                 rot_range=[-np.pi / 6, np.pi / 6],
                 scale_range=[0.85, 1.15],
                 color_mean=[0.5, 0.5, 0.5]):
        self.use_rotate = use_rotate
        self.use_color = use_color
        self.use_scale = use_scale
        self.use_height = use_height
        self.rot_range = rot_range
        self.scale_range = scale_range
        self.color_mean = color_mean

    def _rotate_aligned_boxes(self, input_boxes, rot_mat):
        """Rotate Aligned Boxes.

        Rotate function for the aligned boxes.

        Args:
            input_boxes (ndarray): 3D boxes.
            rot_mat (ndarray): Rotation matrix.

        Returns:
            rotated_boxes (ndarry): 3D boxes after rotation.
        """
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

    def __call__(self, results):
        points = results.get('points', None)
        gt_bboxes_3d = results.get('gt_bboxes_3d', None)
        name = 'scannet' if gt_bboxes_3d.shape[1] == 6 else 'sunrgbd'

        if self.use_rotate:
            rot_angle = np.random.random() * (
                self.rot_range[1] - self.rot_range[0]) + self.rot_range[0]
            rot_mat = _rotz(rot_angle)
            points[:, 0:3] = np.dot(points[:, 0:3], np.transpose(rot_mat))

            if name == 'scannet':
                gt_bboxes_3d = self._rotate_aligned_boxes(
                    gt_bboxes_3d, rot_mat)
            else:
                gt_bboxes_3d[:, 0:3] = np.dot(gt_bboxes_3d[:, 0:3],
                                              np.transpose(rot_mat))
                gt_bboxes_3d[:, 6] -= rot_angle

        # Augment RGB color
        if self.use_color:
            rgb_color = points[:, 3:6] + self.color_mean
            # brightness change for each channel
            rgb_color *= (1 + 0.4 * np.random.random(3) - 0.2)
            # color shift for each channel
            rgb_color += (0.1 * np.random.random(3) - 0.05)
            # jittering on each pixel
            rgb_color += np.expand_dims(
                (0.05 * np.random.random(points.shape[0]) - 0.025), -1)
            rgb_color = np.clip(rgb_color, 0, 1)
            # randomly drop out 30% of the points' colors
            rgb_color *= np.expand_dims(
                np.random.random(points.shape[0]) > 0.3, -1)
            points[:, 3:6] = rgb_color - self.color_mean

        if self.use_scale:
            # Augment point cloud scale
            scale_ratio = np.random.random() * (
                self.scale_range[1] -
                self.scale_range[0]) + self.scale_range[0]
            scale_ratio = np.tile(scale_ratio, 3)[None, ...]
            points[:, 0:3] *= scale_ratio
            gt_bboxes_3d[:, 0:3] *= scale_ratio
            gt_bboxes_3d[:, 3:6] *= scale_ratio
            if self.use_height:
                points[:, -1] *= scale_ratio[0, 0]

        results['points'] = points
        results['gt_bboxes_3d'] = gt_bboxes_3d
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(use_rotate={})'.format(self.use_rotate)
        repr_str += '(use_color={})'.format(self.use_color)
        repr_str += '(use_scale={})'.format(self.use_scale)
        repr_str += '(use_height={})'.format(self.use_height)
        return repr_str
