import numpy as np

from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class IndoorFlipData(object):
    """Indoor flip data.

    Flip point cloud and ground truth boxes.
    The point cloud will be flipped along the yz plane
    and the xz plane with a certain probability.

    Args:
        flip_ratio_yz (float): Probability of being flipped along yz plane.
            Default: 0.5.
        flip_ratio_xz (float): Probability of being flipped along xz plane.
            Default: 0.5.
    """

    def __init__(self, flip_ratio_yz=0.5, flip_ratio_xz=0.5):
        self.flip_ratio_yz = flip_ratio_yz
        self.flip_ratio_xz = flip_ratio_xz

    def __call__(self, results):
        points = results['points']
        gt_bboxes_3d = results['gt_bboxes_3d']
        results['flip_yz'] = False
        results['flip_xz'] = False
        if np.random.random() < self.flip_ratio_yz:
            # Flipping along the YZ plane
            points[:, 0] = -1 * points[:, 0]
            gt_bboxes_3d.flip('horizontal')
            results['flip_yz'] = True

        if not gt_bboxes_3d.with_yaw and np.random.random(
        ) < self.flip_ratio_xz:
            # Flipping along the XZ plane
            points[:, 1] = -1 * points[:, 1]
            gt_bboxes_3d.flip('vertical')
            results['flip_xz'] = True

        results['points'] = points
        results['gt_bboxes_3d'] = gt_bboxes_3d
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(flip_ratio_yz={})'.format(self.flip_ratio_yz)
        repr_str += '(flip_ratio_xz={})'.format(self.flip_ratio_xz)
        return repr_str


@PIPELINES.register_module()
class IndoorPointsColorJitter(object):
    """Indoor points color jitter.

    Randomly change the brightness and color of the point cloud, and
    drop out the points' colors with a certain range and probability.

    Args:
        color_mean (list[float]): Mean color of the point cloud.
            Default: [0.5, 0.5, 0.5].
        bright_range (list[float]): Range of brightness.
            Default: [0.8, 1.2].
        color_shift_range (list[float]): Range of color shift.
            Default: [0.95, 1.05].
        jitter_range (list[float]): Range of jittering.
            Default: [-0.025, 0.025].
        drop_prob (float): Probability to drop out points' color.
            Default: 0.3
    """

    def __init__(self,
                 color_mean=[0.5, 0.5, 0.5],
                 bright_range=[0.8, 1.2],
                 color_shift_range=[0.95, 1.05],
                 jitter_range=[-0.025, 0.025],
                 drop_prob=0.3):
        self.color_mean = color_mean
        self.bright_range = bright_range
        self.color_shift_range = color_shift_range
        self.jitter_range = jitter_range
        self.drop_prob = drop_prob

    def __call__(self, results):
        points = results['points']
        assert points.shape[1] >= 6, \
            f'Expect points have channel >=6, got {points.shape[1]}.'
        rgb_color = points[:, 3:6] + self.color_mean
        # brightness change for each channel
        rgb_color *= np.random.uniform(self.bright_range[0],
                                       self.bright_range[1], 3)
        # color shift for each channel
        rgb_color += np.random.uniform(self.color_shift_range[0],
                                       self.color_shift_range[1], 3)
        # jittering on each pixel
        rgb_color += np.expand_dims(
            np.random.uniform(self.jitter_range[0], self.jitter_range[1]), -1)
        rgb_color = np.clip(rgb_color, 0, 1)
        # randomly drop out points' colors
        rgb_color *= np.expand_dims(
            np.random.random(points.shape[0]) > self.drop_prob, -1)
        points[:, 3:6] = rgb_color - self.color_mean
        results['points'] = points
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(color_mean={})'.format(self.color_mean)
        repr_str += '(bright_range={})'.format(self.bright_range)
        repr_str += '(color_shift_range={})'.format(self.color_shift_range)
        repr_str += '(jitter_range={})'.format(self.jitter_range)
        repr_str += '(drop_prob={})'.format(self.drop_prob)


# TODO: merge outdoor indoor transform.
# TODO: try transform noise.
@PIPELINES.register_module()
class IndoorGlobalRotScaleTrans(object):
    """Indoor global rotate and scale.

    Augment sunrgbd and scannet data with global rotating and scaling.

    Args:
        shift_height (bool): Whether to use height.
            Default: True.
        rot_range (list[float]): Range of rotation.
            Default: None.
        scale_range (list[float]): Range of scale.
            Default: None.
    """

    def __init__(self, shift_height=True, rot_range=None, scale_range=None):
        self.shift_height = shift_height
        self.rot_range = np.pi * np.array(rot_range)
        self.scale_range = scale_range

    def _rotz(self, t):
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

    def __call__(self, results):
        points = results['points']
        gt_bboxes_3d = results['gt_bboxes_3d']

        if self.rot_range is not None:
            assert len(self.rot_range) == 2, \
                f'Expect length of rot range =2, ' \
                f'got {len(self.rot_range)}.'
            rot_angle = np.random.uniform(self.rot_range[0], self.rot_range[1])
            if gt_bboxes_3d.tensor.shape[0] != 0:
                gt_bboxes_3d.rotate(rot_angle)
            points[:, :3] = np.dot(points[:, :3], self._rotz(rot_angle).T)
            results['rot_angle'] = rot_angle

        if self.scale_range is not None:
            assert len(self.scale_range) == 2, \
                f'Expect length of scale range =2, ' \
                f'got {len(self.scale_range)}.'
            # Augment point cloud scale
            scale_ratio = np.random.uniform(self.scale_range[0],
                                            self.scale_range[1])

            points[:, :3] *= scale_ratio
            gt_bboxes_3d.scale(scale_ratio)
            if self.shift_height:
                points[:, -1] *= scale_ratio

            results['scale_ratio'] = scale_ratio

        results['points'] = points
        results['gt_bboxes_3d'] = gt_bboxes_3d
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(shift_height={})'.format(self.shift_height)
        repr_str += '(rot_range={})'.format(self.rot_range)
        repr_str += '(scale_range={})'.format(self.scale_range)
        return repr_str
