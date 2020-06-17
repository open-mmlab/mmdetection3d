import numpy as np
from mmcv.utils import build_from_cfg

from mmdet3d.core.bbox import box_np_ops
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import RandomFlip
from ..registry import OBJECTSAMPLERS
from .data_augment_utils import noise_per_object_v3_


@PIPELINES.register_module()
class RandomFlip3D(RandomFlip):
    """Flip the points & bbox.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        sync_2d (bool, optional): Whether to apply flip according to the 2D
            images. If True, it will apply the same flip as that to 2D images.
            If False, it will decide whether to flip randomly and independently
            to that of 2D images.
        flip_ratio (float, optional): The flipping probability.
    """

    def __init__(self, sync_2d=True, **kwargs):
        super(RandomFlip3D, self).__init__(**kwargs)
        self.sync_2d = sync_2d

    def random_flip_data_3d(self, input_dict):
        input_dict['points'][:, 1] = -input_dict['points'][:, 1]
        for key in input_dict['bbox3d_fields']:
            input_dict[key].flip()

    def __call__(self, input_dict):
        # filp 2D image and its annotations
        super(RandomFlip3D, self).__call__(input_dict)

        if self.sync_2d:
            input_dict['pcd_flip'] = input_dict['flip']
        else:
            flip = True if np.random.rand() < self.flip_ratio else False
            input_dict['pcd_flip'] = flip

        if input_dict['pcd_flip']:
            self.random_flip_data_3d(input_dict)
        return input_dict

    def __repr__(self):
        return self.__class__.__name__ + '(flip_ratio={}, sync_2d={})'.format(
            self.flip_ratio, self.sync_2d)


@PIPELINES.register_module()
class ObjectSample(object):
    """Sample GT objects to the data

    Args:
        db_sampler (dict): Config dict of the database sampler.
        sample_2d (bool): Whether to also paste 2D image patch to the images
            This should be true when applying multi-modality cut-and-paste.
    """

    def __init__(self, db_sampler, sample_2d=False):
        self.sampler_cfg = db_sampler
        self.sample_2d = sample_2d
        if 'type' not in db_sampler.keys():
            db_sampler['type'] = 'DataBaseSampler'
        self.db_sampler = build_from_cfg(db_sampler, OBJECTSAMPLERS)

    @staticmethod
    def remove_points_in_boxes(points, boxes):
        masks = box_np_ops.points_in_rbbox(points, boxes)
        points = points[np.logical_not(masks.any(-1))]
        return points

    def __call__(self, input_dict):
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']

        # change to float for blending operation
        points = input_dict['points']
        if self.sample_2d:
            img = input_dict['img']
            gt_bboxes_2d = input_dict['gt_bboxes']
            # Assume for now 3D & 2D bboxes are the same
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(),
                gt_labels_3d,
                gt_bboxes_2d=gt_bboxes_2d,
                img=img)
        else:
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(), gt_labels_3d, img=None)

        if sampled_dict is not None:
            sampled_gt_bboxes_3d = sampled_dict['gt_bboxes_3d']
            sampled_points = sampled_dict['points']
            sampled_gt_labels = sampled_dict['gt_labels_3d']

            gt_labels_3d = np.concatenate([gt_labels_3d, sampled_gt_labels],
                                          axis=0)
            gt_bboxes_3d = gt_bboxes_3d.new_box(
                np.concatenate(
                    [gt_bboxes_3d.tensor.numpy(), sampled_gt_bboxes_3d]))

            points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
            # check the points dimension
            dim_inds = points.shape[-1]
            points = np.concatenate([sampled_points[:, :dim_inds], points],
                                    axis=0)

            if self.sample_2d:
                sampled_gt_bboxes_2d = sampled_dict['gt_bboxes_2d']
                gt_bboxes_2d = np.concatenate(
                    [gt_bboxes_2d, sampled_gt_bboxes_2d]).astype(np.float32)

                input_dict['gt_bboxes'] = gt_bboxes_2d
                input_dict['img'] = sampled_dict['img']

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d
        input_dict['points'] = points

        return input_dict

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class ObjectNoise(object):
    """Apply noise to each GT objects in the scene

    Args:
        translation_std (list, optional): Standard deviation of the
            distribution where translation noise are sampled from.
            Defaults to [0.25, 0.25, 0.25].
        global_rot_range (list, optional): Global rotation to the scene.
            Defaults to [0.0, 0.0].
        rot_range (list, optional): Object rotation range.
            Defaults to [-0.15707963267, 0.15707963267].
        num_try (int, optional): Number of times to try if the noise applied is
            invalid. Defaults to 100.
    """

    def __init__(self,
                 translation_std=[0.25, 0.25, 0.25],
                 global_rot_range=[0.0, 0.0],
                 rot_range=[-0.15707963267, 0.15707963267],
                 num_try=100):
        self.translation_std = translation_std
        self.global_rot_range = global_rot_range
        self.rot_range = rot_range
        self.num_try = num_try

    def __call__(self, input_dict):
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        points = input_dict['points']

        # TODO: check this inplace function
        numpy_box = gt_bboxes_3d.tensor.numpy()
        noise_per_object_v3_(
            numpy_box,
            points,
            rotation_perturb=self.rot_range,
            center_noise_std=self.translation_std,
            global_random_rot_range=self.global_rot_range,
            num_try=self.num_try)

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d.new_box(numpy_box)
        input_dict['points'] = points
        return input_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(num_try={},'.format(self.num_try)
        repr_str += ' translation_std={},'.format(self.translation_std)
        repr_str += ' global_rot_range={},'.format(self.global_rot_range)
        repr_str += ' rot_range={})'.format(self.rot_range)
        return repr_str


@PIPELINES.register_module()
class GlobalRotScaleTrans(object):
    """Apply global rotation, scaling and translation to a 3D scene

    Args:
        rot_range (list[float]): Range of rotation angle.
            Default to [-0.78539816, 0.78539816] (close to [-pi/4, pi/4]).
        scale_ratio_range (list[float]): Range of scale ratio.
            Default to [0.95, 1.05].
        translation_std (list[float]): The standard deviation of ranslation
            noise. This apply random translation to a scene by a noise, which
            is sampled from a gaussian distribution whose standard deviation
            is set by ``translation_std``. Default to [0, 0, 0]
    """

    def __init__(self,
                 rot_range=[-0.78539816, 0.78539816],
                 scale_ratio_range=[0.95, 1.05],
                 translation_std=[0, 0, 0]):
        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std

    def _trans_bbox_points(self, input_dict):
        if not isinstance(self.translation_std, (list, tuple, np.ndarray)):
            translation_std = [
                self.translation_std, self.translation_std,
                self.translation_std
            ]
        else:
            translation_std = self.translation_std
        translation_std = np.array(translation_std, dtype=np.float32)
        trans_factor = np.random.normal(scale=translation_std, size=3).T

        input_dict['points'][:, :3] += trans_factor
        input_dict['pcd_trans'] = trans_factor
        for key in input_dict['bbox3d_fields']:
            input_dict[key].translate(trans_factor)

    def _rot_bbox_points(self, input_dict):
        rotation = self.rot_range
        if not isinstance(rotation, list):
            rotation = [-rotation, rotation]
        noise_rotation = np.random.uniform(rotation[0], rotation[1])

        points = input_dict['points']
        points[:, :3], rot_mat_T = box_np_ops.rotation_points_single_angle(
            points[:, :3], noise_rotation, axis=2)
        input_dict['points'] = points
        input_dict['pcd_rotation'] = rot_mat_T

        for key in input_dict['bbox3d_fields']:
            input_dict[key].rotate(noise_rotation)

    def _scale_bbox_points(self, input_dict):
        scale = input_dict['pcd_scale_factor']
        input_dict['points'][:, :3] *= scale
        for key in input_dict['bbox3d_fields']:
            input_dict[key].scale(scale)

    def _random_scale(self, input_dict):
        scale_factor = np.random.uniform(self.scale_ratio_range[0],
                                         self.scale_ratio_range[1])
        input_dict['pcd_scale_factor'] = scale_factor

    def __call__(self, input_dict):
        self._rot_bbox_points(input_dict)

        if 'pcd_scale_factor' not in input_dict:
            self._random_scale(input_dict)
        self._scale_bbox_points(input_dict)

        self._trans_bbox_points(input_dict)
        return input_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(rot_range={},'.format(self.rot_range)
        repr_str += ' scale_ratio_range={},'.format(self.scale_ratio_range)
        repr_str += ' translation_std={})'.format(self.translation_std)
        return repr_str


@PIPELINES.register_module()
class PointShuffle(object):

    def __call__(self, input_dict):
        np.random.shuffle(input_dict['points'])
        return input_dict

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class ObjectRangeFilter(object):

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)
        self.bev_range = self.pcd_range[[0, 1, 3, 4]]

    @staticmethod
    def filter_gt_box_outside_range(gt_bboxes_3d, limit_range):
        """remove gtbox outside training range.
        this function should be applied after other prep functions
        Args:
            gt_bboxes_3d ([type]): [description]
            limit_range ([type]): [description]
        """
        gt_bboxes_3d_bv = box_np_ops.center_to_corner_box2d(
            gt_bboxes_3d[:, [0, 1]], gt_bboxes_3d[:, [3, 3 + 1]],
            gt_bboxes_3d[:, 6])
        bounding_box = box_np_ops.minmax_to_corner_2d(
            np.asarray(limit_range)[np.newaxis, ...])
        ret = box_np_ops.points_in_convex_polygon_jit(
            gt_bboxes_3d_bv.reshape(-1, 2), bounding_box)
        return np.any(ret.reshape(-1, 4), axis=1)

    def __call__(self, input_dict):
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        mask = gt_bboxes_3d.in_range_bev(self.bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        gt_labels_3d = gt_labels_3d[mask.numpy().astype(np.bool)]

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d

        return input_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(point_cloud_range={})'.format(self.pcd_range.tolist())
        return repr_str


@PIPELINES.register_module()
class PointsRangeFilter(object):

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(
            point_cloud_range, dtype=np.float32)[np.newaxis, :]

    def __call__(self, input_dict):
        points = input_dict['points']
        points_mask = ((points[:, :3] >= self.pcd_range[:, :3])
                       & (points[:, :3] < self.pcd_range[:, 3:]))
        points_mask = points_mask[:, 0] & points_mask[:, 1] & points_mask[:, 2]
        clean_points = points[points_mask, :]
        input_dict['points'] = clean_points
        return input_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(point_cloud_range={})'.format(self.pcd_range.tolist())
        return repr_str


@PIPELINES.register_module()
class ObjectNameFilter(object):
    """Filter GT objects by their names

    Args:
        classes (list[str]): list of class names to be kept for training
    """

    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def __call__(self, input_dict):
        gt_labels_3d = input_dict['gt_labels_3d']
        gt_bboxes_mask = np.array([n in self.labels for n in gt_labels_3d],
                                  dtype=np.bool_)
        input_dict['gt_bboxes_3d'] = input_dict['gt_bboxes_3d'][gt_bboxes_mask]
        input_dict['gt_labels_3d'] = input_dict['gt_labels_3d'][gt_bboxes_mask]

        return input_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(classes={self.classes})'
        return repr_str
