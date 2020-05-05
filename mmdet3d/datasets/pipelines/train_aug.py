import mmcv
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
        flip_ratio (float, optional): The flipping probability.
    """

    def __init__(self, sync_2d=True, **kwargs):
        super(RandomFlip3D, self).__init__(**kwargs)
        self.sync_2d = sync_2d

    def random_flip_points(self, gt_bboxes_3d, points):
        gt_bboxes_3d[:, 1] = -gt_bboxes_3d[:, 1]
        gt_bboxes_3d[:, 6] = -gt_bboxes_3d[:, 6] + np.pi
        points[:, 1] = -points[:, 1]
        if gt_bboxes_3d.shape[1] == 9:
            # flip velocitys at the same time
            gt_bboxes_3d[:, 8] = -gt_bboxes_3d[:, 8]
        return gt_bboxes_3d, points

    def __call__(self, input_dict):
        # filp 2D image and its annotations
        if 'flip' not in input_dict:
            flip = True if np.random.rand() < self.flip_ratio else False
            input_dict['flip'] = flip
        if 'flip_direction' not in input_dict:
            input_dict['flip_direction'] = self.direction
        if input_dict['flip']:
            # flip image
            if 'img' in input_dict:
                if isinstance(input_dict['img'], list):
                    input_dict['img'] = [
                        mmcv.imflip(
                            img, direction=input_dict['flip_direction'])
                        for img in input_dict['img']
                    ]
                else:
                    input_dict['img'] = mmcv.imflip(
                        input_dict['img'],
                        direction=input_dict['flip_direction'])
            # flip bboxes
            for key in input_dict.get('bbox_fields', []):
                input_dict[key] = self.bbox_flip(input_dict[key],
                                                 input_dict['img_shape'],
                                                 input_dict['flip_direction'])
            # flip masks
            for key in input_dict.get('mask_fields', []):
                input_dict[key] = [
                    mmcv.imflip(mask, direction=input_dict['flip_direction'])
                    for mask in input_dict[key]
                ]

            # flip segs
            for key in input_dict.get('seg_fields', []):
                input_dict[key] = mmcv.imflip(
                    input_dict[key], direction=input_dict['flip_direction'])

        if self.sync_2d:
            input_dict['pcd_flip'] = input_dict['flip']
        else:
            flip = True if np.random.rand() < self.flip_ratio else False
            input_dict['pcd_flip'] = flip
        if input_dict['pcd_flip']:
            # flip image
            gt_bboxes_3d = input_dict['gt_bboxes_3d']
            points = input_dict['points']
            gt_bboxes_3d, points = self.random_flip_points(
                gt_bboxes_3d, points)
            input_dict['gt_bboxes_3d'] = gt_bboxes_3d
            input_dict['points'] = points
        return input_dict

    def __repr__(self):
        return self.__class__.__name__ + '(flip_ratio={}, sync_2d={})'.format(
            self.flip_ratio, self.sync_2d)


@PIPELINES.register_module()
class ObjectSample(object):

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
        gt_names_3d = input_dict['gt_names_3d']
        gt_bboxes_3d_mask = input_dict['gt_bboxes_3d_mask']
        # change to float for blending operation
        points = input_dict['points']
        #         rect = input_dict['rect']
        #         Trv2c = input_dict['Trv2c']
        #         P2 = input_dict['P2']
        if self.sample_2d:
            img = input_dict['img']  # .astype(np.float32)
            gt_bboxes_2d = input_dict['gt_bboxes']
            gt_bboxes_mask = input_dict['gt_bboxes_mask']
            gt_names = input_dict['gt_names']
            # Assume for now 3D & 2D bboxes are the same
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d, gt_names_3d, gt_bboxes_2d=gt_bboxes_2d, img=img)
        else:
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d, gt_names_3d, img=None)

        if sampled_dict is not None:
            sampled_gt_names = sampled_dict['gt_names']
            sampled_gt_bboxes_3d = sampled_dict['gt_bboxes_3d']
            sampled_points = sampled_dict['points']
            sampled_gt_masks = sampled_dict['gt_masks']

            gt_names_3d = np.concatenate([gt_names_3d, sampled_gt_names],
                                         axis=0)
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, sampled_gt_bboxes_3d
                                           ]).astype(np.float32)
            gt_bboxes_3d_mask = np.concatenate(
                [gt_bboxes_3d_mask, sampled_gt_masks], axis=0)
            points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
            # check the points dimension
            dim_inds = points.shape[-1]
            points = np.concatenate([sampled_points[:, :dim_inds], points],
                                    axis=0)

            if self.sample_2d:
                sampled_gt_bboxes_2d = sampled_dict['gt_bboxes_2d']
                gt_bboxes_2d = np.concatenate(
                    [gt_bboxes_2d, sampled_gt_bboxes_2d]).astype(np.float32)
                gt_bboxes_mask = np.concatenate(
                    [gt_bboxes_mask, sampled_gt_masks], axis=0)
                gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
                input_dict['gt_names'] = gt_names
                input_dict['gt_bboxes'] = gt_bboxes_2d
                input_dict['gt_bboxes_mask'] = gt_bboxes_mask
                input_dict['img'] = sampled_dict['img']  # .astype(np.uint8)

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_names_3d'] = gt_names_3d
        input_dict['points'] = points
        input_dict['gt_bboxes_3d_mask'] = gt_bboxes_3d_mask
        return input_dict

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class ObjectNoise(object):

    def __init__(self,
                 loc_noise_std=[0.25, 0.25, 0.25],
                 global_rot_range=[0.0, 0.0],
                 rot_uniform_noise=[-0.15707963267, 0.15707963267],
                 num_try=100):
        self.loc_noise_std = loc_noise_std
        self.global_rot_range = global_rot_range
        self.rot_uniform_noise = rot_uniform_noise
        self.num_try = num_try

    def __call__(self, input_dict):
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        points = input_dict['points']
        gt_bboxes_3d_mask = input_dict['gt_bboxes_3d_mask']
        # TODO: check this inplace function
        noise_per_object_v3_(
            gt_bboxes_3d,
            points,
            gt_bboxes_3d_mask,
            rotation_perturb=self.rot_uniform_noise,
            center_noise_std=self.loc_noise_std,
            global_random_rot_range=self.global_rot_range,
            num_try=self.num_try)
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d.astype('float32')
        input_dict['points'] = points
        return input_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(num_try={},'.format(self.num_try)
        repr_str += ' loc_noise_std={},'.format(self.loc_noise_std)
        repr_str += ' global_rot_range={},'.format(self.global_rot_range)
        repr_str += ' rot_uniform_noise={})'.format(self.rot_uniform_noise)
        return repr_str


@PIPELINES.register_module()
class GlobalRotScale(object):

    def __init__(self,
                 rot_uniform_noise=[-0.78539816, 0.78539816],
                 scaling_uniform_noise=[0.95, 1.05],
                 trans_normal_noise=[0, 0, 0]):
        self.rot_uniform_noise = rot_uniform_noise
        self.scaling_uniform_noise = scaling_uniform_noise
        self.trans_normal_noise = trans_normal_noise

    def _trans_bbox_points(self, gt_boxes, points):
        noise_trans = np.random.normal(0, self.trans_normal_noise[0], 3).T
        points[:, :3] += noise_trans
        gt_boxes[:, :3] += noise_trans
        return gt_boxes, points, noise_trans

    def _rot_bbox_points(self, gt_boxes, points, rotation=np.pi / 4):
        if not isinstance(rotation, list):
            rotation = [-rotation, rotation]
        noise_rotation = np.random.uniform(rotation[0], rotation[1])
        points[:, :3], rot_mat_T = box_np_ops.rotation_points_single_angle(
            points[:, :3], noise_rotation, axis=2)
        gt_boxes[:, :3], _ = box_np_ops.rotation_points_single_angle(
            gt_boxes[:, :3], noise_rotation, axis=2)
        gt_boxes[:, 6] += noise_rotation
        if gt_boxes.shape[1] == 9:
            # rotate velo vector
            rot_cos = np.cos(noise_rotation)
            rot_sin = np.sin(noise_rotation)
            rot_mat_T_bev = np.array([[rot_cos, -rot_sin], [rot_sin, rot_cos]],
                                     dtype=points.dtype)
            gt_boxes[:, 7:9] = gt_boxes[:, 7:9] @ rot_mat_T_bev
        return gt_boxes, points, rot_mat_T

    def _scale_bbox_points(self,
                           gt_boxes,
                           points,
                           min_scale=0.95,
                           max_scale=1.05):
        noise_scale = np.random.uniform(min_scale, max_scale)
        points[:, :3] *= noise_scale
        gt_boxes[:, :6] *= noise_scale
        if gt_boxes.shape[1] == 9:
            gt_boxes[:, 7:] *= noise_scale
        return gt_boxes, points, noise_scale

    def __call__(self, input_dict):
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        points = input_dict['points']

        gt_bboxes_3d, points, rotation_factor = self._rot_bbox_points(
            gt_bboxes_3d, points, rotation=self.rot_uniform_noise)
        gt_bboxes_3d, points, scale_factor = self._scale_bbox_points(
            gt_bboxes_3d, points, *self.scaling_uniform_noise)
        gt_bboxes_3d, points, trans_factor = self._trans_bbox_points(
            gt_bboxes_3d, points)

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d.astype('float32')
        input_dict['points'] = points
        input_dict['pcd_scale_factor'] = scale_factor
        input_dict['pcd_rotation'] = rotation_factor
        input_dict['pcd_trans'] = trans_factor
        return input_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(rot_uniform_noise={},'.format(self.rot_uniform_noise)
        repr_str += ' scaling_uniform_noise={},'.format(
            self.scaling_uniform_noise)
        repr_str += ' trans_normal_noise={})'.format(self.trans_normal_noise)
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
    def limit_period(val, offset=0.5, period=np.pi):
        return val - np.floor(val / period + offset) * period

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
        gt_names_3d = input_dict['gt_names_3d']
        gt_bboxes_3d_mask = input_dict['gt_bboxes_3d_mask']
        mask = self.filter_gt_box_outside_range(gt_bboxes_3d, self.bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        gt_names_3d = gt_names_3d[mask]
        # the mask should also be updated
        gt_bboxes_3d_mask = gt_bboxes_3d_mask[mask]

        # limit rad to [-pi, pi]
        gt_bboxes_3d[:, 6] = self.limit_period(
            gt_bboxes_3d[:, 6], offset=0.5, period=2 * np.pi)
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d.astype('float32')
        input_dict['gt_names_3d'] = gt_names_3d
        input_dict['gt_bboxes_3d_mask'] = gt_bboxes_3d_mask
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
