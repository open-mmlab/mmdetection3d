# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence, Union

import mmengine
import numpy as np
import torch
from mmcv import BaseTransform
from mmengine.structures import InstanceData
from numpy import dtype

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import BaseInstance3DBoxes, PointData
from mmdet3d.structures.points import BasePoints
# from .det3d_data_sample import Det3DDataSample
from .nerf_det3d_data_sample import NeRFDet3DDataSample


def to_tensor(
    data: Union[torch.Tensor, np.ndarray, Sequence, int,
                float]) -> torch.Tensor:
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.

    Returns:
        torch.Tensor: the converted data.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        if data.dtype is dtype('float64'):
            data = data.astype(np.float32)
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmengine.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@TRANSFORMS.register_module()
class PackNeRFDetInputs(BaseTransform):
    INPUTS_KEYS = ['points', 'img']
    NERF_INPUT_KEYS = [
        'img', 'denorm_images', 'depth', 'lightpos', 'nerf_sizes', 'raydirs'
    ]

    INSTANCEDATA_3D_KEYS = [
        'gt_bboxes_3d', 'gt_labels_3d', 'attr_labels', 'depths', 'centers_2d'
    ]
    INSTANCEDATA_2D_KEYS = [
        'gt_bboxes',
        'gt_bboxes_labels',
    ]
    NERF_3D_KEYS = ['gt_images', 'gt_depths']

    SEG_KEYS = [
        'gt_seg_map', 'pts_instance_mask', 'pts_semantic_mask',
        'gt_semantic_seg'
    ]

    def __init__(
        self,
        keys: tuple,
        meta_keys: tuple = ('img_path', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'num_pts_feats', 'pcd_trans',
                            'sample_idx', 'pcd_scale_factor', 'pcd_rotation',
                            'pcd_rotation_angle', 'lidar_path',
                            'transformation_3d_flow', 'trans_mat',
                            'affine_aug', 'sweep_img_metas', 'ori_cam2img',
                            'cam2global', 'crop_offset', 'img_crop_offset',
                            'resize_img_shape', 'lidar2cam', 'ori_lidar2img',
                            'num_ref_frames', 'num_views', 'ego2global',
                            'axis_align_matrix')
    ) -> None:
        self.keys = keys
        self.meta_keys = meta_keys

    def _remove_prefix(self, key: str) -> str:
        if key.startswith('gt_'):
            key = key[3:]
        return key

    def transform(self, results: Union[dict,
                                       List[dict]]) -> Union[dict, List[dict]]:
        """Method to pack the input data. when the value in this dict is a
        list, it usually is in Augmentations Testing.

        Args:
            results (dict | list[dict]): Result dict from the data pipeline.

        Returns:
            dict | List[dict]:

            - 'inputs' (dict): The forward data of models. It usually contains
              following keys:

                - points
                - img

            - 'data_samples' (:obj:`NeRFDet3DDataSample`): The annotation info
              of the sample.
        """
        # augtest
        if isinstance(results, list):
            if len(results) == 1:
                # simple test
                return self.pack_single_results(results[0])
            pack_results = []
            for single_result in results:
                pack_results.append(self.pack_single_results(single_result))
            return pack_results
        # norm training and simple testing
        elif isinstance(results, dict):
            return self.pack_single_results(results)
        else:
            raise NotImplementedError

    def pack_single_results(self, results: dict) -> dict:
        """Method to pack the single input data. when the value in this dict is
        a list, it usually is in Augmentations Testing.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: A dict contains

            - 'inputs' (dict): The forward data of models. It usually contains
              following keys:

                - points
                - img

            - 'data_samples' (:obj:`NeRFDet3DDataSample`): The annotation info
              of the sample.
        """
        # Format 3D data
        if 'points' in results:
            if isinstance(results['points'], BasePoints):
                results['points'] = results['points'].tensor

        if 'img' in results:
            if isinstance(results['img'], list):
                # process multiple imgs in single frame
                imgs = np.stack(results['img'], axis=0)
                if imgs.flags.c_contiguous:
                    imgs = to_tensor(imgs).permute(0, 3, 1, 2).contiguous()
                else:
                    imgs = to_tensor(
                        np.ascontiguousarray(imgs.transpose(0, 3, 1, 2)))
                results['img'] = imgs
            else:
                img = results['img']
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                # To improve the computational speed by by 3-5 times, apply:
                # `torch.permute()` rather than `np.transpose()`.
                # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
                # for more details
                if img.flags.c_contiguous:
                    img = to_tensor(img).permute(2, 0, 1).contiguous()
                else:
                    img = to_tensor(
                        np.ascontiguousarray(img.transpose(2, 0, 1)))
                results['img'] = img

        if 'depth' in results:
            if isinstance(results['depth'], list):
                # process multiple depth imgs in single frame
                depth_imgs = np.stack(results['depth'], axis=0)
                if depth_imgs.flags.c_contiguous:
                    depth_imgs = to_tensor(depth_imgs).contiguous()
                else:
                    depth_imgs = to_tensor(np.ascontiguousarray(depth_imgs))
                results['depth'] = depth_imgs
            else:
                depth_img = results['depth']
                if len(depth_img.shape) < 3:
                    depth_img = np.expand_dims(depth_img, -1)
                if depth_img.flags.c_contiguous:
                    depth_img = to_tensor(depth_img).contiguous()
                else:
                    depth_img = to_tensor(np.ascontiguousarray(depth_img))
                results['depth'] = depth_img

        if 'ray_info' in results:
            if isinstance(results['raydirs'], list):
                raydirs = np.stack(results['raydirs'], axis=0)
                if raydirs.flags.c_contiguous:
                    raydirs = to_tensor(raydirs).contiguous()
                else:
                    raydirs = to_tensor(np.ascontiguousarray(raydirs))
                results['raydirs'] = raydirs

            if isinstance(results['lightpos'], list):
                lightposes = np.stack(results['lightpos'], axis=0)
                if lightposes.flags.c_contiguous:
                    lightposes = to_tensor(lightposes).contiguous()
                else:
                    lightposes = to_tensor(np.ascontiguousarray(lightposes))
                lightposes = lightposes.unsqueeze(1).repeat(
                    1, raydirs.shape[1], 1)
                results['lightpos'] = lightposes

            if isinstance(results['gt_images'], list):
                gt_images = np.stack(results['gt_images'], axis=0)
                if gt_images.flags.c_contiguous:
                    gt_images = to_tensor(gt_images).contiguous()
                else:
                    gt_images = to_tensor(np.ascontiguousarray(gt_images))
                results['gt_images'] = gt_images

            if isinstance(results['gt_depths'],
                          list) and len(results['gt_depths']) != 0:
                gt_depths = np.stack(results['gt_depths'], axis=0)
                if gt_depths.flags.c_contiguous:
                    gt_depths = to_tensor(gt_depths).contiguous()
                else:
                    gt_depths = to_tensor(np.ascontiguousarray(gt_depths))
                results['gt_depths'] = gt_depths

            if isinstance(results['denorm_images'], list):
                denorm_imgs = np.stack(results['denorm_images'], axis=0)
                if denorm_imgs.flags.c_contiguous:
                    denorm_imgs = to_tensor(denorm_imgs).permute(
                        0, 3, 1, 2).contiguous()
                else:
                    denorm_imgs = to_tensor(
                        np.ascontiguousarray(
                            denorm_imgs.transpose(0, 3, 1, 2)))
                results['denorm_images'] = denorm_imgs

        for key in [
                'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                'gt_bboxes_labels', 'attr_labels', 'pts_instance_mask',
                'pts_semantic_mask', 'centers_2d', 'depths', 'gt_labels_3d'
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = [to_tensor(res) for res in results[key]]
            else:
                results[key] = to_tensor(results[key])
        if 'gt_bboxes_3d' in results:
            if not isinstance(results['gt_bboxes_3d'], BaseInstance3DBoxes):
                results['gt_bboxes_3d'] = to_tensor(results['gt_bboxes_3d'])

        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = to_tensor(
                results['gt_semantic_seg'][None])
        if 'gt_seg_map' in results:
            results['gt_seg_map'] = results['gt_seg_map'][None, ...]

        if 'gt_images' in results:
            results['gt_images'] = to_tensor(results['gt_images'])
        if 'gt_depths' in results:
            results['gt_depths'] = to_tensor(results['gt_depths'])

        data_sample = NeRFDet3DDataSample()
        gt_instances_3d = InstanceData()
        gt_instances = InstanceData()
        gt_pts_seg = PointData()
        gt_nerf_images = InstanceData()
        gt_nerf_depths = InstanceData()

        data_metas = {}
        for key in self.meta_keys:
            if key in results:
                data_metas[key] = results[key]
            elif 'images' in results:
                if len(results['images'].keys()) == 1:
                    cam_type = list(results['images'].keys())[0]
                    # single-view image
                    if key in results['images'][cam_type]:
                        data_metas[key] = results['images'][cam_type][key]
                else:
                    # multi-view image
                    img_metas = []
                    cam_types = list(results['images'].keys())
                    for cam_type in cam_types:
                        if key in results['images'][cam_type]:
                            img_metas.append(results['images'][cam_type][key])
                    if len(img_metas) > 0:
                        data_metas[key] = img_metas
            elif 'lidar_points' in results:
                if key in results['lidar_points']:
                    data_metas[key] = results['lidar_points'][key]
        data_sample.set_metainfo(data_metas)

        inputs = {}
        for key in self.keys:
            if key in results:
                # if key in self.INPUTS_KEYS:
                if key in self.NERF_INPUT_KEYS:
                    inputs[key] = results[key]
                elif key in self.NERF_3D_KEYS:
                    if key == 'gt_images':
                        gt_nerf_images[self._remove_prefix(key)] = results[key]
                    else:
                        gt_nerf_depths[self._remove_prefix(key)] = results[key]
                elif key in self.INSTANCEDATA_3D_KEYS:
                    gt_instances_3d[self._remove_prefix(key)] = results[key]
                elif key in self.INSTANCEDATA_2D_KEYS:
                    if key == 'gt_bboxes_labels':
                        gt_instances['labels'] = results[key]
                    else:
                        gt_instances[self._remove_prefix(key)] = results[key]
                elif key in self.SEG_KEYS:
                    gt_pts_seg[self._remove_prefix(key)] = results[key]
                else:
                    raise NotImplementedError(f'Please modified '
                                              f'`Pack3DDetInputs` '
                                              f'to put {key} to '
                                              f'corresponding field')

        data_sample.gt_instances_3d = gt_instances_3d
        data_sample.gt_instances = gt_instances
        data_sample.gt_pts_seg = gt_pts_seg
        data_sample.gt_nerf_images = gt_nerf_images
        data_sample.gt_nerf_depths = gt_nerf_depths
        if 'eval_ann_info' in results:
            data_sample.eval_ann_info = results['eval_ann_info']
        else:
            data_sample.eval_ann_info = None

        packed_results = dict()
        packed_results['data_samples'] = data_sample
        packed_results['inputs'] = inputs

        return packed_results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys})'
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str
