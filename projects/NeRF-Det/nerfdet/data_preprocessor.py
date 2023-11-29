# Copyright (c) OpenMMLab. All rights reserved.
import math
from numbers import Number
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from mmdet.models import DetDataPreprocessor
from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mmengine.model import stack_batch
from mmengine.utils import is_seq_of
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.models.data_preprocessors.utils import multiview_img_stack_batch
from mmdet3d.models.data_preprocessors.voxelize import (
    VoxelizationByGridShape, dynamic_scatter_3d)
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import OptConfigType


@MODELS.register_module()
class NeRFDetDataPreprocessor(DetDataPreprocessor):
    """In NeRF-Det, some extra information is needed in NeRF branch. We put the
    datapreprocessor operations of these new information such as stack and pack
    operations in this class. You can find the stack operations in subfuction
    'collate_data' and the pack operations in 'simple_process'. Other codes are
    the same as the default class 'DetDataPreprocessor'.

    Points / Image pre-processor for point clouds / vision-only / multi-
    modality 3D detection tasks.

    It provides the data pre-processing as follows

    - Collate and move image and point cloud data to the target device.

    - 1) For image data:

      - Pad images in inputs to the maximum size of current batch with defined
        ``pad_value``. The padding size can be divisible by a defined
        ``pad_size_divisor``.
      - Stack images in inputs to batch_imgs.
      - Convert images in inputs from bgr to rgb if the shape of input is
        (3, H, W).
      - Normalize images in inputs with defined std and mean.
      - Do batch augmentations during training.

    - 2) For point cloud data:

      - If no voxelization, directly return list of point cloud data.
      - If voxelization is applied, voxelize point cloud according to
        ``voxel_type`` and obtain ``voxels``.

    Args:
        voxel (bool): Whether to apply voxelization to point cloud.
            Defaults to False.
        voxel_type (str): Voxelization type. Two voxelization types are
            provided: 'hard' and 'dynamic', respectively for hard voxelization
            and dynamic voxelization. Defaults to 'hard'.
        voxel_layer (dict or :obj:`ConfigDict`, optional): Voxelization layer
            config. Defaults to None.
        batch_first (bool): Whether to put the batch dimension to the first
            dimension when getting voxel coordinates. Defaults to True.
        max_voxels (int, optional): Maximum number of voxels in each voxel
            grid. Defaults to None.
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        pad_size_divisor (int): The size of padded image should be divisible by
            ``pad_size_divisor``. Defaults to 1.
        pad_value (float or int): The padded pixel value. Defaults to 0.
        pad_mask (bool): Whether to pad instance masks. Defaults to False.
        mask_pad_value (int): The padded pixel value for instance masks.
            Defaults to 0.
        pad_seg (bool): Whether to pad semantic segmentation maps.
            Defaults to False.
        seg_pad_value (int): The padded pixel value for semantic segmentation
            maps. Defaults to 255.
        bgr_to_rgb (bool): Whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): Whether to convert image from RGB to BGR.
            Defaults to False.
        boxtype2tensor (bool): Whether to convert the ``BaseBoxes`` type of
            bboxes data to ``Tensor`` type. Defaults to True.
        non_blocking (bool): Whether to block current process when transferring
            data to device. Defaults to False.
        batch_augments (List[dict], optional): Batch-level augmentations.
            Defaults to None.
    """

    def __init__(self,
                 voxel: bool = False,
                 voxel_type: str = 'hard',
                 voxel_layer: OptConfigType = None,
                 batch_first: bool = True,
                 max_voxels: Optional[int] = None,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 pad_mask: bool = False,
                 mask_pad_value: int = 0,
                 pad_seg: bool = False,
                 seg_pad_value: int = 255,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 boxtype2tensor: bool = True,
                 non_blocking: bool = False,
                 batch_augments: Optional[List[dict]] = None) -> None:
        super(NeRFDetDataPreprocessor, self).__init__(
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            pad_mask=pad_mask,
            mask_pad_value=mask_pad_value,
            pad_seg=pad_seg,
            seg_pad_value=seg_pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            boxtype2tensor=boxtype2tensor,
            non_blocking=non_blocking,
            batch_augments=batch_augments)
        self.voxel = voxel
        self.voxel_type = voxel_type
        self.batch_first = batch_first
        self.max_voxels = max_voxels
        if voxel:
            self.voxel_layer = VoxelizationByGridShape(**voxel_layer)

    def forward(self,
                data: Union[dict, List[dict]],
                training: bool = False) -> Union[dict, List[dict]]:
        """Perform normalization, padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict or List[dict]): Data from dataloader. The dict contains
                the whole batch data, when it is a list[dict], the list
                indicates test time augmentation.
            training (bool): Whether to enable training time augmentation.
                Defaults to False.

        Returns:
            dict or List[dict]: Data in the same format as the model input.
        """
        if isinstance(data, list):
            num_augs = len(data)
            aug_batch_data = []
            for aug_id in range(num_augs):
                single_aug_batch_data = self.simple_process(
                    data[aug_id], training)
                aug_batch_data.append(single_aug_batch_data)
            return aug_batch_data

        else:
            return self.simple_process(data, training)

    def simple_process(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding and bgr2rgb conversion for img data
        based on ``BaseDataPreprocessor``, and voxelize point cloud if `voxel`
        is set to be True.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.
                Defaults to False.

        Returns:
            dict: Data in the same format as the model input.
        """
        if 'img' in data['inputs']:
            batch_pad_shape = self._get_pad_shape(data)

        data = self.collate_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']
        batch_inputs = dict()

        if 'points' in inputs:
            batch_inputs['points'] = inputs['points']

            if self.voxel:
                voxel_dict = self.voxelize(inputs['points'], data_samples)
                batch_inputs['voxels'] = voxel_dict

        if 'imgs' in inputs:
            imgs = inputs['imgs']

            if data_samples is not None:
                # NOTE the batched image size information may be useful, e.g.
                # in DETR, this is needed for the construction of masks, which
                # is then used for the transformer_head.
                batch_input_shape = tuple(imgs[0].size()[-2:])
                for data_sample, pad_shape in zip(data_samples,
                                                  batch_pad_shape):
                    data_sample.set_metainfo({
                        'batch_input_shape': batch_input_shape,
                        'pad_shape': pad_shape
                    })

                if self.boxtype2tensor:
                    samplelist_boxtype2tensor(data_samples)
                if self.pad_mask:
                    self.pad_gt_masks(data_samples)
                if self.pad_seg:
                    self.pad_gt_sem_seg(data_samples)

            if training and self.batch_augments is not None:
                for batch_aug in self.batch_augments:
                    imgs, data_samples = batch_aug(imgs, data_samples)
            batch_inputs['imgs'] = imgs
        # Hard code here, will be changed later.
        # if len(inputs['depth']) != 0:
        if 'depth' in inputs.keys():
            batch_inputs['depth'] = inputs['depth']
        batch_inputs['lightpos'] = inputs['lightpos']
        batch_inputs['nerf_sizes'] = inputs['nerf_sizes']
        batch_inputs['denorm_images'] = inputs['denorm_images']
        batch_inputs['raydirs'] = inputs['raydirs']

        return {'inputs': batch_inputs, 'data_samples': data_samples}

    def preprocess_img(self, _batch_img: Tensor) -> Tensor:
        # channel transform
        if self._channel_conversion:
            _batch_img = _batch_img[[2, 1, 0], ...]
        # Convert to float after channel conversion to ensure
        # efficiency
        _batch_img = _batch_img.float()
        # Normalization.
        if self._enable_normalize:
            if self.mean.shape[0] == 3:
                assert _batch_img.dim() == 3 and _batch_img.shape[0] == 3, (
                    'If the mean has 3 values, the input tensor '
                    'should in shape of (3, H, W), but got the '
                    f'tensor with shape {_batch_img.shape}')
            _batch_img = (_batch_img - self.mean) / self.std
        return _batch_img

    def collate_data(self, data: dict) -> dict:
        """Copy data to the target device and perform normalization, padding
        and bgr2rgb conversion and stack based on ``BaseDataPreprocessor``.

        Collates the data sampled from dataloader into a list of dict and list
        of labels, and then copies tensor to the target device.

        Args:
            data (dict): Data sampled from dataloader.

        Returns:
            dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)  # type: ignore

        if 'img' in data['inputs']:
            _batch_imgs = data['inputs']['img']
            # Process data with `pseudo_collate`.
            if is_seq_of(_batch_imgs, torch.Tensor):
                batch_imgs = []
                img_dim = _batch_imgs[0].dim()
                for _batch_img in _batch_imgs:
                    if img_dim == 3:  # standard img
                        _batch_img = self.preprocess_img(_batch_img)
                    elif img_dim == 4:
                        _batch_img = [
                            self.preprocess_img(_img) for _img in _batch_img
                        ]

                        _batch_img = torch.stack(_batch_img, dim=0)

                    batch_imgs.append(_batch_img)

                # Pad and stack Tensor.
                if img_dim == 3:
                    batch_imgs = stack_batch(batch_imgs, self.pad_size_divisor,
                                             self.pad_value)
                elif img_dim == 4:
                    batch_imgs = multiview_img_stack_batch(
                        batch_imgs, self.pad_size_divisor, self.pad_value)

            # Process data with `default_collate`.
            elif isinstance(_batch_imgs, torch.Tensor):
                assert _batch_imgs.dim() == 4, (
                    'The input of `ImgDataPreprocessor` should be a NCHW '
                    'tensor or a list of tensor, but got a tensor with '
                    f'shape: {_batch_imgs.shape}')
                if self._channel_conversion:
                    _batch_imgs = _batch_imgs[:, [2, 1, 0], ...]
                # Convert to float after channel conversion to ensure
                # efficiency
                _batch_imgs = _batch_imgs.float()
                if self._enable_normalize:
                    _batch_imgs = (_batch_imgs - self.mean) / self.std
                h, w = _batch_imgs.shape[2:]
                target_h = math.ceil(
                    h / self.pad_size_divisor) * self.pad_size_divisor
                target_w = math.ceil(
                    w / self.pad_size_divisor) * self.pad_size_divisor
                pad_h = target_h - h
                pad_w = target_w - w
                batch_imgs = F.pad(_batch_imgs, (0, pad_w, 0, pad_h),
                                   'constant', self.pad_value)
            else:
                raise TypeError(
                    'Output of `cast_data` should be a list of dict '
                    'or a tuple with inputs and data_samples, but got '
                    f'{type(data)}: {data}')

            data['inputs']['imgs'] = batch_imgs
        if 'raydirs' in data['inputs']:
            _batch_dirs = data['inputs']['raydirs']
            batch_dirs = stack_batch(_batch_dirs)
            data['inputs']['raydirs'] = batch_dirs

        if 'lightpos' in data['inputs']:
            _batch_poses = data['inputs']['lightpos']
            batch_poses = stack_batch(_batch_poses)
            data['inputs']['lightpos'] = batch_poses

        if 'denorm_images' in data['inputs']:
            _batch_denorm_imgs = data['inputs']['denorm_images']
            # Process data with `pseudo_collate`.
            if is_seq_of(_batch_denorm_imgs, torch.Tensor):
                denorm_img_dim = _batch_denorm_imgs[0].dim()
                # Pad and stack Tensor.
                if denorm_img_dim == 3:
                    batch_denorm_imgs = stack_batch(_batch_denorm_imgs,
                                                    self.pad_size_divisor,
                                                    self.pad_value)
                elif denorm_img_dim == 4:
                    batch_denorm_imgs = multiview_img_stack_batch(
                        _batch_denorm_imgs, self.pad_size_divisor,
                        self.pad_value)
            data['inputs']['denorm_images'] = batch_denorm_imgs

        data.setdefault('data_samples', None)

        return data

    def _get_pad_shape(self, data: dict) -> List[Tuple[int, int]]:
        """Get the pad_shape of each image based on data and
        pad_size_divisor."""
        # rewrite `_get_pad_shape` for obtaining image inputs.
        _batch_inputs = data['inputs']['img']
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_pad_shape = []
            for ori_input in _batch_inputs:
                if ori_input.dim() == 4:
                    # mean multiview input, select one of the
                    # image to calculate the pad shape
                    ori_input = ori_input[0]
                pad_h = int(
                    np.ceil(ori_input.shape[1] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                pad_w = int(
                    np.ceil(ori_input.shape[2] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                batch_pad_shape.append((pad_h, pad_w))
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_inputs.shape}')
            pad_h = int(
                np.ceil(_batch_inputs.shape[1] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            pad_w = int(
                np.ceil(_batch_inputs.shape[2] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            batch_pad_shape = [(pad_h, pad_w)] * _batch_inputs.shape[0]
        else:
            raise TypeError('Output of `cast_data` should be a list of dict '
                            'or a tuple with inputs and data_samples, but got '
                            f'{type(data)}: {data}')
        return batch_pad_shape

    @torch.no_grad()
    def voxelize(self, points: List[Tensor],
                 data_samples: SampleList) -> Dict[str, Tensor]:
        """Apply voxelization to point cloud.

        Args:
            points (List[Tensor]): Point cloud in one data batch.
            data_samples: (list[:obj:`NeRFDet3DDataSample`]): The annotation
                data of every samples. Add voxel-wise annotation for
                segmentation.

        Returns:
            Dict[str, Tensor]: Voxelization information.

            - voxels (Tensor): Features of voxels, shape is MxNxC for hard
              voxelization, NxC for dynamic voxelization.
            - coors (Tensor): Coordinates of voxels, shape is Nx(1+NDim),
              where 1 represents the batch index.
            - num_points (Tensor, optional): Number of points in each voxel.
            - voxel_centers (Tensor, optional): Centers of voxels.
        """

        voxel_dict = dict()

        if self.voxel_type == 'hard':
            voxels, coors, num_points, voxel_centers = [], [], [], []
            for i, res in enumerate(points):
                res_voxels, res_coors, res_num_points = self.voxel_layer(res)
                res_voxel_centers = (
                    res_coors[:, [2, 1, 0]] + 0.5) * res_voxels.new_tensor(
                        self.voxel_layer.voxel_size) + res_voxels.new_tensor(
                            self.voxel_layer.point_cloud_range[0:3])
                res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
                voxels.append(res_voxels)
                coors.append(res_coors)
                num_points.append(res_num_points)
                voxel_centers.append(res_voxel_centers)

            voxels = torch.cat(voxels, dim=0)
            coors = torch.cat(coors, dim=0)
            num_points = torch.cat(num_points, dim=0)
            voxel_centers = torch.cat(voxel_centers, dim=0)

            voxel_dict['num_points'] = num_points
            voxel_dict['voxel_centers'] = voxel_centers
        elif self.voxel_type == 'dynamic':
            coors = []
            # dynamic voxelization only provide a coors mapping
            for i, res in enumerate(points):
                res_coors = self.voxel_layer(res)
                res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
                coors.append(res_coors)
            voxels = torch.cat(points, dim=0)
            coors = torch.cat(coors, dim=0)
        elif self.voxel_type == 'cylindrical':
            voxels, coors = [], []
            for i, (res, data_sample) in enumerate(zip(points, data_samples)):
                rho = torch.sqrt(res[:, 0]**2 + res[:, 1]**2)
                phi = torch.atan2(res[:, 1], res[:, 0])
                polar_res = torch.stack((rho, phi, res[:, 2]), dim=-1)
                min_bound = polar_res.new_tensor(
                    self.voxel_layer.point_cloud_range[:3])
                max_bound = polar_res.new_tensor(
                    self.voxel_layer.point_cloud_range[3:])
                try:  # only support PyTorch >= 1.9.0
                    polar_res_clamp = torch.clamp(polar_res, min_bound,
                                                  max_bound)
                except TypeError:
                    polar_res_clamp = polar_res.clone()
                    for coor_idx in range(3):
                        polar_res_clamp[:, coor_idx][
                            polar_res[:, coor_idx] >
                            max_bound[coor_idx]] = max_bound[coor_idx]
                        polar_res_clamp[:, coor_idx][
                            polar_res[:, coor_idx] <
                            min_bound[coor_idx]] = min_bound[coor_idx]
                res_coors = torch.floor(
                    (polar_res_clamp - min_bound) / polar_res_clamp.new_tensor(
                        self.voxel_layer.voxel_size)).int()
                self.get_voxel_seg(res_coors, data_sample)
                res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
                res_voxels = torch.cat((polar_res, res[:, :2], res[:, 3:]),
                                       dim=-1)
                voxels.append(res_voxels)
                coors.append(res_coors)
            voxels = torch.cat(voxels, dim=0)
            coors = torch.cat(coors, dim=0)
        elif self.voxel_type == 'minkunet':
            voxels, coors = [], []
            voxel_size = points[0].new_tensor(self.voxel_layer.voxel_size)
            for i, (res, data_sample) in enumerate(zip(points, data_samples)):
                res_coors = torch.round(res[:, :3] / voxel_size).int()
                res_coors -= res_coors.min(0)[0]

                res_coors_numpy = res_coors.cpu().numpy()
                inds, point2voxel_map = self.sparse_quantize(
                    res_coors_numpy, return_index=True, return_inverse=True)
                point2voxel_map = torch.from_numpy(point2voxel_map).cuda()
                if self.training and self.max_voxels is not None:
                    if len(inds) > self.max_voxels:
                        inds = np.random.choice(
                            inds, self.max_voxels, replace=False)
                inds = torch.from_numpy(inds).cuda()
                if hasattr(data_sample.gt_pts_seg, 'pts_semantic_mask'):
                    data_sample.gt_pts_seg.voxel_semantic_mask \
                        = data_sample.gt_pts_seg.pts_semantic_mask[inds]
                res_voxel_coors = res_coors[inds]
                res_voxels = res[inds]
                if self.batch_first:
                    res_voxel_coors = F.pad(
                        res_voxel_coors, (1, 0), mode='constant', value=i)
                    data_sample.batch_idx = res_voxel_coors[:, 0]
                else:
                    res_voxel_coors = F.pad(
                        res_voxel_coors, (0, 1), mode='constant', value=i)
                    data_sample.batch_idx = res_voxel_coors[:, -1]
                data_sample.point2voxel_map = point2voxel_map.long()
                voxels.append(res_voxels)
                coors.append(res_voxel_coors)
            voxels = torch.cat(voxels, dim=0)
            coors = torch.cat(coors, dim=0)

        else:
            raise ValueError(f'Invalid voxelization type {self.voxel_type}')

        voxel_dict['voxels'] = voxels
        voxel_dict['coors'] = coors

        return voxel_dict

    def get_voxel_seg(self, res_coors: Tensor,
                      data_sample: SampleList) -> None:
        """Get voxel-wise segmentation label and point2voxel map.

        Args:
            res_coors (Tensor): The voxel coordinates of points, Nx3.
            data_sample: (:obj:`NeRFDet3DDataSample`): The annotation data of
                every samples. Add voxel-wise annotation forsegmentation.
        """

        if self.training:
            pts_semantic_mask = data_sample.gt_pts_seg.pts_semantic_mask
            voxel_semantic_mask, _, point2voxel_map = dynamic_scatter_3d(
                F.one_hot(pts_semantic_mask.long()).float(), res_coors, 'mean',
                True)
            voxel_semantic_mask = torch.argmax(voxel_semantic_mask, dim=-1)
            data_sample.gt_pts_seg.voxel_semantic_mask = voxel_semantic_mask
            data_sample.point2voxel_map = point2voxel_map
        else:
            pseudo_tensor = res_coors.new_ones([res_coors.shape[0], 1]).float()
            _, _, point2voxel_map = dynamic_scatter_3d(pseudo_tensor,
                                                       res_coors, 'mean', True)
            data_sample.point2voxel_map = point2voxel_map

    def ravel_hash(self, x: np.ndarray) -> np.ndarray:
        """Get voxel coordinates hash for np.unique.

        Args:
            x (np.ndarray): The voxel coordinates of points, Nx3.

        Returns:
            np.ndarray: Voxels coordinates hash.
        """
        assert x.ndim == 2, x.shape

        x = x - np.min(x, axis=0)
        x = x.astype(np.uint64, copy=False)
        xmax = np.max(x, axis=0).astype(np.uint64) + 1

        h = np.zeros(x.shape[0], dtype=np.uint64)
        for k in range(x.shape[1] - 1):
            h += x[:, k]
            h *= xmax[k + 1]
        h += x[:, -1]
        return h

    def sparse_quantize(self,
                        coords: np.ndarray,
                        return_index: bool = False,
                        return_inverse: bool = False) -> List[np.ndarray]:
        """Sparse Quantization for voxel coordinates used in Minkunet.

        Args:
            coords (np.ndarray): The voxel coordinates of points, Nx3.
            return_index (bool): Whether to return the indices of the unique
                coords, shape (M,).
            return_inverse (bool): Whether to return the indices of the
                original coords, shape (N,).

        Returns:
            List[np.ndarray]: Return index and inverse map if return_index and
            return_inverse is True.
        """
        _, indices, inverse_indices = np.unique(
            self.ravel_hash(coords), return_index=True, return_inverse=True)
        coords = coords[indices]

        outputs = []
        if return_index:
            outputs += [indices]
        if return_inverse:
            outputs += [inverse_indices]
        return outputs
