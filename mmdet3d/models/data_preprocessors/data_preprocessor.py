# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from mmcv.ops import Voxelization
from mmengine.data import BaseDataElement
from mmengine.model import stack_batch
from torch.nn import functional as F

from mmdet3d.registry import MODELS
from mmdet3d.utils import OptConfigType
from mmdet.models import DetDataPreprocessor


@MODELS.register_module()
class Det3DDataPreprocessor(DetDataPreprocessor):
    """Points (Image) pre-processor for point clouds / multi-modality 3D
    detection tasks.

    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad images in inputs to the maximum size of current batch with defined
      ``pad_value``. The padding size can be divisible by a defined
      ``pad_size_divisor``
    - Stack images in inputs to batch_imgs.
    - Convert images in inputs from bgr to rgb if the shape of input is
        (3, H, W).
    - Normalize images in inputs with defined std and mean.

    Args:
        voxel (bool): Whether to apply voxelziation to points.
        voxel_type (str): Voxelization type.
        voxel_layer (OptConfigType): Voxelization layer config.
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (Number): The padded pixel value. Defaults to 0.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
    """

    def __init__(self,
                 voxel: bool = False,
                 voxel_type: str = 'hard',
                 voxel_layer: OptConfigType = None,
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
                 batch_augments: Optional[List[dict]] = None):
        super().__init__(
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
            batch_augments=batch_augments)
        self.voxel = voxel
        self.voxel_type = voxel_type
        if voxel:
            self.voxel_layer = Voxelization(**voxel_layer)

    def forward(self,
                data: List[Union[dict, List[dict]]],
                training: bool = False
                ) -> Tuple[Union[dict, List[dict]], Optional[list]]:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (List[dict] | List[List[dict]]): data from dataloader.
                The outer list always represent the batch size, when it is
                a list[list[dict]], the inter list indicate test time
                augmentation.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Tuple[Dict, Optional[list]] |
            Tuple[List[Dict], Optional[list[list]]]:
            Data in the same format as the model input.
        """
        if isinstance(data[0], list):
            num_augs = len(data[0])
            aug_batch_data = []
            aug_batch_data_sample = []
            for aug_id in range(num_augs):
                single_aug_batch_data, \
                    single_aug_batch_data_sample = self.simple_process(
                        [item[aug_id] for item in data], training)
                aug_batch_data.append(single_aug_batch_data)
                aug_batch_data_sample.append(single_aug_batch_data_sample)

            return aug_batch_data, aug_batch_data_sample

        else:
            return self.simple_process(data, training)

    def simple_process(self, data: Sequence[dict], training: bool = False):
        inputs_dict, batch_data_samples = self.collate_data(data)

        if 'points' in inputs_dict[0].keys():
            points = [input['points'] for input in inputs_dict]
        else:
            points = None

        if 'img' in inputs_dict[0].keys():

            imgs = [input['img'] for input in inputs_dict]

            # channel transform
            if self.channel_conversion:
                imgs = [_img[[2, 1, 0], ...] for _img in imgs]
            # Normalization.
            if self._enable_normalize:
                imgs = [(_img.float() - self.mean) / self.std for _img in imgs]
            # Pad and stack Tensor.
            batch_imgs = stack_batch(imgs, self.pad_size_divisor,
                                     self.pad_value)

            batch_pad_shape = self._get_pad_shape(data)

            if batch_data_samples is not None:
                # NOTE the batched image size information may be useful, e.g.
                batch_input_shape = tuple(batch_imgs[0].size()[-2:])
                for data_samples, pad_shape in zip(batch_data_samples,
                                                   batch_pad_shape):
                    data_samples.set_metainfo({
                        'batch_input_shape': batch_input_shape,
                        'pad_shape': pad_shape
                    })

                if self.pad_mask:
                    self.pad_gt_masks(batch_data_samples)

                if self.pad_seg:
                    self.pad_gt_sem_seg(batch_data_samples)

            if training and self.batch_augments is not None:
                for batch_aug in self.batch_augments:
                    batch_imgs, batch_data_samples = batch_aug(
                        batch_imgs, batch_data_samples)
        else:
            imgs = None

        batch_inputs_dict = {
            'points': points,
            'imgs': batch_imgs if imgs is not None else None
        }

        return batch_inputs_dict, batch_data_samples

    def collate_data(
            self, data: Sequence[dict]) -> Tuple[List[dict], Optional[list]]:
        """Collating and copying data to the target device.

        Collates the data sampled from dataloader into a list of dict and
        list of labels, and then copies tensor to the target device.

        Args:
            data (Sequence[dict]): Data sampled from dataloader.

        Returns:
            Tuple[List[Dict], Optional[list]]: Unstacked list of input
            data dict and list of labels at target device.
        """
        # rewrite `collate_data` since the inputs is a dict instead of
        # image tensor.
        inputs_dict = [{
            k: v.to(self._device)
            for k, v in _data['inputs'].items() if v is not None
        } for _data in data]

        batch_data_samples: List[BaseDataElement] = []
        # Model can get predictions without any data samples.
        for _data in data:
            if 'data_sample' in _data:
                batch_data_samples.append(_data['data_sample'])
        # Move data from CPU to corresponding device.
        batch_data_samples = [
            data_sample.to(self._device) for data_sample in batch_data_samples
        ]

        if not batch_data_samples:
            batch_data_samples = None  # type: ignore

        return inputs_dict, batch_data_samples

    def _get_pad_shape(self, data: Sequence[dict]) -> List[tuple]:
        """Get the pad_shape of each image based on data and
        pad_size_divisor."""
        # rewrite `_get_pad_shape` for obaining image inputs.
        ori_inputs = [_data['inputs']['img'] for _data in data]
        batch_pad_shape = []
        for ori_input in ori_inputs:
            pad_h = int(np.ceil(ori_input.shape[1] /
                                self.pad_size_divisor)) * self.pad_size_divisor
            pad_w = int(np.ceil(ori_input.shape[2] /
                                self.pad_size_divisor)) * self.pad_size_divisor
            batch_pad_shape.append((pad_h, pad_w))
        return batch_pad_shape

    def voxelize(self, points: List[torch.Tensor], voxel_type: str) -> tuple:
        """Apply voxelization to points."""

        if voxel_type == 'hard':
            voxels, coors, num_points = [], [], []
            for res in points:
                res_voxels, res_coors, res_num_points = self.voxel_layer(res)
                voxels.append(res_voxels)
                coors.append(res_coors)
                num_points.append(res_num_points)
            voxels = torch.cat(voxels, dim=0)
            num_points = torch.cat(num_points, dim=0)
            coors_batch = []
            for i, coor in enumerate(coors):
                coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
                coors_batch.append(coor_pad)
            coors_batch = torch.cat(coors_batch, dim=0)

            return voxels, num_points, coors_batch

        elif voxel_type == 'dynamic':
            coors = []
            # dynamic voxelization only provide a coors mapping
            for res in points:
                res_coors = self.voxel_layer(res)
                coors.append(res_coors)
            points = torch.cat(points, dim=0)
            coors_batch = []
            for i, coor in enumerate(coors):
                coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
                coors_batch.append(coor_pad)
            coors_batch = torch.cat(coors_batch, dim=0)

            return points, coors_batch
