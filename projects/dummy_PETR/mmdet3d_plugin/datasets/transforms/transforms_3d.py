# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from PIL import Image

import mmdet3d
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes, limit_period


@TRANSFORMS.register_module()
class AddPETR(BaseTransform):

    def __init__(self, data_aug_conf=None, training=True):
        self.data_aug_conf = data_aug_conf
        self.training = training

    def transform(self, input_dict):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            input_dict (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        image_paths = []
        lidar2img_rts = []
        intrinsics = []
        extrinsics = []
        img_timestamp = []
        for cam_type, cam_info in input_dict['images'].items():
            img_timestamp.append(cam_info['timestamp'] / 1e6)
            image_paths.append(cam_info['img_path'])
            # obtain lidar to image transformation matrix
            lidar2cam_rt = np.array(cam_info['lidar2cam']).T
            intrinsic = np.array(cam_info['cam2img'])
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            # The extrinsics mean the transformation from lidar to camera.
            # If anyone want to use the extrinsics as sensor to lidar,
            # please use np.linalg.inv(lidar2cam_rt.T)
            # and modify the ResizeCropFlipImage
            # and LoadMultiViewImageFromMultiSweepsFiles.
            lidar2img_rts.append(lidar2img_rt)  # Different!!!
            intrinsics.append(viewpad)  # Exactly same!!!
            extrinsics.append(lidar2cam_rt)  # Different!!!

        input_dict.update(
            dict(
                img_timestamp=img_timestamp,
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                intrinsics=intrinsics,
                extrinsics=extrinsics))
        return input_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(dir={self.data_aug_conf}, '
        return repr_str


@TRANSFORMS.register_module()
class ResizeCropFlipImage(BaseTransform):
    """Random resize, Crop and flip the image
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(self, data_aug_conf=None, training=True):
        self.data_aug_conf = data_aug_conf
        self.training = training

    def transform(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """

        imgs = results['img']
        N = len(imgs)
        new_imgs = []
        resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
        # print(results.keys())
        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))
            # augmentation (resize, crop, horizontal flip, rotate)
            # different view use different aug (BEV Det)
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            new_imgs.append(np.array(img).astype(np.float32))
            results['intrinsics'][
                i][:3, :3] = ida_mat @ results['intrinsics'][i][:3, :3]

        results['img'] = new_imgs
        results['lidar2img'] = [
            results['intrinsics'][i] @ results['extrinsics'][i].T
            for i in range(len(results['extrinsics']))
        ]

        return results

    def _get_rot(self, h):

        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat

    def _sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.training:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim'])) *
                newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.mean(self.data_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate


@TRANSFORMS.register_module()
class GlobalRotScaleTransImage(BaseTransform):
    """Random resize, Crop and flip the image
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(
        self,
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        reverse_angle=False,
        training=True,
    ):

        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std

        self.reverse_angle = reverse_angle
        self.training = training

    def transform(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        # random rotate
        rot_angle = np.random.uniform(*self.rot_range)

        self.rotate_bev_along_z(results, rot_angle)
        if self.reverse_angle:
            rot_angle *= -1
        results['gt_bboxes_3d'].rotate(np.array(rot_angle))

        # random scale
        scale_ratio = np.random.uniform(*self.scale_ratio_range)
        self.scale_xyz(results, scale_ratio)
        results['gt_bboxes_3d'].scale(scale_ratio)

        # TODO: support translation
        if not self.reverse_angle:
            gt_bboxes_3d = results['gt_bboxes_3d'].tensor.numpy()
            gt_bboxes_3d[:, 6] -= 2 * rot_angle
            results['gt_bboxes_3d'] = LiDARInstance3DBoxes(
                gt_bboxes_3d, box_dim=9)

        return results

    def rotate_bev_along_z(self, results, angle):
        rot_cos = torch.cos(torch.tensor(angle))
        rot_sin = torch.sin(torch.tensor(angle))

        rot_mat = torch.tensor([[rot_cos, -rot_sin, 0, 0],
                                [rot_sin, rot_cos, 0, 0], [0, 0, 1, 0],
                                [0, 0, 0, 1]])
        rot_mat_inv = torch.inverse(rot_mat)

        num_view = len(results['lidar2img'])
        for view in range(num_view):
            results['lidar2img'][view] = (torch.tensor(
                results['lidar2img'][view]).float() @ rot_mat_inv).numpy()
            results['extrinsics'][view] = (torch.tensor(
                results['extrinsics'][view]).float() @ rot_mat_inv).numpy()

        return

    def scale_xyz(self, results, scale_ratio):
        rot_mat = torch.tensor([
            [scale_ratio, 0, 0, 0],
            [0, scale_ratio, 0, 0],
            [0, 0, scale_ratio, 0],
            [0, 0, 0, 1],
        ])

        rot_mat_inv = torch.inverse(rot_mat)

        num_view = len(results['lidar2img'])
        for view in range(num_view):
            results['lidar2img'][view] = (torch.tensor(
                results['lidar2img'][view]).float() @ rot_mat_inv).numpy()
            results['extrinsics'][view] = (torch.tensor(
                rot_mat_inv.T @ results['extrinsics'][view]).float()).numpy()
        return


@TRANSFORMS.register_module()
class NormalizeMultiviewImage(BaseTransform):
    """Normalize the image.

    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def transform(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results['img'] = [
            mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            for img in results['img']
        ]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@TRANSFORMS.register_module()
class PadMultiViewImage(BaseTransform):
    """Pad the multi-view image.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = [
                mmcv.impad(img, shape=self.size, pad_val=self.pad_val)
                for img in results['img']
            ]
        elif self.size_divisor is not None:
            padded_img = [
                mmcv.impad_to_multiple(
                    img, self.size_divisor, pad_val=self.pad_val)
                for img in results['img']
            ]
        results['img_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def transform(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'


@TRANSFORMS.register_module()
class LidarBox3dVersionTransfrom(BaseTransform):
    """Transform the LiDARInstance3DBoxes from mmdet3d v1.x to v0.x. Due to the
    Coordinate system refactoring: https://mmdetection3d.readthedocs.io/en/late
    st/compatibility.html#v1-0-0rc0.

    Args:
        dir(bool): transform forward or backward (Fake parameter)
    """

    def __init__(self, dir=1):

        self.dir = dir

    def transform(self, input_dict):
        """Call function to transform the LiDARInstance3DBoxes from mmdet3d
        v1.x to v0.x.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after transformation,
                'gt_bboxes_3d' key is updated in the result dict.
        """
        if int(mmdet3d.__version__[0]) >= 1:
            # Begin hack adaptation to mmdet3d v1.0 ####
            gt_bboxes_3d = input_dict['gt_bboxes_3d'].tensor

            gt_bboxes_3d[:, [3, 4]] = gt_bboxes_3d[:, [4, 3]]
            gt_bboxes_3d[:, 6] = -gt_bboxes_3d[:, 6] - np.pi / 2
            gt_bboxes_3d[:, 6] = limit_period(
                gt_bboxes_3d[:, 6], period=np.pi * 2)

            input_dict['gt_bboxes_3d'] = LiDARInstance3DBoxes(
                gt_bboxes_3d, box_dim=9)
            # End hack adaptation to mmdet3d v1.0 ####
        return input_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(dir={self.dir}, '
        return repr_str
