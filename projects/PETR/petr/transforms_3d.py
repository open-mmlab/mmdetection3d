# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from PIL import Image

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes


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
        results['lidar2cam'] = np.array(results['lidar2cam'])
        for i in range(N):
            intrinsic = np.array(results['cam2img'][i])
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            results['cam2img'][i] = viewpad
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
            results['cam2img'][
                i][:3, :3] = ida_mat @ results['cam2img'][i][:3, :3]

        results['img'] = new_imgs

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
            gt_bboxes_3d = results['gt_bboxes_3d'].numpy()
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
        num_view = len(results['lidar2cam'])
        for view in range(num_view):
            results['lidar2cam'][view] = (
                torch.tensor(np.array(results['lidar2cam'][view]).T).float()
                @ rot_mat_inv).T.numpy()

        return

    def scale_xyz(self, results, scale_ratio):
        rot_mat = torch.tensor([
            [scale_ratio, 0, 0, 0],
            [0, scale_ratio, 0, 0],
            [0, 0, scale_ratio, 0],
            [0, 0, 0, 1],
        ])

        rot_mat_inv = torch.inverse(rot_mat)

        num_view = len(results['lidar2cam'])
        for view in range(num_view):
            results['lidar2cam'][view] = (torch.tensor(
                rot_mat_inv.T
                @ results['lidar2cam'][view].T).float()).T.numpy()

        return
