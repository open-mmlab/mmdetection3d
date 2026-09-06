"""Camera loading + resize for FocalFormer3D-LC on mmdet3d 1.4.

Why this file exists
--------------------
FocalFormer3D-LC needs two things the plugin cannot currently provide:

  1. A loader that reads the mmdet3d 1.4 ``results['images']`` dict and emits
     ``lidar2img`` / ``cam2img`` / ``lidar2cam`` / ``cam2lidar``. Core's
     ``LoadMultiViewImageFromFiles`` still expects the pre-1.0
     ``results['img_filename']`` key, which mmdet3d 1.x datasets no longer
     populate.
  2. A resize/crop that records ``img_aug_matrix``, which LiftSplatShoot reads
     to undo the resize when it projects camera features into BEV.

``projects/BEVFusion/bevfusion`` has exactly these two, but they cannot be
imported: its ``__init__`` pulls in all 18 of its modules, and several of those
names are already registered by this plugin, among them HungarianAssigner3D,
HeuristicAssigner3D, TransFusionBBoxCoder, BBoxBEVL1Cost and IoU3DCost.
mmengine raises on duplicate registration, so importing BEVFusion alongside
this plugin fails outright.

Hence FF-prefixed copies of the 1.4-native BEVFusion implementations, in one
file, registered under names nothing else uses. Nothing existing is modified.

Adapted from projects/BEVFusion/bevfusion/{loading,transforms_3d}.py
(OpenMMLab, Apache-2.0), in this same repository.

One deliberate difference from the original loader: ``img_shape`` stays the
2-tuple (H, W) that ``FFImageAug3D`` reads via ``ori_shape``. These two classes
are a matched pair, so do not change one without the other.

Colour: images are decoded with ``channel_order='rgb'``, so they are ALREADY
RGB by the time the preprocessor sees them. Det3DDataPreprocessor must
therefore run with ``bgr_to_rgb=False``. The original FocalFormer
``img_norm_cfg`` says ``to_rgb=True``, which describes the mmdet 2.x pipeline
converting BGR->RGB; it does NOT mean a second swap belongs here. Swapping
again feeds BGR to a model trained on RGB, which costs accuracy silently.
"""
import copy
from typing import Any, Dict, Optional

import mmcv
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmengine.fileio import get
from PIL import Image

from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class FFLoadMultiViewImage(LoadMultiViewImageFromFiles):
    """Load the six nuScenes views and build the projection matrices.

    Same contract as BEVLoadMultiViewImageFromFiles. The multi-sweep
    (``num_ref_frames``) branch of the original is dropped: FocalFormer-LC
    consumes a single frame of images, that branch is dead code here, and it
    reads ``results['img_filename']`` which this pipeline never sets.
    """

    def transform(self, results: dict) -> Optional[dict]:
        assert self.num_ref_frames <= 0, (
            'FFLoadMultiViewImage does not implement multi-sweep image '
            'loading; use BEVLoadMultiViewImageFromFiles if that is needed.')

        filename, cam2img, lidar2cam, cam2lidar, lidar2img = [], [], [], [], []
        for _, cam_item in results['images'].items():
            filename.append(cam_item['img_path'])
            lidar2cam.append(cam_item['lidar2cam'])

            lidar2cam_array = np.array(cam_item['lidar2cam']).astype(
                np.float32)
            lidar2cam_rot = lidar2cam_array[:3, :3]
            lidar2cam_trans = lidar2cam_array[:3, 3:4]
            camera2lidar = np.eye(4)
            camera2lidar[:3, :3] = lidar2cam_rot.T
            camera2lidar[:3, 3:4] = -1 * np.matmul(
                lidar2cam_rot.T, lidar2cam_trans.reshape(3, 1))
            cam2lidar.append(camera2lidar)

            cam2img_array = np.eye(4).astype(np.float32)
            cam2img_array[:3, :3] = np.array(cam_item['cam2img']).astype(
                np.float32)
            cam2img.append(cam2img_array)
            # FocalEncoder inverts this to get the per-camera rotation and
            # translation it hands to LiftSplatShoot, so this value affects
            # where every camera feature is splatted.
            lidar2img.append(cam2img_array @ lidar2cam_array)

        results['img_path'] = filename
        results['cam2img'] = np.stack(cam2img, axis=0)
        results['lidar2cam'] = np.stack(lidar2cam, axis=0)
        results['cam2lidar'] = np.stack(cam2lidar, axis=0)
        results['lidar2img'] = np.stack(lidar2img, axis=0)
        results['ori_cam2img'] = copy.deepcopy(results['cam2img'])

        img_bytes = [
            get(name, backend_args=self.backend_args) for name in filename
        ]
        imgs = [
            mmcv.imfrombytes(
                img_byte,
                flag=self.color_type,
                backend='pillow',
                channel_order='rgb') for img_byte in img_bytes
        ]

        # nuScenes views are all 1600x900, but tolerate ragged shapes the same
        # way the original does rather than assuming it.
        img_shapes = np.stack([img.shape for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        assert img_shape_min[-1] == img_shape_max[-1]
        if not np.all(img_shape_max == img_shape_min):
            imgs = [
                mmcv.impad(img, shape=img_shape_max[:2], pad_val=0)
                for img in imgs
            ]
        img = np.stack(imgs, axis=-1)  # (H, W, C, N)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        results['pad_shape'] = img.shape[:2]
        if self.set_default_scale:
            results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        # Identity norm: Det3DDataPreprocessor does the real normalisation.
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['num_views'] = self.num_views
        results['num_ref_frames'] = self.num_ref_frames
        return results


@TRANSFORMS.register_module()
class FFImageAug3D(BaseTransform):
    """Resize/crop the six views and record the homography as img_aug_matrix.

    For FocalFormer-LC at test time: resize_lim [0.5, 0.5] takes 1600x900 to
    800x450, then bot_pct_lim [0, 0] crops 2 rows off the top to reach the
    448x800 the checkpoint's LSS frustum expects (41, 112, 200, 3 -> 112*4=448,
    200*4=800). Aspect ratio is preserved, which a direct resize would not do.
    """

    def __init__(self, final_dim, resize_lim, bot_pct_lim, rot_lim, rand_flip,
                 is_train):
        self.final_dim = final_dim
        self.resize_lim = resize_lim
        self.bot_pct_lim = bot_pct_lim
        self.rand_flip = rand_flip
        self.rot_lim = rot_lim
        self.is_train = is_train

    def sample_augmentation(self, results):
        H, W = results['ori_shape']
        fH, fW = self.final_dim
        if self.is_train:
            resize = np.random.uniform(*self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.random.uniform(*self.bot_pct_lim)) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.rand_flip and bool(np.random.choice([0, 1]))
            rotate = np.random.uniform(*self.rot_lim)
        else:
            resize = np.mean(self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.bot_pct_lim)) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform(self, img, rotation, translation, resize, resize_dims,
                      crop, flip, rotate):
        img = Image.fromarray(img.astype('uint8'), mode='RGB')
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # The same operations as a homography, so LSS can invert them.
        rotation *= resize
        translation -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
        theta = rotate / 180 * np.pi
        A = torch.Tensor([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)],
        ])
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        rotation = A.matmul(rotation)
        translation = A.matmul(translation) + b
        return img, rotation, translation

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        new_imgs, transforms = [], []
        for img in data['img']:
            resize, resize_dims, crop, flip, rotate = \
                self.sample_augmentation(data)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            new_img, rotation, translation = self.img_transform(
                img,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate)
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            new_imgs.append(np.array(new_img).astype(np.float32))
            transforms.append(transform.numpy())
        data['img'] = new_imgs
        # A STACKED (N, 4, 4) tensor, not the list BEVFusion's ImageAug3D
        # leaves behind. This plugin's LiftSplatShoot.get_geometry does
        #     i['img_aug_matrix'][..., :3, :3]
        # on each sample's entry, which needs an array. A list raises
        # "TypeError: list indices must be integers or slices, not tuple".
        data['img_aug_matrix'] = torch.as_tensor(np.stack(transforms))
        return data
