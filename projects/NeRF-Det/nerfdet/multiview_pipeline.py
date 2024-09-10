# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
from mmcv.transforms import BaseTransform, Compose
from PIL import Image

from mmdet3d.registry import TRANSFORMS


def get_dtu_raydir(pixelcoords, intrinsic, rot, dir_norm=None):
    # rot is c2w
    # pixelcoords: H x W x 2
    x = (pixelcoords[..., 0] + 0.5 - intrinsic[0, 2]) / intrinsic[0, 0]
    y = (pixelcoords[..., 1] + 0.5 - intrinsic[1, 2]) / intrinsic[1, 1]
    z = np.ones_like(x)
    dirs = np.stack([x, y, z], axis=-1)
    # dirs = np.sum(dirs[...,None,:] * rot[:,:], axis=-1) # h*w*1*3   x   3*3
    dirs = dirs @ rot[:, :].T  #
    if dir_norm:
        dirs = dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-5)

    return dirs


@TRANSFORMS.register_module()
class MultiViewPipeline(BaseTransform):
    """MultiViewPipeline used in nerfdet.

    Required Keys:

    - depth_info
    - img_prefix
    - img_info
    - lidar2img
    - c2w
    - cammrotc2w
    - lightpos
    - ray_info

    Modified Keys:

    - lidar2img

    Added Keys:

    - img
    - denorm_images
    - depth
    - c2w
    - camrotc2w
    - lightpos
    - pixels
    - raydirs
    - gt_images
    - gt_depths
    - nerf_sizes
    - depth_range

    Args:
        transforms (list[dict]): The transform pipeline
            used to process the imgs.
        n_images (int): The number of sampled views.
        mean (array): The mean values used in normalization.
        std (array): The variance values used in normalization.
        margin (int): The margin value. Defaults to 10.
        depth_range (array): The range of the depth.
            Defaults to [0.5, 5.5].
        loading (str): The mode of loading. Defaults to 'random'.
        nerf_target_views (int): The number of novel views.
        sample_freq (int): The frequency of sampling.
    """

    def __init__(self,
                 transforms: dict,
                 n_images: int,
                 mean: tuple = [123.675, 116.28, 103.53],
                 std: tuple = [58.395, 57.12, 57.375],
                 margin: int = 10,
                 depth_range: tuple = [0.5, 5.5],
                 loading: str = 'random',
                 nerf_target_views: int = 0,
                 sample_freq: int = 3):
        self.transforms = Compose(transforms)
        self.depth_transforms = Compose(transforms[1])
        self.n_images = n_images
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.margin = margin
        self.depth_range = depth_range
        self.loading = loading
        self.sample_freq = sample_freq
        self.nerf_target_views = nerf_target_views

    def transform(self, results: dict) -> dict:
        """Nerfdet transform function.

        Args:
            results (dict): Result dict from loading pipeline

        Returns:
            dict: The result dict containing the processed results.
            Updated key and value are described below.

                - img (list): The loaded origin image.
                - denorm_images (list): The denormalized image.
                - depth (list): The origin depth image.
                - c2w (list): The c2w matrixes.
                - camrotc2w (list): The rotation matrixes.
                - lightpos (list): The transform parameters of the camera.
                - pixels (list): Some pixel information.
                - raydirs (list): The ray-directions.
                - gt_images (list): The groundtruth images.
                - gt_depths (list): The groundtruth depth images.
                - nerf_sizes (array): The size of the groundtruth images.
                - depth_range (array): The range of the depth.

        Here we give a detailed explanation of some keys mentioned above.
        Let P_c be the coordinate of camera, P_w be the coordinate of world.
        There is such a conversion relationship: P_c = R @ P_w + T.
        The 'camrotc2w' mentioned above corresponds to the R matrix here.
        The 'lightpos' corresponds to the T matrix here. And if you put
        R and T together, you can get the camera extrinsics matrix. It
        corresponds to the 'c2w' mentioned above.
        """
        imgs = []
        depths = []
        extrinsics = []
        c2ws = []
        camrotc2ws = []
        lightposes = []
        pixels = []
        raydirs = []
        gt_images = []
        gt_depths = []
        denorm_imgs_list = []
        nerf_sizes = []

        if self.loading == 'random':
            ids = np.arange(len(results['img_info']))
            replace = True if self.n_images > len(ids) else False
            ids = np.random.choice(ids, self.n_images, replace=replace)
            if self.nerf_target_views != 0:
                target_id = np.random.choice(
                    ids, self.nerf_target_views, replace=False)
                ids = np.setdiff1d(ids, target_id)
                ids = ids.tolist()
                target_id = target_id.tolist()

        else:
            ids = np.arange(len(results['img_info']))
            begin_id = 0
            ids = np.arange(begin_id,
                            begin_id + self.n_images * self.sample_freq,
                            self.sample_freq)
            if self.nerf_target_views != 0:
                target_id = ids

        ratio = 0
        size = (240, 320)
        for i in ids:
            _results = dict()
            _results['img_path'] = results['img_info'][i]['filename']
            _results = self.transforms(_results)
            imgs.append(_results['img'])
            # normalize
            for key in _results.get('img_fields', ['img']):
                _results[key] = mmcv.imnormalize(_results[key], self.mean,
                                                 self.std, True)
            _results['img_norm_cfg'] = dict(
                mean=self.mean, std=self.std, to_rgb=True)
            # pad
            for key in _results.get('img_fields', ['img']):
                padded_img = mmcv.impad(_results[key], shape=size, pad_val=0)
                _results[key] = padded_img
            _results['pad_shape'] = padded_img.shape
            _results['pad_fixed_size'] = size
            ori_shape = _results['ori_shape']
            aft_shape = _results['img_shape']
            ratio = ori_shape[0] / aft_shape[0]
            # prepare the depth information
            if 'depth_info' in results.keys():
                if '.npy' in results['depth_info'][i]['filename']:
                    _results['depth'] = np.load(
                        results['depth_info'][i]['filename'])
                else:
                    _results['depth'] = np.asarray((Image.open(
                        results['depth_info'][i]['filename']))) / 1000
                    _results['depth'] = mmcv.imresize(
                        _results['depth'], (aft_shape[1], aft_shape[0]))
                depths.append(_results['depth'])

            denorm_img = mmcv.imdenormalize(
                _results['img'], self.mean, self.std, to_bgr=True).astype(
                    np.uint8) / 255.0
            denorm_imgs_list.append(denorm_img)
            height, width = padded_img.shape[:2]
            extrinsics.append(results['lidar2img']['extrinsic'][i])

        # prepare the nerf information
        if 'ray_info' in results.keys():
            intrinsics_nerf = results['lidar2img']['intrinsic'].copy()
            intrinsics_nerf[:2] = intrinsics_nerf[:2] / ratio
            assert self.nerf_target_views > 0
            for i in target_id:
                c2ws.append(results['c2w'][i])
                camrotc2ws.append(results['camrotc2w'][i])
                lightposes.append(results['lightpos'][i])
                px, py = np.meshgrid(
                    np.arange(self.margin,
                              width - self.margin).astype(np.float32),
                    np.arange(self.margin,
                              height - self.margin).astype(np.float32))
                pixelcoords = np.stack((px, py),
                                       axis=-1).astype(np.float32)  # H x W x 2
                pixels.append(pixelcoords)
                raydir = get_dtu_raydir(pixelcoords, intrinsics_nerf,
                                        results['camrotc2w'][i])
                raydirs.append(np.reshape(raydir.astype(np.float32), (-1, 3)))
                # read target images
                temp_results = dict()
                temp_results['img_path'] = results['img_info'][i]['filename']

                temp_results_ = self.transforms(temp_results)
                # normalize
                for key in temp_results.get('img_fields', ['img']):
                    temp_results[key] = mmcv.imnormalize(
                        temp_results[key], self.mean, self.std, True)
                temp_results['img_norm_cfg'] = dict(
                    mean=self.mean, std=self.std, to_rgb=True)
                # pad
                for key in temp_results.get('img_fields', ['img']):
                    padded_img = mmcv.impad(
                        temp_results[key], shape=size, pad_val=0)
                    temp_results[key] = padded_img
                temp_results['pad_shape'] = padded_img.shape
                temp_results['pad_fixed_size'] = size
                # denormalize target_images.
                denorm_imgs = mmcv.imdenormalize(
                    temp_results_['img'], self.mean, self.std,
                    to_bgr=True).astype(np.uint8)
                gt_rgb_shape = denorm_imgs.shape

                gt_image = denorm_imgs[py.astype(np.int32),
                                       px.astype(np.int32), :]
                nerf_sizes.append(np.array(gt_image.shape))
                gt_image = np.reshape(gt_image, (-1, 3))
                gt_images.append(gt_image / 255.0)
                if 'depth_info' in results.keys():
                    if '.npy' in results['depth_info'][i]['filename']:
                        _results['depth'] = np.load(
                            results['depth_info'][i]['filename'])
                    else:
                        depth_image = Image.open(
                            results['depth_info'][i]['filename'])
                        _results['depth'] = np.asarray(depth_image) / 1000
                        _results['depth'] = mmcv.imresize(
                            _results['depth'],
                            (gt_rgb_shape[1], gt_rgb_shape[0]))

                    _results['depth'] = _results['depth']
                    gt_depth = _results['depth'][py.astype(np.int32),
                                                 px.astype(np.int32)]
                    gt_depths.append(gt_depth)

        for key in _results.keys():
            if key not in ['img', 'img_info']:
                results[key] = _results[key]
        results['img'] = imgs

        if 'ray_info' in results.keys():
            results['c2w'] = c2ws
            results['camrotc2w'] = camrotc2ws
            results['lightpos'] = lightposes
            results['pixels'] = pixels
            results['raydirs'] = raydirs
            results['gt_images'] = gt_images
            results['gt_depths'] = gt_depths
            results['nerf_sizes'] = nerf_sizes
            results['denorm_images'] = denorm_imgs_list
            results['depth_range'] = np.array([self.depth_range])

        if len(depths) != 0:
            results['depth'] = depths
        results['lidar2img']['extrinsic'] = extrinsics
        return results


@TRANSFORMS.register_module()
class RandomShiftOrigin(BaseTransform):

    def __init__(self, std):
        self.std = std

    def transform(self, results):
        shift = np.random.normal(.0, self.std, 3)
        results['lidar2img']['origin'] += shift
        return results
