import mmcv
import numpy as np
from mmdet3d.registry import TRANSFORMS
from numpy import random


@TRANSFORMS.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5.

    The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results['img']
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, \
                'PhotoMetricDistortion needs the input image of dtype np.float32,'\
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                       self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                           self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower,
                                              self.saturation_upper)

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                           self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results['img'] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str


# @TRANSFORMS.register_module()
# class CropMultiViewImage(object):
#     """Crop the image
#     Args:
#         size (tuple, optional): Fixed padding size.
#     """

#     def __init__(self, size=None):
#         self.size = size

#     def __call__(self, results):
#         """Call function to pad images, masks, semantic segmentation maps.
#         Args:
#             results (dict): Result dict from loading pipeline.
#         Returns:
#             dict: Updated result dict.
#         """
#         results['img'] = [img[:self.size[0], :self.size[1], ...] for img in results['img']]
#         results['img_shape'] = [img.shape for img in results['img']]
#         results['img_fixed_size'] = self.size
#         return results

#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += f'(size={self.size}, '
#         return repr_str

# @TRANSFORMS.register_module()
# class RandomScaleImageMultiViewImage(object):
#     """Random scale the image
#     Args:
#         scales
#     """

#     def __init__(self, scales=[0.5, 1.0, 1.5]):
#         self.scales = scales

#     def __call__(self, results):
#         """Call function to pad images, masks, semantic segmentation maps.
#         Args:
#             results (dict): Result dict from loading pipeline.
#         Returns:
#             dict: Updated result dict.
#         """
#         np.random.shuffle(self.scales)
#         rand_scale = self.scales[0]
#         img_shape = results['img_shape'][0]
#         y_size = int(img_shape[0] * rand_scale)
#         x_size = int(img_shape[1] * rand_scale)
#         scale_factor = np.eye(4)
#         scale_factor[0, 0] *= rand_scale
#         scale_factor[1, 1] *= rand_scale
#         results['img'] = [mmcv.imresize(img, (x_size, y_size), return_scale=False) for img in results['img']]
#         lidar2img = [scale_factor @ l2i for l2i in results['lidar2img']]
#         results['lidar2img'] = lidar2img
#         results['img_shape'] = [img.shape for img in results['img']]
#         results['gt_bboxes_3d'].tensor[:, :6] *= rand_scale
#         return results

#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += f'(size={self.scales}, '
#         return repr_str

# @TRANSFORMS.register_module()
# class HorizontalRandomFlipMultiViewImage(object):

#     def __init__(self, flip_ratio=0.5):
#         self.flip_ratio = 0.5

#     def __call__(self, results):
#         if np.random.rand() >= self.flip_ratio:
#             return results
#         results = self.flip_bbox(results)
#         results = self.flip_cam_params(results)
#         results = self.flip_img(results)
#         return results

#     def flip_img(self, results, direction='horizontal'):
#         results['img'] = [mmcv.imflip(img, direction) for img in results['img']]
#         return results

#     def flip_cam_params(self, results):
#         flip_factor = np.eye(4)
#         flip_factor[1, 1] = -1
#         lidar2cam = [l2c @ flip_factor for l2c in results['lidar2cam']]
#         w = results['img_shape'][0][1]
#         lidar2img = []
#         for cam_intrinsic, l2c in zip(results['cam_intrinsic'], lidar2cam):
#             cam_intrinsic[0, 2] = w - cam_intrinsic[0, 2]
#             lidar2img.append(cam_intrinsic @ l2c)
#         results['lidar2cam'] = lidar2cam
#         results['lidar2img'] = lidar2img
#         return results

#     def flip_bbox(self, input_dict, direction='horizontal'):
#         assert direction in ['horizontal', 'vertical']
#         if len(input_dict['bbox3d_fields']) == 0:  # test mode
#             input_dict['bbox3d_fields'].append('empty_box3d')
#             input_dict['empty_box3d'] = input_dict['box_type_3d'](
#                 np.array([], dtype=np.float32))
#         assert len(input_dict['bbox3d_fields']) == 1
#         for key in input_dict['bbox3d_fields']:
#             if 'points' in input_dict:
#                 input_dict['points'] = input_dict[key].flip(
#                     direction, points=input_dict['points'])
#             else:
#                 input_dict[key].flip(direction)
#         return input_dict
