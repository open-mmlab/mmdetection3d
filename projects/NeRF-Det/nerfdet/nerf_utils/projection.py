# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F


class Projector():

    def __init__(self, device='cuda'):
        self.device = device

    def inbound(self, pixel_locations, h, w):
        """check if the pixel locations are in valid range.

        :param pixel_locations: [..., 2]
        :param h: height
        :param w: weight
        :return: mask, bool, [...]
        """
        return (pixel_locations[..., 0] <= w - 1.) & \
               (pixel_locations[..., 0] >= 0) & \
               (pixel_locations[..., 1] <= h - 1.) &\
               (pixel_locations[..., 1] >= 0)

    def normalize(self, pixel_locations, h, w):
        resize_factor = torch.tensor([w - 1., h - 1.
                                      ]).to(pixel_locations.device)[None,
                                                                    None, :]
        normalized_pixel_locations = 2 * pixel_locations / resize_factor - 1.
        # [n_views, n_points, 2]
        return normalized_pixel_locations

    def compute_projections(self, xyz, train_cameras):
        """project 3D points into cameras.

        :param xyz: [..., 3]
        :param train_cameras: [n_views, 34], 34 =
            img_size(2) + intrinsics(16) + extrinsics(16)
        :return: pixel locations [..., 2], mask [...]
        """
        original_shape = xyz.shape[:2]
        xyz = xyz.reshape(-1, 3)
        num_views = len(train_cameras)
        train_intrinsics = train_cameras[:, 2:18].reshape(-1, 4,
                                                          4)  # [n_views, 4, 4]
        train_poses = train_cameras[:, -16:].reshape(-1, 4,
                                                     4)  # [n_views, 4, 4]
        xyz_h = torch.cat([xyz, torch.ones_like(xyz[..., :1])],
                          dim=-1)  # [n_points, 4]
        # projections = train_intrinsics.bmm(torch.inverse(train_poses))
        # we have inverse the pose in dataloader so
        # do not need to inverse here.
        projections = train_intrinsics.bmm(train_poses) \
            .bmm(xyz_h.t()[None, ...].repeat(num_views, 1, 1))
        projections = projections.permute(0, 2, 1)  # [n_views, n_points, 4]
        pixel_locations = projections[..., :2] / torch.clamp(
            projections[..., 2:3], min=1e-8)  # [n_views, n_points, 2]
        pixel_locations = torch.clamp(pixel_locations, min=-1e6, max=1e6)
        mask = projections[...,
                           2] > 0  # a point is invalid if behind the camera
        return pixel_locations.reshape((num_views, ) + original_shape + (2, )), \
               mask.reshape((num_views, ) + original_shape) # noqa

    def compute_angle(self, xyz, query_camera, train_cameras):
        '''
        :param xyz: [..., 3]
        :param query_camera: [34, ]
        :param train_cameras: [n_views, 34]
        :return: [n_views, ..., 4]; The first 3 channels
            are unit-length vector of the difference between
            query and target ray directions, the last channel
            is the inner product of the two directions.
        '''
        original_shape = xyz.shape[:2]
        xyz = xyz.reshape(-1, 3)
        train_poses = train_cameras[:, -16:].reshape(-1, 4,
                                                     4)  # [n_views, 4, 4]
        num_views = len(train_poses)
        query_pose = query_camera[-16:].reshape(-1, 4,
                                                4).repeat(num_views, 1,
                                                          1)  # [n_views, 4, 4]
        ray2tar_pose = (query_pose[:, :3, 3].unsqueeze(1) - xyz.unsqueeze(0))
        ray2tar_pose /= (torch.norm(ray2tar_pose, dim=-1, keepdim=True) + 1e-6)
        ray2train_pose = (
            train_poses[:, :3, 3].unsqueeze(1) - xyz.unsqueeze(0))
        ray2train_pose /= (
            torch.norm(ray2train_pose, dim=-1, keepdim=True) + 1e-6)
        ray_diff = ray2tar_pose - ray2train_pose
        ray_diff_norm = torch.norm(ray_diff, dim=-1, keepdim=True)
        ray_diff_dot = torch.sum(
            ray2tar_pose * ray2train_pose, dim=-1, keepdim=True)
        ray_diff_direction = ray_diff / torch.clamp(ray_diff_norm, min=1e-6)
        ray_diff = torch.cat([ray_diff_direction, ray_diff_dot], dim=-1)
        ray_diff = ray_diff.reshape((num_views, ) + original_shape + (4, ))
        return ray_diff

    def compute(self,
                xyz,
                train_imgs,
                train_cameras,
                featmaps=None,
                grid_sample=True):
        '''
        :param xyz: [n_rays, n_samples, 3]
        :param train_imgs: [1, n_views, h, w, 3]
        :param train_cameras: [1, n_views, 34]
        :param featmaps: [n_views, d, h, w]
        :return: rgb_feat_sampled: [n_rays, n_samples, 3+n_feat],
                 mask: [n_rays, n_samples, 1]
        '''
        assert (train_imgs.shape[0] == 1) \
               and (train_cameras.shape[0] == 1)
        #  and (query_camera.shape[0] == 1),
        # 'only support batch_size=1 for now'

        train_imgs = train_imgs.squeeze(0)  # [n_views, h, w, 3]
        train_cameras = train_cameras.squeeze(0)  # [n_views, 34]
        # query_camera = query_camera.squeeze(0)  # [34, ]

        train_imgs = train_imgs.permute(0, 3, 1, 2)  # [n_views, 3, h, w]

        h, w = train_cameras[0][:2]

        # compute the projection of the query points to each reference image
        pixel_locations, mask_in_front = self.compute_projections(
            xyz, train_cameras)
        normalized_pixel_locations = self.normalize(
            pixel_locations, h, w)  # [n_views, n_rays, n_samples, 2]
        # print(pixel_locations.shape)
        """
        it is still okay to concat RGB !!!
        But I do not use as I want to do nerf_density volume
        """
        # rgb sampling
        rgbs_sampled = F.grid_sample(
            train_imgs, normalized_pixel_locations, align_corners=True)
        rgb_sampled = rgbs_sampled.permute(
            2, 3, 0, 1)  # [n_rays, n_samples, n_views, 3]

        # deep feature sampling
        if featmaps is not None:
            if grid_sample:
                # feat_sampled = F.grid_sample(
                #     featmaps, normalized_pixel_locations,
                #     padding_mode='border', align_corners=True)
                feat_sampled = F.grid_sample(
                    featmaps, normalized_pixel_locations, align_corners=True)
                feat_sampled = feat_sampled.permute(
                    2, 3, 0, 1)  # [n_rays, n_samples, n_views, d]
                rgb_feat_sampled = torch.cat(
                    [rgb_sampled, feat_sampled],
                    dim=-1)  # [n_rays, n_samples, n_views, d+3]
                # rgb_feat_sampled = feat_sampled
            else:
                n_images, n_channels, f_h, f_w = featmaps.shape
                resize_factor = torch.tensor([f_w / w - 1., f_h / h - 1.]).to(
                    pixel_locations.device)[None, None, :]
                sample_location = (pixel_locations *
                                   resize_factor).round().long()
                n_images, n_ray, n_sample, _ = sample_location.shape
                sample_x = sample_location[..., 0].view(
                    n_images, -1)  # n_images, n_ray, n_sample
                sample_y = sample_location[..., 1].view(
                    n_images, -1)  # n_images, n_ray, n_sample
                valid = (sample_x >= 0) & (sample_y >=
                                           0) & (sample_x < f_w) & (
                                               sample_y < f_h)
                valid = valid * mask_in_front.view(n_images, -1)
                feat_sampled = torch.zeros(
                    (n_images, n_channels, sample_x.shape[-1]),
                    device=featmaps.device)
                for i in range(n_images):
                    feat_sampled[i, :,
                                 valid[i]] = featmaps[i, :, sample_y[i,
                                                                     valid[i]],
                                                      sample_y[i, valid[i]]]
                feat_sampled = feat_sampled.view(n_images, n_channels, n_ray,
                                                 n_sample)
                rgb_feat_sampled = feat_sampled.permute(2, 3, 0, 1)

        else:
            rgb_feat_sampled = None
        # mask
        inbound = self.inbound(pixel_locations, h, w)
        mask = (inbound * mask_in_front).float().permute(
            1, 2, 0)[..., None]  # [n_rays, n_samples, n_views, 1]
        return rgb_feat_sampled, mask
