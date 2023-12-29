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

from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

rng = np.random.RandomState(234)


# helper functions for nerf ray rendering
def volume_sampling(sample_pts, features, aabb):
    B, C, D, W, H = features.shape
    '''
    Actually here is hard code since per gpu only occupy one scene.
    hard_code B=1.can directly use point xyz instead of aabb size
    '''
    assert B == 1
    aabb = torch.Tensor(aabb).to(sample_pts.device)
    N_rays, N_samples, coords = sample_pts.shape
    sample_pts = sample_pts.view(1, N_rays * N_samples, 1, 1,
                                 3).repeat(B, 1, 1, 1, 1)
    aabbSize = aabb[1] - aabb[0]
    invgridSize = 1.0 / aabbSize * 2
    norm_pts = (sample_pts - aabb[0]) * invgridSize - 1
    sample_features = F.grid_sample(
        features, norm_pts, align_corners=True, padding_mode='border')
    # 1, C, 1, 1, N_rays*N_samples
    masks = ((norm_pts < 1) & (norm_pts > -1)).float().sum(dim=-1)
    masks = (masks.view(N_rays, N_samples) == 3)
    # x,y,z should be all in the volume.

    return sample_features.view(C, N_rays,
                                N_samples).permute(1, 2, 0).contiguous(), masks


def _compute_projection(img_meta):
    # [n_views, 34], 34 = img_size(2) + intrinsics(16) + extrinsics(16)
    # projection = []
    views = len(img_meta['lidar2img']['extrinsic'])
    intrinsic = torch.tensor(img_meta['lidar2img']['intrinsic'][:4, :4])
    ratio = img_meta['ori_shape'][0] / img_meta['img_shape'][0]
    intrinsic[:2] /= ratio
    intrinsic = intrinsic.unsqueeze(0).view(1, 16).repeat(views, 1)

    img_size = torch.Tensor(img_meta['img_shape'][:2]).to(intrinsic.device)
    img_size = img_size.unsqueeze(0).repeat(views, 1)
    # use predicted pitch and roll for SUNRGBDTotal test

    extrinsics = []
    for v in range(views):
        extrinsics.append(
            torch.Tensor(img_meta['lidar2img']['extrinsic'][v]).to(
                intrinsic.device))
    extrinsic = torch.stack(extrinsics).view(views, 16)
    train_cameras = torch.cat([img_size, intrinsic, extrinsic], dim=-1)
    return train_cameras.unsqueeze(0)


def compute_mask_points(feature, mask):

    weight = mask / (torch.sum(mask, dim=2, keepdim=True) + 1e-8)
    mean = torch.sum(feature * weight, dim=2, keepdim=True)
    # TODO: his would be a problem since non-valid point we assign var = 0!!!
    var = torch.sum((feature - mean)**2, dim=2, keepdim=True)
    var = var / (torch.sum(mask, dim=2, keepdim=True) + 1e-8)
    var = torch.exp(-var)

    # mean = torch.mean(feature, dim=2, keepdim=True)
    # var = torch.mean((feature - mean)**2, dim=2, keepdim=True)

    return mean, var


def sample_pdf(bins, weights, N_samples, det=False):
    """

    Args:
        bins (tensor):Tensor of shape [N_rays, M+1], M is the number of bins
        weights (tensor):Tensor of shape [N_rays, M+1], M is the number of bins
        N_samples (int):Number of samples along each ray
        det (bool):If True, will perform deterministic sampling

    Returns:
        samples (tuple): [N_rays, N_samples]
    """

    M = weights.shape[1]
    weights += 1e-5
    # Get pdf
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)  # [N_rays, M]
    cdf = torch.cumsum(pdf, dim=-1)  # [N_rays, M]
    cdf = torch.cat([torch.zeros_like(cdf[:, 0:1]), cdf],
                    dim=-1)  # [N_rays, M+1]

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., N_samples, device=bins.device)
        u = u.unsqueeze(0).repeat(bins.shape[0], 1)  # [N_rays, N_samples]
    else:
        u = torch.rand(bins.shape[0], N_samples, device=bins.device)

    # Invert CDF
    above_inds = torch.zeros_like(u, dtype=torch.long)  # [N_rays, N_samples]
    for i in range(M):
        above_inds += (u >= cdf[:, i:i + 1]).long()

    # random sample inside each bin
    below_inds = torch.clamp(above_inds - 1, min=0)
    inds_g = torch.stack((below_inds, above_inds),
                         dim=2)  # [N_rays, N_samples, 2]

    cdf = cdf.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, M+1]
    cdf_g = torch.gather(
        input=cdf, dim=-1, index=inds_g)  # [N_rays, N_samples, 2]

    bins = bins.unsqueeze(1).repeat(1, N_samples,
                                    1)  # [N_rays, N_samples, M+1]
    bins_g = torch.gather(
        input=bins, dim=-1, index=inds_g)  # [N_rays, N_samples, 2]

    # t = (u-cdf_g[:, :, 0]) / (cdf_g[:, :, 1] - cdf_g[:, :, 0] + TINY_NUMBER)
    # fix numeric issue
    denom = cdf_g[:, :, 1] - cdf_g[:, :, 0]  # [N_rays, N_samples]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[:, :, 0]) / denom

    samples = bins_g[:, :, 0] + t * (bins_g[:, :, 1] - bins_g[:, :, 0])

    return samples


def sample_along_camera_ray(ray_o,
                            ray_d,
                            depth_range,
                            N_samples,
                            inv_uniform=False,
                            det=False):
    """Sampling along the camera ray.

    Args:
        ray_o (tensor): Origin of the ray in scene coordinate system;
            tensor of shape [N_rays, 3]
        ray_d (tensor): Homogeneous ray direction vectors in
            scene coordinate system; tensor of shape [N_rays, 3]
        depth_range (tuple): [near_depth, far_depth]
        inv_uniform (bool): If True,uniformly sampling inverse depth.
        det (bool): If True, will perform deterministic sampling.
    Returns:
        pts (tensor): Tensor of shape [N_rays, N_samples, 3]
        z_vals (tensor): Tensor of shape [N_rays, N_samples]
    """
    # will sample inside [near_depth, far_depth]
    # assume the nearest possible depth is at least (min_ratio * depth)
    near_depth_value = depth_range[0]
    far_depth_value = depth_range[1]
    assert near_depth_value > 0 and far_depth_value > 0 \
        and far_depth_value > near_depth_value

    near_depth = near_depth_value * torch.ones_like(ray_d[..., 0])

    far_depth = far_depth_value * torch.ones_like(ray_d[..., 0])

    if inv_uniform:
        start = 1. / near_depth  # [N_rays,]
        step = (1. / far_depth - start) / (N_samples - 1)
        inv_z_vals = torch.stack([start + i * step for i in range(N_samples)],
                                 dim=1)  # [N_rays, N_samples]
        z_vals = 1. / inv_z_vals
    else:
        start = near_depth
        step = (far_depth - near_depth) / (N_samples - 1)
        z_vals = torch.stack([start + i * step for i in range(N_samples)],
                             dim=1)  # [N_rays, N_samples]

    if not det:
        # get intervals between samples
        mids = .5 * (z_vals[:, 1:] + z_vals[:, :-1])
        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
        lower = torch.cat([z_vals[:, 0:1], mids], dim=-1)
        # uniform samples in those intervals
        t_rand = torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * t_rand  # [N_rays, N_samples]

    ray_d = ray_d.unsqueeze(1).repeat(1, N_samples,
                                      1)  # [N_rays, N_samples, 3]
    ray_o = ray_o.unsqueeze(1).repeat(1, N_samples, 1)
    pts = z_vals.unsqueeze(2) * ray_d + ray_o  # [N_rays, N_samples, 3]
    return pts, z_vals


# ray rendering of nerf
def raw2outputs(raw, z_vals, mask, white_bkgd=False):
    """Transform raw data to outputs:

    Args:
        raw(tensor):Raw network output.Tensor of shape [N_rays, N_samples, 4]
        z_vals(tensor):Depth of point samples along rays.
            Tensor of shape [N_rays, N_samples]
        ray_d(tensor):[N_rays, 3]

    Returns:
        ret(dict):
            -rgb(tensor):[N_rays, 3]
            -depth(tensor):[N_rays,]
            -weights(tensor):[N_rays,]
            -depth_std(tensor):[N_rays,]
    """
    rgb = raw[:, :, :3]  # [N_rays, N_samples, 3]
    sigma = raw[:, :, 3]  # [N_rays, N_samples]

    # note: we did not use the intervals here,
    # because in practice different scenes from COLMAP can have
    # very different scales, and using interval can affect
    # the model's generalization ability.
    # Therefore we don't use the intervals for both training and evaluation.
    sigma2alpha = lambda sigma, dists: 1. - torch.exp(-sigma)  # noqa

    # point samples are ordered with increasing depth
    # interval between samples
    dists = z_vals[:, 1:] - z_vals[:, :-1]
    dists = torch.cat((dists, dists[:, -1:]), dim=-1)  # [N_rays, N_samples]

    alpha = sigma2alpha(sigma, dists)  # [N_rays, N_samples]

    # Eq. (3): T
    T = torch.cumprod(
        1. - alpha + 1e-10, dim=-1)[:, :-1]  # [N_rays, N_samples-1]
    T = torch.cat((torch.ones_like(T[:, 0:1]), T),
                  dim=-1)  # [N_rays, N_samples]

    # maths show weights, and summation of weights along a ray,
    # are always inside [0, 1]
    weights = alpha * T  # [N_rays, N_samples]
    rgb_map = torch.sum(weights.unsqueeze(2) * rgb, dim=1)  # [N_rays, 3]

    if white_bkgd:
        rgb_map = rgb_map + (1. - torch.sum(weights, dim=-1, keepdim=True))

    if mask is not None:
        mask = mask.float().sum(dim=1) > 8
        # should at least have 8 valid observation on the ray,
        # otherwise don't consider its loss
    # TODO: should be considered in loss, 8 is a tradeoff.

    # TODO: weights may be not sum into 1,
    # so should be re-normalized, if 0, should add eps.
    depth_map = torch.sum(
        weights * z_vals, dim=-1) / (
            torch.sum(weights, dim=-1) + 1e-8)
    depth_map = torch.clamp(depth_map, z_vals.min(), z_vals.max())

    ret = OrderedDict([('rgb', rgb_map), ('depth', depth_map),
                       ('weights', weights), ('mask', mask), ('alpha', alpha),
                       ('z_vals', z_vals), ('transparency', T)])

    return ret


def render_rays_func(
        ray_o,
        ray_d,
        mean_volume,
        cov_volume,
        features_2D,
        img,
        aabb,
        near_far_range,
        N_samples,
        N_rand=4096,
        nerf_mlp=None,
        img_meta=None,
        projector=None,
        mode='volume',  # volume and image
        nerf_sample_view=3,
        inv_uniform=False,
        N_importance=0,
        det=False,
        is_train=True,
        white_bkgd=False,
        gt_rgb=None,
        gt_depth=None):

    ret = {
        'outputs_coarse': None,
        'outputs_fine': None,
        'gt_rgb': gt_rgb,
        'gt_depth': gt_depth
    }

    # pts: [N_rays, N_samples, 3]
    # z_vals: [N_rays, N_samples]
    pts, z_vals = sample_along_camera_ray(
        ray_o=ray_o,
        ray_d=ray_d,
        depth_range=near_far_range,
        N_samples=N_samples,
        inv_uniform=inv_uniform,
        det=det)
    N_rays, N_samples = pts.shape[:2]

    if mode == 'image':
        img = img.permute(0, 2, 3, 1).unsqueeze(0)
        train_camera = _compute_projection(img_meta).to(img.device)
        rgb_feat, mask = projector.compute(
            pts, img, train_camera, features_2D, grid_sample=True)
        # RGB_feat: [N_rays, N_samples, N_views, channel]
        # mask: [n_rays, n_samples, n_views, 1]
        pixel_mask = mask[..., 0].sum(
            dim=2
        ) > 1  # [N_rays, N_samples], should at least have 2 observations
        mean, var = compute_mask_points(rgb_feat,
                                        mask)  # [n_rays, n_samples, 1, n_feat]
        globalfeat = torch.cat([mean, var], dim=-1).squeeze(
            2)  # [n_rays, n_samples, 1, 2*n_feat]
        rgb_pts, density_pts = nerf_mlp(pts, ray_d, globalfeat)
        raw_coarse = torch.cat([rgb_pts, density_pts], dim=-1)
        ret['sigma'] = density_pts

    elif mode == 'volume':
        mean_pts, inbound_masks = volume_sampling(pts, mean_volume, aabb)
        cov_pts, inbound_masks = volume_sampling(pts, cov_volume, aabb)
        # This masks is for indicating which points outside of aabb
        img = img.permute(0, 2, 3, 1).unsqueeze(0)
        train_camera = _compute_projection(img_meta).to(img.device)
        _, view_mask = projector.compute(pts, img, train_camera, None)
        pixel_mask = view_mask[..., 0].sum(dim=2) > 1
        # plot_3D_vis(pts, aabb, img, train_camera)
        # [N_rays, N_samples], should at least have 2 observations
        # This mask is for indicating which points do not have projected point
        globalpts = torch.cat([mean_pts, cov_pts], dim=-1)
        rgb_pts, density_pts = nerf_mlp(pts, ray_d, globalpts)
        density_pts = density_pts * inbound_masks.unsqueeze(dim=-1)

        raw_coarse = torch.cat([rgb_pts, density_pts], dim=-1)

    outputs_coarse = raw2outputs(
        raw_coarse, z_vals, pixel_mask, white_bkgd=white_bkgd)
    ret['outputs_coarse'] = outputs_coarse

    return ret


def render_rays(
        ray_batch,
        mean_volume,
        cov_volume,
        features_2D,
        img,
        aabb,
        near_far_range,
        N_samples,
        N_rand=4096,
        nerf_mlp=None,
        img_meta=None,
        projector=None,
        mode='volume',  # volume and image
        nerf_sample_view=3,
        inv_uniform=False,
        N_importance=0,
        det=False,
        is_train=True,
        white_bkgd=False,
        render_testing=False):
    """The function of the nerf rendering.
    Note:
        There is a risk that data augmentation is random origin
        not influence nerf,but influence using nerf mlp
        to estimate volme density.

    Args:
        ray_batch (dict):
            - ray_o :The light poses.
            - ray_d :The ray directions.
            - gt_rgb :The groundtruth of the rgbs.
            - gt_depth :The groundtruth of the depths.
            - nerf_sizes :The shapes of the groundtruth images.
            - denorm_images :The normalized rgb images.
        mean_volume (? | None) :The average volume of the scene.
        cov_volume ():
        features_2D (): 2D features if needed.
        img ():The imgs used for nerf.
        aabb ():
        near_far_range ():
        N_samples (int): Samples along each ray
            (for coarse and fine model).
        N_rand (int):
        nerf_mlp (): The mlp of nerf.
        img_meta ():
        projector ():
        mode ():'volume' or 'image'
        nerf_sample_view ():
        inv_uniform (bool):If true,
            uniformly sample inverse depth for coarse model.
        N_importance (int): Additional samples along each ray
            produced by importance sampling(for fine model).
        det (bool) :If true,will deterministicly sample depths.
        is_train (bool): If True,means in the train mode.
        white_bkgd(bool):
        render_testing(bool):

    Returns:
        ret (dict):
            - output_coarse (dict):
                - rgb (Tensor):
                - depth (Tensor):
                - weights (Tensor):
                - mask (Tensor):
                - alpha (Tensor):
                - z_vals (Tensor):
                - transparency (Tensor):
            - gt_rgb (Tensor):
            - gt_depth (Tensor):
    """
    ray_o = ray_batch['ray_o']
    ray_d = ray_batch['ray_d']
    gt_rgb = ray_batch['gt_rgb']
    gt_depth = ray_batch['gt_depth']
    nerf_sizes = ray_batch['nerf_sizes']
    if is_train:
        ray_o = ray_o.view(-1, 3)
        ray_d = ray_d.view(-1, 3)
        gt_rgb = gt_rgb.view(-1, 3)
        if gt_depth.shape[1] != 0:
            gt_depth = gt_depth.view(-1, 1)
            non_zero_depth = (gt_depth > 0).squeeze(-1)
            ray_o = ray_o[non_zero_depth]
            ray_d = ray_d[non_zero_depth]
            gt_rgb = gt_rgb[non_zero_depth]
            gt_depth = gt_depth[non_zero_depth]
        else:
            gt_depth = None
        total_rays = ray_d.shape[0]
        select_inds = rng.choice(total_rays, size=(N_rand, ), replace=False)
        ray_o = ray_o[select_inds]
        ray_d = ray_d[select_inds]
        gt_rgb = gt_rgb[select_inds]
        if gt_depth is not None:
            gt_depth = gt_depth[select_inds]

        rets = render_rays_func(
            ray_o,
            ray_d,
            mean_volume,
            cov_volume,
            features_2D,
            img,
            aabb,
            near_far_range,
            N_samples,
            N_rand,
            nerf_mlp,
            img_meta,
            projector,
            mode,  # volume and image
            nerf_sample_view,
            inv_uniform,
            N_importance,
            det,
            is_train,
            white_bkgd,
            gt_rgb,
            gt_depth)

    elif render_testing:
        # height, width = img_meta['nerf_sizes'].shape[:2]
        # num_rays = height * width
        # assert ray_o.shape[0] == num_rays
        nerf_size = nerf_sizes[0]
        view_num = ray_o.shape[1]
        H = nerf_size[0][0]
        W = nerf_size[0][1]
        ray_o = ray_o.view(-1, 3)
        ray_d = ray_d.view(-1, 3)
        gt_rgb = gt_rgb.view(-1, 3)
        print(gt_rgb.shape)
        if len(gt_depth) != 0:
            gt_depth = gt_depth.view(-1, 1)
        else:
            gt_depth = None
        assert view_num * H * W == ray_o.shape[0]
        num_rays = ray_o.shape[0]
        results = []
        rgbs = []
        for i in range(0, num_rays, N_rand):
            ray_o_chunck = ray_o[i:i + N_rand, :]
            ray_d_chunck = ray_d[i:i + N_rand, :]

            ret = render_rays_func(ray_o_chunck, ray_d_chunck, mean_volume,
                                   cov_volume, features_2D, img, aabb,
                                   near_far_range, N_samples, N_rand, nerf_mlp,
                                   img_meta, projector, mode, nerf_sample_view,
                                   inv_uniform, N_importance, True, is_train,
                                   white_bkgd, gt_rgb, gt_depth)
            results.append(ret)

        rgbs = []
        depths = []

        if results[0]['outputs_coarse'] is not None:
            for i in range(len(results)):
                rgb = results[i]['outputs_coarse']['rgb']
                rgbs.append(rgb)
                depth = results[i]['outputs_coarse']['depth']
                depths.append(depth)

        rets = {
            'outputs_coarse': {
                'rgb': torch.cat(rgbs, dim=0).view(view_num, H, W, 3),
                'depth': torch.cat(depths, dim=0).view(view_num, H, W, 1),
            },
            'gt_rgb':
            gt_rgb.view(view_num, H, W, 3),
            'gt_depth':
            gt_depth.view(view_num, H, W, 1) if gt_depth is not None else None,
        }
    else:
        rets = None
    return rets
