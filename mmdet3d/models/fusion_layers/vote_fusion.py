# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn as nn

from mmdet3d.core.bbox import points_cam2img
from ..builder import FUSION_LAYERS
from . import apply_3d_transformation, bbox_2d_transform, coord_2d_transform

EPS = 1e-6


@FUSION_LAYERS.register_module()
class VoteFusion(nn.Module):
    """Fuse 2d features from 3d seeds.

    Args:
        num_classes (int): number of classes.
        max_imvote_per_pixel (int): max number of imvotes.
    """

    def __init__(self, num_classes=10, max_imvote_per_pixel=3):
        super(VoteFusion, self).__init__()
        self.num_classes = num_classes
        self.max_imvote_per_pixel = max_imvote_per_pixel

    def forward(self, imgs, bboxes_2d_rescaled, seeds_3d_depth, img_metas):
        """Forward function.

        Args:
            imgs (list[torch.Tensor]): Image features.
            bboxes_2d_rescaled (list[torch.Tensor]): 2D bboxes.
            seeds_3d_depth (torch.Tensor): 3D seeds.
            img_metas (list[dict]): Meta information of images.

        Returns:
            torch.Tensor: Concatenated cues of each point.
            torch.Tensor: Validity mask of each feature.
        """
        img_features = []
        masks = []
        for i, data in enumerate(
                zip(imgs, bboxes_2d_rescaled, seeds_3d_depth, img_metas)):
            img, bbox_2d_rescaled, seed_3d_depth, img_meta = data
            bbox_num = bbox_2d_rescaled.shape[0]
            seed_num = seed_3d_depth.shape[0]

            img_shape = img_meta['img_shape']
            img_h, img_w, _ = img_shape

            # first reverse the data transformations
            xyz_depth = apply_3d_transformation(
                seed_3d_depth, 'DEPTH', img_meta, reverse=True)

            # project points from depth to image
            depth2img = xyz_depth.new_tensor(img_meta['depth2img'])
            uvz_origin = points_cam2img(xyz_depth, depth2img, True)
            z_cam = uvz_origin[..., 2]
            uv_origin = (uvz_origin[..., :2] - 1).round()

            # rescale 2d coordinates and bboxes
            uv_rescaled = coord_2d_transform(img_meta, uv_origin, True)
            bbox_2d_origin = bbox_2d_transform(img_meta, bbox_2d_rescaled,
                                               False)

            if bbox_num == 0:
                imvote_num = seed_num * self.max_imvote_per_pixel

                # use zero features
                two_cues = torch.zeros((15, imvote_num),
                                       device=seed_3d_depth.device)
                mask_zero = torch.zeros(
                    imvote_num - seed_num, device=seed_3d_depth.device).bool()
                mask_one = torch.ones(
                    seed_num, device=seed_3d_depth.device).bool()
                mask = torch.cat([mask_one, mask_zero], dim=0)
            else:
                # expand bboxes and seeds
                bbox_expanded = bbox_2d_origin.view(1, bbox_num, -1).expand(
                    seed_num, -1, -1)
                seed_2d_expanded = uv_origin.view(seed_num, 1,
                                                  -1).expand(-1, bbox_num, -1)
                seed_2d_expanded_x, seed_2d_expanded_y = \
                    seed_2d_expanded.split(1, dim=-1)

                bbox_expanded_l, bbox_expanded_t, bbox_expanded_r, \
                    bbox_expanded_b, bbox_expanded_conf, bbox_expanded_cls = \
                    bbox_expanded.split(1, dim=-1)
                bbox_expanded_midx = (bbox_expanded_l + bbox_expanded_r) / 2
                bbox_expanded_midy = (bbox_expanded_t + bbox_expanded_b) / 2

                seed_2d_in_bbox_x = (seed_2d_expanded_x > bbox_expanded_l) * \
                    (seed_2d_expanded_x < bbox_expanded_r)
                seed_2d_in_bbox_y = (seed_2d_expanded_y > bbox_expanded_t) * \
                    (seed_2d_expanded_y < bbox_expanded_b)
                seed_2d_in_bbox = seed_2d_in_bbox_x * seed_2d_in_bbox_y

                # semantic cues, dim=class_num
                sem_cue = torch.zeros_like(bbox_expanded_conf).expand(
                    -1, -1, self.num_classes)
                sem_cue = sem_cue.scatter(-1, bbox_expanded_cls.long(),
                                          bbox_expanded_conf)

                # bbox center - uv
                delta_u = bbox_expanded_midx - seed_2d_expanded_x
                delta_v = bbox_expanded_midy - seed_2d_expanded_y

                seed_3d_expanded = seed_3d_depth.view(seed_num, 1, -1).expand(
                    -1, bbox_num, -1)

                z_cam = z_cam.view(seed_num, 1, 1).expand(-1, bbox_num, -1)
                imvote = torch.cat(
                    [delta_u, delta_v,
                     torch.zeros_like(delta_v)], dim=-1).view(-1, 3)
                imvote = imvote * z_cam.reshape(-1, 1)
                imvote = imvote @ torch.inverse(depth2img.t())

                # apply transformation to lifted imvotes
                imvote = apply_3d_transformation(
                    imvote, 'DEPTH', img_meta, reverse=False)

                seed_3d_expanded = seed_3d_expanded.reshape(imvote.shape)

                # ray angle
                ray_angle = seed_3d_expanded + imvote
                ray_angle /= torch.sqrt(torch.sum(ray_angle**2, -1) +
                                        EPS).unsqueeze(-1)

                # imvote lifted to 3d
                xz = ray_angle[:, [0, 2]] / (ray_angle[:, [1]] + EPS) \
                    * seed_3d_expanded[:, [1]] - seed_3d_expanded[:, [0, 2]]

                # geometric cues, dim=5
                geo_cue = torch.cat([xz, ray_angle],
                                    dim=-1).view(seed_num, -1, 5)

                two_cues = torch.cat([geo_cue, sem_cue], dim=-1)
                # mask to 0 if seed not in bbox
                two_cues = two_cues * seed_2d_in_bbox.float()

                feature_size = two_cues.shape[-1]
                # if bbox number is too small, append zeros
                if bbox_num < self.max_imvote_per_pixel:
                    append_num = self.max_imvote_per_pixel - bbox_num
                    append_zeros = torch.zeros(
                        (seed_num, append_num, 1),
                        device=seed_2d_in_bbox.device).bool()
                    seed_2d_in_bbox = torch.cat(
                        [seed_2d_in_bbox, append_zeros], dim=1)
                    append_zeros = torch.zeros(
                        (seed_num, append_num, feature_size),
                        device=two_cues.device)
                    two_cues = torch.cat([two_cues, append_zeros], dim=1)
                    append_zeros = torch.zeros((seed_num, append_num, 1),
                                               device=two_cues.device)
                    bbox_expanded_conf = torch.cat(
                        [bbox_expanded_conf, append_zeros], dim=1)

                # sort the valid seed-bbox pair according to confidence
                pair_score = seed_2d_in_bbox.float() + bbox_expanded_conf
                # and find the largests
                mask, indices = pair_score.topk(
                    self.max_imvote_per_pixel,
                    dim=1,
                    largest=True,
                    sorted=True)

                indices_img = indices.expand(-1, -1, feature_size)
                two_cues = two_cues.gather(dim=1, index=indices_img)
                two_cues = two_cues.transpose(1, 0)
                two_cues = two_cues.reshape(-1, feature_size).transpose(
                    1, 0).contiguous()

                # since conf is ~ (0, 1), floor gives us validity
                mask = mask.floor().int()
                mask = mask.transpose(1, 0).reshape(-1).bool()

            # clear the padding
            img = img[:, :img_shape[0], :img_shape[1]]
            img_flatten = img.reshape(3, -1).float()
            img_flatten /= 255.

            # take the normalized pixel value as texture cue
            uv_rescaled[:, 0] = torch.clamp(uv_rescaled[:, 0].round(), 0,
                                            img_shape[1] - 1)
            uv_rescaled[:, 1] = torch.clamp(uv_rescaled[:, 1].round(), 0,
                                            img_shape[0] - 1)
            uv_flatten = uv_rescaled[:, 1].round() * \
                img_shape[1] + uv_rescaled[:, 0].round()
            uv_expanded = uv_flatten.unsqueeze(0).expand(3, -1).long()
            txt_cue = torch.gather(img_flatten, dim=-1, index=uv_expanded)
            txt_cue = txt_cue.unsqueeze(1).expand(-1,
                                                  self.max_imvote_per_pixel,
                                                  -1).reshape(3, -1)

            # append texture cue
            img_feature = torch.cat([two_cues, txt_cue], dim=0)
            img_features.append(img_feature)
            masks.append(mask)

        return torch.stack(img_features, 0), torch.stack(masks, 0)
