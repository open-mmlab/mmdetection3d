import torch
from torch import nn as nn

from mmdet3d.core.bbox import Coord3DMode, points_cam2img
from tools.data_converter.sunrgbd_data_utils import SUNRGBD_Calibration
from ..registry import FUSION_LAYERS


@FUSION_LAYERS.register_module()
class VoteFusion(nn.Module):
    """Fuse 2d features from 3d seeds.

    Args:
        num_classes (int): number of classes.
        txt_sample_mode (str): sample mode for texture cues.
    """

    def __init__(self,
                 num_classes=None,
                 txt_sample_mode='bilinear',
                 img_norm_cfg=None):
        super(VoteFusion, self).__init__()
        self.num_classes = num_classes
        self.txt_sample_mode = txt_sample_mode
        self.img_norm_cfg = img_norm_cfg

    def forward(self, img, bboxes_2d, seeds_3d, pts_feats, calibs, img_metas):
        """Forward function.

        Args:
            img_feats (list[torch.Tensor]): Image features.
            pts: [list[torch.Tensor]]: A batch of points with shape N x 3.
            pts_feats (torch.Tensor): A tensor consist of point features of the
                total batch.
            img_metas (list[dict]): Meta information of images.
            calibs: Camera calibration information of the images.

        Returns:
            torch.Tensor: Fused features of each point.
        """
        seeds_3d_origin, seeds_2d_origin, seeds_2d_trans, bboxes_2d_origin = \
            self.map_seeds_and_bboxes(seeds_3d, bboxes_2d, calibs, img_metas)
        img_features = self.calculate_cues(bboxes_2d_origin, bboxes_2d,
                                           seeds_2d_origin, seeds_2d_trans,
                                           seeds_3d_origin, seeds_3d, img,
                                           img_metas, calibs)
        return img_features

    def calculate_cues(self, bboxes_2d_origin, bboxes_2d_trans,
                       seeds_2d_origin, seeds_2d_trans, seeds_3d_origin,
                       seeds_3d_trans, imgs, img_metas, calibs):
        """Calculate semantic/texture/geometric cues.

        Args:
            bboxes_2d_origin (list[torch.Tensor]): Bounding box before
                transformation.
            bboxes_2d_trans (list[torch.Tensor]): Bounding box after
                transformation.
            seeds_2d_origin (list[torch.Tensor]): Seed points before
                transformation, mapped to 2d.
            seeds_2d_trans (list[torch.Tensor]): Seed points after
                transformation, mapped to 2d.
            seeds_3d_origin (list[torch.Tensor]): Seed points before
                transformation.
            seeds_3d_trans (list[torch.Tensor]): Seed points after
                transformation.
            imgs: [list[torch.Tensor]]: Images after transformation.
            img_metas (list[dict]): Meta information of images.
            calibs: Camera calibration information of the images.

        Returns:
            torch.Tensor: Concatenated cues of each point.
        """
        img_features = []
        for i, data in enumerate(
                zip(bboxes_2d_origin, bboxes_2d_trans, seeds_2d_origin,
                    seeds_2d_trans, seeds_3d_origin, seeds_3d_trans)):
            bbox_origin, bbox_trans, seed_2d_origin, \
                seed_2d_trans, seed_3d_origin, seed_3d_trans = data
            raw_img = imgs[i]
            img_shape = img_metas[i]['img_shape']
            sun_calib = SUNRGBD_Calibration(
                Rt=calibs['Rt'][i], K=calibs['K'][i])
            bbox_num = bbox_trans.shape[0]
            seed_num = seed_2d_trans.shape[0]
            bbox_expanded = bbox_origin.view(1, bbox_num,
                                             -1).expand(seed_num, -1, -1)
            seed_2d_expanded = seed_2d_origin.view(seed_num, 1, -1).expand(
                -1, bbox_num, -1)
            seed_3d_expanded = seed_3d_origin.view(seed_num, 1, -1).expand(
                -1, bbox_num, -1)
            seed_2d_expanded_x, seed_2d_expanded_y = seed_2d_expanded.split(
                1, dim=-1)
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

            # semantic cues, dim=20
            sem_cue = torch.zeros_like(bbox_expanded_conf).expand(
                -1, -1, self.num_classes)
            sem_cue = sem_cue.scatter(-1, bbox_expanded_cls.long(),
                                      bbox_expanded_conf)

            # texture cues, dim=3
            raw_img_flatten = raw_img.view(3, -1)
            img_norm_mean = raw_img.new_tensor(self.img_norm_cfg['mean'])[:,
                                                                          None]
            raw_img_flatten = raw_img_flatten + img_norm_mean - 128
            raw_img_flatten = raw_img_flatten / 128
            assert (raw_img_flatten <= 1).all() and (raw_img_flatten >=
                                                     -1).all()

            seed_2d_trans_flatten = seed_2d_trans[:, 1] * \
                img_shape[1] + seed_2d_trans[:, 0]
            seed_2d_trans_expanded = seed_2d_trans_flatten.unsqueeze(0).expand(
                3, -1).long()
            # point_feature = F.grid_sample(
            #     raw_img[None],
            #     seed_2d_trans[None][None],
            #     mode=self.txt_sample_mode,
            #     padding_mode='zeros',
            #     align_corners=True)
            txt_cue = torch.gather(
                raw_img_flatten, dim=-1, index=seed_2d_trans_expanded)
            txt_cue = txt_cue.transpose(1, 0)
            txt_cue = txt_cue.unsqueeze(1).expand(-1, bbox_num, -1)

            # geometric cues, dim=5
            delta_u = bbox_expanded_midx - seed_2d_expanded_x
            delta_v = bbox_expanded_midy - seed_2d_expanded_y
            x_3d, y_3d, z_3d = seed_3d_expanded.split(1, dim=-1)

            z_div_f_u = z_3d / sun_calib.f_u
            z_div_f_v = z_3d / sun_calib.f_v

            geo_0 = delta_u * z_div_f_u
            geo_1 = delta_v * z_div_f_v
            geo_2 = geo_0 + x_3d
            geo_3 = geo_1 + y_3d
            geo_4 = z_3d
            geo_xy = torch.cat([geo_0, geo_1, torch.zeros_like(geo_0)], dim=-1)
            geo_xy = Coord3DMode.convert_point(
                geo_xy.view((-1, 3)).double(),
                Coord3DMode.CAM,
                Coord3DMode.DEPTH,
                rt_mat=calibs['Rt'][i]).float()
            geo_xy = geo_xy.view(seed_num, -1, 3)
            seed_depth = seed_3d_trans[:, None, 1]
            delta_depth = geo_xy[:, :, 1]
            ratio = seed_depth / (seed_depth + delta_depth)
            geo_xy = geo_xy * ratio[:, :, None]

            geo_vec = torch.cat([geo_2, geo_3, geo_4], dim=-1)
            geo_vec = sun_calib.project_camera_to_upright_depth(
                geo_vec.view((-1, 3)).cpu().numpy())
            geo_vec = geo_0.new_tensor(geo_vec).view(seed_num, -1, 3)
            geo_vec_norm = geo_vec.norm(dim=-1, keepdim=True)
            geo_vec = geo_vec / geo_vec_norm
            geo_cue = torch.cat([geo_xy[:, :, [0, 2]], geo_vec], dim=-1)

            # seed_num, bbox_num, 18
            all_cue = torch.cat([sem_cue, txt_cue, geo_cue], dim=-1)

            valid_2d_mask = seed_2d_in_bbox.sum(dim=1)  # seed_num, 1
            box_asgn = (delta_u * delta_u + delta_v * delta_v).argmin(dim=1)
            box_asgn = box_asgn.unsqueeze(-1).expand(-1, -1, 18)
            pos_img_feature = torch.gather(all_cue, 1, box_asgn).squeeze(1)
            neg_img_feature = torch.zeros_like(pos_img_feature)
            valid_2d_mask_expanded = valid_2d_mask.expand(-1, 18).bool()

            img_feature = torch.where(valid_2d_mask_expanded, pos_img_feature,
                                      neg_img_feature)  # seed_num, 18
            img_feature = img_feature.transpose(1, 0)
            img_features.append(img_feature)
        return torch.stack(img_features, dim=0)

    def map_seeds_and_bboxes(self, seeds_3d, bboxes_2d, calibs, img_metas):
        """Calculate semantic/texture/geometric cues.

        Args:
            bboxes_2d (list[torch.Tensor]): Bounding box after
                transformation.
            seeds_3d (list[torch.Tensor]): Seed points after
                transformation.
            img_metas (list[dict]): Meta information of images.
            calibs: Camera calibration information of the images.

        Returns:
            tuple: Bounding boxes and seed points (in 2d/3d)
                before and after transformation
        """
        seeds_2d_trans = []
        seeds_3d_origin = []
        seeds_2d_origin = []
        bboxes_2d_origin = []
        for i in range(len(seeds_3d)):
            img_meta = img_metas[i]
            pts = seeds_3d[i]
            bbox_2d = bboxes_2d[i].float()
            img_shape = img_meta['img_shape']
            ori_shape = img_meta['ori_shape']
            pcd_scale_factor = (
                img_meta['pcd_scale_factor']
                if 'pcd_scale_factor' in img_meta.keys() else 1)
            pcd_trans_factor = (
                pts.new_tensor(img_meta['pcd_trans'])
                if 'pcd_trans' in img_meta.keys() else 0)
            pcd_rotate_mat = (
                pts.new_tensor(img_meta['pcd_rotation'])
                if 'pcd_rotation' in img_meta.keys() else
                torch.eye(3).type_as(pts).to(pts.device))
            img_scale_factor = (
                pts.new_tensor(img_meta['scale_factor'][:2])
                if 'scale_factor' in img_meta.keys() else 1)
            pcd_horizontal_flip = img_meta[
                'pcd_horizontal_flip'] if 'pcd_horizontal_flip' in \
                img_meta.keys() else False
            img_flip = img_meta['flip'] if 'flip' in \
                img_meta.keys() else False
            img_crop_offset = (
                pts.new_tensor(img_meta['img_crop_offset'])
                if 'img_crop_offset' in img_meta.keys() else 0)

            pts -= pcd_trans_factor
            pts /= pcd_scale_factor
            pts = pts @ pcd_rotate_mat.inverse()

            if pcd_horizontal_flip:
                pts[..., 0] = -pts[..., 0]

            # origin_pts means the pts in the not upright camera coordinate
            origin_pts = Coord3DMode.convert_point(
                pts.double(),
                Coord3DMode.DEPTH,
                Coord3DMode.CAM,
                rt_mat=calibs['Rt'][i])
            origin_pix = points_cam2img(origin_pts, calibs['K'][i]).float()
            origin_pts = origin_pts.float()

            # transformed pixel coordinates
            trans_pix = origin_pix.clone()
            trans_pix[:, 0] = trans_pix[:, 0] * img_scale_factor[0]  # u
            trans_pix[:, 1] = trans_pix[:, 1] * img_scale_factor[1]  # v
            trans_pix -= img_crop_offset

            # origin bboxes
            bbox_2d_origin = bbox_2d.clone()

            img_h, img_w, _ = img_shape
            ori_h, ori_w, _ = ori_shape
            if img_flip:
                # by default we take it as horizontal flip
                # use img_shape before padding for flip
                trans_pix[:, 0] = img_w - trans_pix[:, 0]
                bbox_2d_origin_r = img_w - bbox_2d_origin[:, 0]
                bbox_2d_origin_l = img_w - bbox_2d_origin[:, 2]
                bbox_2d_origin[:, 0] = bbox_2d_origin_l
                bbox_2d_origin[:, 2] = bbox_2d_origin_r

            # bboxes rescale
            bbox_2d_origin[:, 0] = bbox_2d_origin[:, 0] / img_scale_factor[0]
            bbox_2d_origin[:, 2] = bbox_2d_origin[:, 2] / img_scale_factor[0]
            bbox_2d_origin[:, 1] = bbox_2d_origin[:, 1] / img_scale_factor[1]
            bbox_2d_origin[:, 3] = bbox_2d_origin[:, 3] / img_scale_factor[1]

            bboxes_2d_origin.append(bbox_2d_origin)
            seeds_2d_trans.append(trans_pix)
            seeds_2d_origin.append(origin_pix)
            seeds_3d_origin.append(origin_pts)

        return (seeds_3d_origin, seeds_2d_origin, seeds_2d_trans,
                bboxes_2d_origin)
