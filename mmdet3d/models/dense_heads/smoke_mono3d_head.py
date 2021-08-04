import torch
# from mmcv.runner import force_fp32
from torch import nn as nn
from torch.nn import functional as F
from mmdet.core import multi_apply
from mmdet.models.builder import HEADS
from .anchor_free_mono3d_head import AnchorFreeMono3DHead
from mmdet.models.utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
                                                transpose_and_gather_feat)

INF = 1e8
PI = 3.14159


@HEADS.register_module()
class SMOKEMono3DHead(AnchorFreeMono3DHead):
    """Anchor-free head used in SMOKE.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: True.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: True.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: True.
        centerness_alpha: Parameter used to adjust the intensity attenuation
            from the center to the periphery. Default: 2.5.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_dir (dict): Config of direction classification loss.
        loss_attr (dict): Config of attribute classification loss.
        loss_centerness (dict): Config of centerness loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        centerness_branch (tuple[int]): Channels for centerness branch.
            Default: (64, ).
    """  # noqa: E501
    def __init__(
            self,
            num_classes,
            in_channels,
            depth_ref,
            dim_ref,
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(
                type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
            loss_dir=None,
            loss_attr=None,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            init_cfg=None,
            **kwargs):
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_dir=loss_dir,
            loss_attr=loss_attr,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.depth_ref = depth_ref
        self.dim_ref = dim_ref
        self.dim_channel = [3, 4, 5]
        self.ori_channel = [6, 7]

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        self.conv_cls_prev = self._init_branch(  
            conv_channels=self.cls_branch,
            conv_strides=(1, ) * len(self.cls_branch))
        self.conv_cls = nn.Conv2d(self.cls_branch[-1], self.cls_out_channels,
                                  1)
        self.conv_reg_prevs = nn.ModuleList()
        self.conv_regs = nn.ModuleList()
        for i in range(len(self.group_reg_dims)):  
            reg_dim = self.group_reg_dims[i]
            reg_branch_channels = self.reg_branch[i]  # （ 256， ）
            out_channel = self.out_channels[i]  # 256
            if len(reg_branch_channels) > 0:
                self.conv_reg_prevs.append(
                    self._init_branch(
                        conv_channels=reg_branch_channels,
                        conv_strides=(1, ) * len(reg_branch_channels)))
                self.conv_regs.append(nn.Conv2d(out_channel, reg_dim, 1))
            else:
                self.conv_reg_prevs.append(None)
                self.conv_regs.append(
                    nn.Conv2d(self.feat_channels, reg_dim, 1))
        if self.use_direction_classifier:
            self.conv_dir_cls_prev = self._init_branch(
                conv_channels=self.dir_branch,
                conv_strides=(1, ) * len(self.dir_branch))
            self.conv_dir_cls = nn.Conv2d(self.dir_branch[-1], 2, 1)
        if self.pred_attrs:
            self.conv_attr_prev = self._init_branch(
                conv_channels=self.attr_branch,
                conv_strides=(1, ) * len(self.attr_branch))
            self.conv_attr = nn.Conv2d(self.attr_branch[-1], self.num_attrs, 1)

    def forward(self, feats):

        return multi_apply(self.forward_single, feats)
    
    def forward_single(self, x):

        cls_score, bbox_pred, dir_cls_pred, attr_pred, cls_feat, reg_feat = \
            super().forward_single(x)

        cls_score = cls_score.sigmoid()  # turn to 0-1 
        cls_score = cls_score.clamp(min=1e-4, max=1 - 1e-4)
        # (N, C, H, W)
        offset_dims = bbox_pred[:, self.dim_channel, ...].clone()  
        bbox_pred[:, self.dim_channel, ...] = torch.sigmoid(offset_dims) - 0.5  

        vector_ori = bbox_pred[:, self.ori_channel, ...].clone()
        bbox_pred[:, self.ori_channel, ...] = F.normalize(vector_ori)  
        return cls_score, bbox_pred

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas):

        assert len(cls_scores) == len(bbox_preds) == 1
        K = torch.stack([img_meta['cam_intrinsic'] for img_meta in img_metas])
        batch_bboxes, batch_scores, batch_topk_labels = self.decode_heatmap(
            cls_scores[0],
            bbox_preds[0],
            img_metas[0]['batch_input_shape'],
            view=K,
            k=self.test_cfg.topk,
            kernel=self.test_cfg.local_maximum_kernel)

        result_list = []
        for img_id in range(len(img_metas)):            
            bboxes = batch_bboxes[img_id]
            scores = batch_scores[img_id]
            labels = batch_topk_labels[img_id]
            bboxes = img_metas[img_id]['box_type_3d'](
                    bboxes, box_dim=self.bbox_code_size, origin=(0.5, 0.5, 0.5))
            attrs = None
            result_list.append((bboxes, scores, labels, attrs))
        return result_list
            
    def decode_heatmap(self,
                       cls_scores,
                       bbox_preds,
                       view,
                       k=100,
                       kernel=3):
        """Transform outputs into detections raw bbox prediction.
        Args:
            class_score (Tensor): center predict heatmap,
               shape (B, num_classes, H, W).
            bbox_preds(Tensor): box predict regression map
               shape (B,8, H, W).
            view(Tensor): camera intrinsic matrix for batch images
            k (int): Get top k center keypoints from heatmap. Default 100.
            kernel (int): Max pooling kernel for extract local maximum pixels.
               Default 3.
        Returns:
            tuple[torch.Tensor]: Decoded output of CenterNetHead, containing
               the following Tensors:
              - batch_bboxes (Tensor): Coords of each box with shape (B, k, 5)
               - batch_scores (Tensor): score of each box with shape (B, k)
              - batch_topk_labels (Tensor): Categories of each box with \
                  shape (B, k)
        """
        # height, width = cls_scores.shape[2:]
        # inp_h, inp_w = img_shape
        batch_size = cls_scores.shape[0]
        center_heatmap_pred = get_local_maximum(
            cls_scores, kernel=kernel)  # (B, num_classes, H, W)

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        pred_regression = transpose_and_gather_feat(bbox_preds, batch_index)  # (B, K, 8)
        pred_regression_pois = pred_regression.view(-1, 8)  # (B*K, 8)

        pred_proj_points = torch.cat([topk_xs.view(-1, 1), topk_ys.view(-1, 1)], dim=1)  # (batch * k, 2)

        pred_depths_offset = pred_regression_pois[:, 0]  # (b*k)  depth_offset
        pred_proj_offsets = pred_regression_pois[:, 1:3]  # (b*k, 2) center_offset
        pred_dimensions_offsets = pred_regression_pois[:, 3:6]  # (b*k, 3) dim_offset
        pred_orientation = pred_regression_pois[:, 6:]  # rot_y
        pred_depths = self.decode_depth(pred_depths_offset)  # (B*K)
        pred_locations = self.decode_location(
            pred_proj_points,
            pred_proj_offsets,
            pred_depths,
            view,
        ) 
        pred_dimensions = self.decode_dimension(
            batch_topk_labels,
            pred_dimensions_offsets
        )
        pred_rotys = self.decode_orientation(
            pred_orientation,
            pred_locations
        )
        batch_bboxes = torch.cat((pred_locations, pred_dimensions, pred_rotys), dim=1)  # (b*K, 7)
        # batch_bboxes = input_meta['box_type_3d'](
        #     batch_bboxes, box_dim=7, origin=(0.5, 0.5, 0.5))
        batch_bboxes = batch_bboxes.view(batch_size, -1, self.bbox_code_size)
        return batch_bboxes, batch_scores, batch_topk_labels

    def decode_depth(self, depths_offset):
        '''
        Transform depth offset to depth
        '''
        depth_ref = torch.as_tensor(self.depth_ref).to(depths_offset)
        depth = depths_offset * depth_ref[1] + depth_ref[0]

        return depth
    
    def decode_location(self,
                        points,
                        points_offset,
                        depths,
                        Ks,
                        trans_mats):
        '''
        retrieve objects location in camera coordinate based on projected points
        Args:
            points: projected points on feature map in (x, y)
            points_offset: project points offset in (delata_x, delta_y)
            depths: object depth z
            Ks: camera intrinsic matrix, shape = [N, 3, 3]  
            trans_mats: transformation matrix from image to feature map, shape = [N, 3, 3]   # image -> feature map

        Returns:
            locations: objects location, shape = [N, 3]  # N = batch * k ?
        '''
        device = points.device

        Ks = Ks.to(device=device)
        trans_mats = trans_mats.to(device=device)

        # number of points
        N = points_offset.shape[0]
        # batch size
        N_batch = Ks.shape[0]
        batch_id = torch.arange(N_batch).unsqueeze(1)  # (N_batch, 1)
        obj_id = batch_id.repeat(1, N // N_batch).flatten()  # (N_batch, k) ->  (N_batch * k)

        trans_mats_inv = trans_mats.inverse()[obj_id]  # (N_batch * k, 3, 3)
        Ks_inv = Ks.inverse()[obj_id]  # (N_batch * k, 3, 3)

        points = points.view(-1, 2)   # （N_batch * k, 2)
        assert points.shape[0] == N
        proj_points = points + points_offset
        # transform project points in homogeneous form.
        proj_points_extend = torch.cat(
            (proj_points, torch.ones(N, 1).to(device=device)), dim=1)  # （N_batch * k, 3)
        # expand project points as [N, 3, 1]
        proj_points_extend = proj_points_extend.unsqueeze(-1)
        # transform project points back on image
        proj_points_img = torch.matmul(trans_mats_inv, proj_points_extend)
        # with depth
        proj_points_img = proj_points_img * depths.view(N, -1, 1)  
        # transform image coordinates back to object locations
        locations = torch.matmul(Ks_inv, proj_points_img)  # [N, 3, 1]

        return locations.squeeze(2)  # ( N, 3)

    def decode_dimension(self, cls_id, dims_offset):
        '''
        retrieve object dimensions
        Args:
            cls_id: each object id
            dims_offset: dimension offsets, shape = (N, 3)

        Returns:

        '''
        cls_id = cls_id.flatten().long()
        dim_ref = torch.as_tensor(self.dim_ref).to(dims_offset)
        dims_select = dim_ref[cls_id, :]
        dimensions = dims_offset.exp() * dims_select

        return dimensions

    def decode_orientation(self, vector_ori, locations, flip_mask=None):
        '''
        retrieve object orientation
        Args:
            vector_ori: local orientation in [sin, cos] format
            locations: object location

        Returns: for training we only need roty
                 for testing we need both alpha and roty

        '''

        locations = locations.view(-1, 3)
        rays = torch.atan(locations[:, 0] / (locations[:, 2] + 1e-7))
        alphas = torch.atan(vector_ori[:, 0] / (vector_ori[:, 1] + 1e-7))

        # get cosine value positive and negtive index.
        cos_pos_idx = (vector_ori[:, 1] >= 0).nonzero()
        cos_neg_idx = (vector_ori[:, 1] < 0).nonzero()

        alphas[cos_pos_idx] -= PI / 2
        alphas[cos_neg_idx] += PI / 2

        # retrieve object rotation y angle.
        rotys = alphas + rays

        # in training time, it does not matter if angle lies in [-PI, PI]
        # it matters at inference time? todo: does it really matter if it exceeds.
        larger_idx = (rotys > PI).nonzero()
        small_idx = (rotys < -PI).nonzero()

        if len(larger_idx) != 0:
            rotys[larger_idx] -= 2 * PI
        if len(small_idx) != 0:
            rotys[small_idx] += 2 * PI

        if flip_mask is not None:
            fm = flip_mask.flatten()
            rotys_flip = fm.float() * rotys

            rotys_flip_pos_idx = rotys_flip > 0
            rotys_flip_neg_idx = rotys_flip < 0
            rotys_flip[rotys_flip_pos_idx] -= PI
            rotys_flip[rotys_flip_neg_idx] += PI

            rotys_all = fm.float() * rotys_flip + (1 - fm.float()) * rotys

            return rotys_all

        else:
            return rotys
    
    def get_targets(self):
        """Compute regression, classification and centerss targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
            gt_bboxes_3d_list (list[Tensor]): 3D Ground truth bboxes of each
                image, each has shape (num_gt, bbox_code_size).
            gt_labels_3d_list (list[Tensor]): 3D Ground truth labels of each
                box, each has shape (num_gt,).
            centers2d_list (list[Tensor]): Projected 3D centers onto 2D image,
                each has shape (num_gt, 2).
            depths_list (list[Tensor]): Depth of projected 3D centers onto 2D
                image, each has shape (num_gt, 1).
            attr_labels_list (list[Tensor]): Attribute labels of each box,
                each has shape (num_gt,).
        """
        return None