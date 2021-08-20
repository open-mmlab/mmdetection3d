import torch
# from mmcv.runner import force_fp32
from torch import nn as nn
from torch.nn import functional as F

from mmdet.core import multi_apply
from mmdet.core.bbox.builder import build_bbox_coder
from mmdet.models.builder import HEADS
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from mmdet.models.utils.gaussian_target import (get_local_maximum,
                                                get_topk_from_heatmap,
                                                transpose_and_gather_feat)
from .anchor_free_mono3d_head import AnchorFreeMono3DHead


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

    def __init__(self,
                 num_classes,
                 in_channels,
                 bbox_coder=None,
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
        self.dim_channel = [3, 4, 5]
        self.ori_channel = [6, 7]
        self.bbox_coder = build_bbox_coder(bbox_coder)

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
        offset_dims = bbox_pred[:, self.dim_channel, ...].clone(
        )  # 预测的是 dim_offset, 我们有一个 predefined 的平均 h, w, l.
        bbox_pred[:, self.dim_channel,
                  ...] = offset_dims.sigmoid() - 0.5  # 统一范围
        # (N, C, H, W)
        vector_ori = bbox_pred[:, self.ori_channel, ...].clone()
        bbox_pred[:, self.ori_channel,
                  ...] = F.normalize(vector_ori)  # 转换为 sin 和 cos结果
        return cls_score, bbox_pred

    def get_bboxes(self, cls_scores, bbox_preds, img_metas, rescale=None):

        assert len(cls_scores) == len(bbox_preds) == 1
        K = torch.stack([img_meta['cam_intrinsic']
                         for img_meta in img_metas])  # (b, 4, 4)
        batch_bboxes, batch_scores, batch_topk_labels = self.decode_heatmap(
            cls_scores[0],
            bbox_preds[0],
            img_metas,
            cam_intrinsics=K,
            k=100,
            kernel=3)

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
                       cls_score,
                       reg_pred,
                       img_metas,
                       cam_intrinsics,
                       k=100,
                       kernel=3):
        """Transform outputs into detections raw bbox predictions.
        Args:
            class_score (Tensor): center predict heatmap,
                shape (B, num_classes, H, W).
            reg_pred (Tensor): Regression map.
                shape (B, channel, H , W).
            img_metas (List[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cam_intrinsics (Tensor): camera intrinsic matrix.
                shape (B, 4, 4)
            k (int): Get top k center keypoints from heatmap. Default 100.
            kernel (int): Max pooling kernel for extract local maximum pixels.
               Default 3.
        Returns:
            tuple[torch.Tensor]: Decoded output of SMOKEHead, containing
               the following Tensors:
              - batch_bboxes (Tensor): Coords of each 3D box with shape
                    (B, k, 7)
              - batch_scores (Tensor): Scores of each 3D box with shape
                    (B, k)
              - batch_topk_labels (Tensor): Categories of each 3D box with \
                  shape (B, k)
        """
        img_h, img_w = img_metas[0]['pad_shape'][:2]
        bs, _, feat_h, feat_w = cls_score.shape
        ratio = float(feat_w / img_w)
        center_heatmap_pred = get_local_maximum(
            cls_score, kernel=kernel)  # (B, num_classes, H, W)

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        regression = transpose_and_gather_feat(reg_pred,
                                               batch_index)  # (B, K, 8)
        regression = regression.view(-1, 8)  # (B*K, 8)

        points = torch.cat(
            [topk_xs.view(-1, 1), topk_ys.view(-1, 1)],
            dim=1)  # (batch * k, 2)
        locations, dimensions, orientations = self.bbox_coder.decode(
            regression, points, batch_topk_labels, cam_intrinsics, ratio)

        batch_bboxes = torch.cat((locations, dimensions, orientations),
                                 dim=1)  # (b*k, 7)
        batch_bboxes = batch_bboxes.view(bs, -1,
                                         self.bbox_code_size)  # (b*k, 7)
        return batch_bboxes, batch_scores, batch_topk_labels

    def get_predictions(self, labels3d, centers2d, gt_locations, gt_dimensions,
                        gt_orientations, img_metas, pred_reg, ratio):
        """Compute regression, classification and centerss targets for points
        centers2d (list[Tensor]): Projected 3D label centers onto 2D image,
        each has shape (num_gt, 2).

        img_metas (list[dict]): Meta information of each image, e.g., image
        size, scaling factor, etc.
        """
        batch, channel = pred_reg.shape[0], pred_reg.shape[1]
        w = pred_reg.shape[3]
        cam_intrinsics = torch.stack(
            [img_meta['cam_intrinsic'] for img_meta in img_metas])
        ct_num = [center2d.shape[0] for center2d in centers2d]
        max_objs = max(ct_num)

        indexs = torch.zeros([batch, ct_num],
                             dtype=torch.uint8).to(centers2d[0].device)

        batch_centers2d = centers2d[0].new_zeros((batch, max_objs, 2))

        for i in range(batch):
            indexs[i, :ct_num[i]] = 1
            batch_centers2d[i, :ct_num[i]] = centers2d[i]

        indexs = indexs.flatten()  # ( batch * ct_num )

        if len(centers2d.shape) == 3:
            centers2d_inds = \
                batch_centers2d[:, :, 1] * w + batch_centers2d[:, :, 0]
        centers2d_inds = centers2d_inds.view(batch,
                                             -1)  # ( batch * ct_num, 1 )
        batch_centers2d = batch_centers2d.view(-1, 2)  # ( batch * ct_num, 2 )
        pred_regression = transpose_and_gather_feat(
            pred_reg, centers2d_inds)  # (B, max_objs, channel)
        pred_regression_pois = pred_regression.view(-1,
                                                    channel)  # (N, channel)

        # (B * max_objs, xxx)
        locations, dimensions, orientations = self.bbox_coder.decode(
            pred_regression_pois, batch_centers2d, labels3d, cam_intrinsics,
            ratio)

        # (B * max_objs, xxx) -> ( batch_num_gt, xxx )
        locations, dimensions, orientations = locations[indexs], dimensions[
            indexs], orientations[indexs]

        assert len(locations) == len(gt_locations)
        assert len(dimensions) == len(gt_dimensions)
        assert len(orientations) == len(gt_orientations)
        bbox3d_rotys = self.bbox_coder.encode(gt_locations, gt_dimensions,
                                              orientations, img_metas)
        bbox3d_dims = self.bbox_coder.encode(gt_locations, dimensions,
                                             gt_orientations, img_metas)
        bbox3d_locs = self.bbox_coder.encode(locations, gt_dimensions,
                                             gt_orientations, img_metas)

        pred_bboxes = dict(ori=bbox3d_rotys, dim=bbox3d_dims, loc=bbox3d_locs)

        return pred_bboxes

    def get_targets(self, gt_bboxes, gt_labels, gt_bboxes_3d, gt_labels_3d,
                    centers2d, depths, feat_shape, img_shape):
        """Compute regression, classification and centerss targets for points
        in multiple images.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
            gt_bboxes_3d (list[Tensor]): 3D Ground truth bboxes of each
                image, each has shape (num_gt, bbox_code_size).
            gt_labels_3d (list[Tensor]): 3D Ground truth labels of each
                box, each has shape (num_gt,).
            centers2d (list[Tensor]): Projected 3D centers onto 2D image,
                each has shape (num_gt, 2).
            depths (list[Tensor]): Depth of projected 3D centers onto 2D
                image, each has shape (num_gt, 1).
            feat_shape (list[int]): feature map shape with value [B, _, H, W].
            img_shape (list[int]): image shape in [h, w] format.
        """
        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)  # 1/4
        height_ratio = float(feat_h / img_h)  # 1/4

        center_heatmap_target = gt_bboxes[-1].new_zeros(
            [bs, self.num_classes, feat_h, feat_w])
        # 这里的 gt_box 应该是 intersection 之后的 box, 而且box也是随着image augment一起变化的
        for batch_id in range(bs):
            gt_bbox = gt_bboxes[batch_id]
            gt_label = gt_labels[batch_id]
            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2
            gt_centers = torch.cat((center_x, center_y), dim=1)

            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.int()  # 用 int point 去产生高斯分布
                ctx, cty = ct
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio
                radius = gaussian_radius([scale_box_h, scale_box_w],
                                         min_overlap=0.3)
                radius = max(0, int(radius))
                ind = gt_label[j]
                gen_gaussian_target(center_heatmap_target[batch_id, ind],
                                    [ctx_int, cty_int], radius)

        ct_num = [center2d.shape[0] for center2d in centers2d]
        max_objs = max(ct_num)

        indexs = torch.zeros([bs, ct_num],
                             dtype=torch.uint8).to(centers2d[0].device)

        batch_centers2d = centers2d[0].new_zeros((bs, max_objs, 2))
        batch_labels_3d = gt_labels_3d[0].new_zeros((bs, max_objs))

        for i in range(bs):
            indexs[i, :ct_num[i]] = 1
            batch_centers2d[i, :ct_num[i]] = centers2d[i]
            batch_labels_3d[i, :ct_num[i]] = gt_labels_3d[i]

        indexs = indexs.flatten()  # ( batch * ct_num )
        batch_labels_3d = batch_labels_3d.flatten()
        batch_centers2d = batch_centers2d.view(-1, 2)

        gt_locations = torch.cat(
            [gt_bbox_3d.tensor[:, :3] for gt_bbox_3d in gt_bboxes_3d])
        gt_dimensions = torch.cat(
            [gt_bbox_3d.tensor[:, 3:6] for gt_bbox_3d in gt_bboxes_3d])
        gt_orientations = torch.cat(
            [gt_bbox_3d.tensor[:, 6] for gt_bbox_3d in gt_bboxes_3d])
        gt_corners = torch.cat(
            [gt_bbox_3d.corners() for gt_bbox_3d in gt_bboxes_3d])

        target_labels = dict(
            gt_centers2d=batch_centers2d,
            gt_labels3d=batch_labels_3d,
            indexs=indexs,
            gt_locs=gt_locations,
            gt_dims=gt_dimensions,
            gt_royts=gt_orientations,
            gt_cors=gt_corners)

        return center_heatmap_target, target_labels

    def loss(
            self,
            cls_scores,
            bbox_preds,
            gt_bboxes,
            gt_labels,
            gt_bboxes_3d,
            gt_labels_3d,
            centers2d,
            depths,
            attr_labels,  # None
            img_metas,
            gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box dims is a 4D-tensor,
                the channel number is bbox_code_size.  [B, 8, H, W]
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_3d (list[Tensor]): 3D boxes ground truth with shape of
                (num_gts, code_size).  it is the flipped gt_bboxes
            gt_labels_3d (list[Tensor]): same as gt_labels
            centers2d (list[Tensor]): 2D centers on the image with shape of
                (num_gts, 2).
            depths (list[Tensor]): Depth ground truth with shape of
                (num_gts, ).
            attr_labels (list[Tensor]): Attributes indices of each box.
                In kitti its None
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == 1
        assert attr_labels is None
        center2d_heatmap = cls_scores[0]  # （ B, C, H, W )
        pred_reg = bbox_preds[0]

        img_h, img_w = img_metas[0]['pad_shape']
        bs, _, feat_h, feat_w = center2d_heatmap.shape
        ratio = float(feat_w / img_w)
        # (B, C, H, W)

        center2d_heatmap_target, target_labels = \
            self.get_targets(gt_bboxes, gt_labels, gt_bboxes_3d,
                             gt_labels_3d, centers2d, depths,
                             center2d_heatmap.shape, img_metas[0]['pad_shape'])
        # TO DO: gt_labels2d 转化
        # CameraInstance3D
        pred_bboxes = self.get_predictions(
            labels3d=target_labels['gt_labels3d'],
            centers2d=target_labels['gt_centers2d'],
            gt_locations=target_labels['gt_locs'],
            gt_dimensions=target_labels['gt_dims'],
            gt_orientations=target_labels['gt_rotys'],
            img_metas=img_metas,
            pred_reg=pred_reg,
            ratio=ratio)

        loss_cls = self.loss_cls(center2d_heatmap, center2d_heatmap_target)

        loss_bbox_oris = self.loss_bbox(pred_bboxes['ori'].corners(),
                                        target_labels['gt_cors'])

        loss_bbox_dims = self.loss_bbox(pred_bboxes['dims'].corners(),
                                        target_labels['gt_cors'])

        loss_bbox_locs = self.loss_bbox(pred_bboxes['locs'].corners(),
                                        target_labels['gt_cors'])

        loss_bbox = loss_bbox_dims + loss_bbox_locs + loss_bbox_oris

        loss_dict = dict(loss_cls=loss_cls, loss_bbox=loss_bbox)

        return loss_dict
