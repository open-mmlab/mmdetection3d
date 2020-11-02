import torch
from mmcv.cnn import ConvModule
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.models.builder import build_loss
from mmdet3d.models.model_utils import VoteModule
from mmdet3d.ops import build_sa_module, furthest_point_sample
from mmdet.core import multi_apply
from mmdet.models import HEADS


@HEADS.register_module()
class PrimitiveHead(nn.Module):
    r"""Primitive head of `H3DNet <https://arxiv.org/abs/2006.05682>`_.

    Args:
        num_dims (int): The dimension of primitive semantic information.
        num_classes (int): The number of class.
        primitive_mode (str): The mode of primitive module,
            avaliable mode ['z', 'xy', 'line'].
        bbox_coder (:obj:`BaseBBoxCoder`): Bbox coder for encoding and
            decoding boxes.
        train_cfg (dict): Config for training.
        test_cfg (dict): Config for testing.
        vote_module_cfg (dict): Config of VoteModule for point-wise votes.
        vote_aggregation_cfg (dict): Config of vote aggregation layer.
        feat_channels (tuple[int]): Convolution channels of
            prediction layer.
        upper_thresh (float): Threshold for line matching.
        surface_thresh (float): Threshold for suface matching.
        conv_cfg (dict): Config of convolution in prediction layer.
        norm_cfg (dict): Config of BN in prediction layer.
        objectness_loss (dict): Config of objectness loss.
        center_loss (dict): Config of center loss.
        semantic_loss (dict): Config of point-wise semantic segmentation loss.
    """

    def __init__(self,
                 num_dims,
                 num_classes,
                 primitive_mode,
                 train_cfg=None,
                 test_cfg=None,
                 vote_module_cfg=None,
                 vote_aggregation_cfg=None,
                 feat_channels=(128, 128),
                 upper_thresh=100.0,
                 surface_thresh=0.5,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 objectness_loss=None,
                 center_loss=None,
                 semantic_reg_loss=None,
                 semantic_cls_loss=None):
        super(PrimitiveHead, self).__init__()
        assert primitive_mode in ['z', 'xy', 'line']
        # The dimension of primitive semantic information.
        self.num_dims = num_dims
        self.num_classes = num_classes
        self.primitive_mode = primitive_mode
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.gt_per_seed = vote_module_cfg['gt_per_seed']
        self.num_proposal = vote_aggregation_cfg['num_point']
        self.upper_thresh = upper_thresh
        self.surface_thresh = surface_thresh

        self.objectness_loss = build_loss(objectness_loss)
        self.center_loss = build_loss(center_loss)
        self.semantic_reg_loss = build_loss(semantic_reg_loss)
        self.semantic_cls_loss = build_loss(semantic_cls_loss)

        assert vote_aggregation_cfg['mlp_channels'][0] == vote_module_cfg[
            'in_channels']

        # Primitive existence flag prediction
        self.flag_conv = ConvModule(
            vote_module_cfg['conv_channels'][-1],
            vote_module_cfg['conv_channels'][-1] // 2,
            1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=True,
            inplace=True)
        self.flag_pred = torch.nn.Conv1d(
            vote_module_cfg['conv_channels'][-1] // 2, 2, 1)

        self.vote_module = VoteModule(**vote_module_cfg)
        self.vote_aggregation = build_sa_module(vote_aggregation_cfg)

        prev_channel = vote_aggregation_cfg['mlp_channels'][-1]
        conv_pred_list = list()
        for k in range(len(feat_channels)):
            conv_pred_list.append(
                ConvModule(
                    prev_channel,
                    feat_channels[k],
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    bias=True,
                    inplace=True))
            prev_channel = feat_channels[k]
        self.conv_pred = nn.Sequential(*conv_pred_list)

        conv_out_channel = 3 + num_dims + num_classes
        self.conv_pred.add_module('conv_out',
                                  nn.Conv1d(prev_channel, conv_out_channel, 1))

    def init_weights(self):
        """Initialize weights of VoteHead."""
        pass

    def forward(self, feats_dict, sample_mod):
        """Forward pass.

        Args:
            feats_dict (dict): Feature dict from backbone.
            sample_mod (str): Sample mode for vote aggregation layer.
                valid modes are "vote", "seed" and "random".

        Returns:
            dict: Predictions of primitive head.
        """
        assert sample_mod in ['vote', 'seed', 'random']

        seed_points = feats_dict['fp_xyz_net0'][-1]
        seed_features = feats_dict['hd_feature']
        results = {}

        primitive_flag = self.flag_conv(seed_features)
        primitive_flag = self.flag_pred(primitive_flag)

        results['pred_flag_' + self.primitive_mode] = primitive_flag

        # 1. generate vote_points from seed_points
        vote_points, vote_features, _ = self.vote_module(
            seed_points, seed_features)
        results['vote_' + self.primitive_mode] = vote_points
        results['vote_features_' + self.primitive_mode] = vote_features

        # 2. aggregate vote_points
        if sample_mod == 'vote':
            # use fps in vote_aggregation
            sample_indices = None
        elif sample_mod == 'seed':
            # FPS on seed and choose the votes corresponding to the seeds
            sample_indices = furthest_point_sample(seed_points,
                                                   self.num_proposal)
        elif sample_mod == 'random':
            # Random sampling from the votes
            batch_size, num_seed = seed_points.shape[:2]
            sample_indices = torch.randint(
                0,
                num_seed, (batch_size, self.num_proposal),
                dtype=torch.int32,
                device=seed_points.device)
        else:
            raise NotImplementedError('Unsupported sample mod!')

        vote_aggregation_ret = self.vote_aggregation(vote_points,
                                                     vote_features,
                                                     sample_indices)
        aggregated_points, features, aggregated_indices = vote_aggregation_ret
        results['aggregated_points_' + self.primitive_mode] = aggregated_points
        results['aggregated_features_' + self.primitive_mode] = features
        results['aggregated_indices_' +
                self.primitive_mode] = aggregated_indices

        # 3. predict primitive offsets and semantic information
        predictions = self.conv_pred(features)

        # 4. decode predictions
        decode_ret = self.primitive_decode_scores(predictions,
                                                  aggregated_points)
        results.update(decode_ret)

        center, pred_ind = self.get_primitive_center(
            primitive_flag, decode_ret['center_' + self.primitive_mode])

        results['pred_' + self.primitive_mode + '_ind'] = pred_ind
        results['pred_' + self.primitive_mode + '_center'] = center
        return results

    def loss(self,
             bbox_preds,
             points,
             gt_bboxes_3d,
             gt_labels_3d,
             pts_semantic_mask=None,
             pts_instance_mask=None,
             img_metas=None,
             gt_bboxes_ignore=None):
        """Compute loss.

        Args:
            bbox_preds (dict): Predictions from forward of primitive head.
            points (list[torch.Tensor]): Input points.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth \
                bboxes of each sample.
            gt_labels_3d (list[torch.Tensor]): Labels of each sample.
            pts_semantic_mask (None | list[torch.Tensor]): Point-wise
                semantic mask.
            pts_instance_mask (None | list[torch.Tensor]): Point-wise
                instance mask.
            img_metas (list[dict]): Contain pcd and img's meta info.
            gt_bboxes_ignore (None | list[torch.Tensor]): Specify
                which bounding.

        Returns:
            dict: Losses of Primitive Head.
        """
        targets = self.get_targets(points, gt_bboxes_3d, gt_labels_3d,
                                   pts_semantic_mask, pts_instance_mask,
                                   bbox_preds)

        (point_mask, point_offset, gt_primitive_center, gt_primitive_semantic,
         gt_sem_cls_label, gt_primitive_mask) = targets

        losses = {}
        # Compute the loss of primitive existence flag
        pred_flag = bbox_preds['pred_flag_' + self.primitive_mode]
        flag_loss = self.objectness_loss(pred_flag, gt_primitive_mask.long())
        losses['flag_loss_' + self.primitive_mode] = flag_loss

        # calculate vote loss
        vote_loss = self.vote_module.get_loss(
            bbox_preds['seed_points'],
            bbox_preds['vote_' + self.primitive_mode],
            bbox_preds['seed_indices'], point_mask, point_offset)
        losses['vote_loss_' + self.primitive_mode] = vote_loss

        num_proposal = bbox_preds['aggregated_points_' +
                                  self.primitive_mode].shape[1]
        primitive_center = bbox_preds['center_' + self.primitive_mode]
        if self.primitive_mode != 'line':
            primitive_semantic = bbox_preds['size_residuals_' +
                                            self.primitive_mode].contiguous()
        else:
            primitive_semantic = None
        semancitc_scores = bbox_preds['sem_cls_scores_' +
                                      self.primitive_mode].transpose(2, 1)

        gt_primitive_mask = gt_primitive_mask / \
            (gt_primitive_mask.sum() + 1e-6)
        center_loss, size_loss, sem_cls_loss = self.compute_primitive_loss(
            primitive_center, primitive_semantic, semancitc_scores,
            num_proposal, gt_primitive_center, gt_primitive_semantic,
            gt_sem_cls_label, gt_primitive_mask)
        losses['center_loss_' + self.primitive_mode] = center_loss
        losses['size_loss_' + self.primitive_mode] = size_loss
        losses['sem_loss_' + self.primitive_mode] = sem_cls_loss

        return losses

    def get_targets(self,
                    points,
                    gt_bboxes_3d,
                    gt_labels_3d,
                    pts_semantic_mask=None,
                    pts_instance_mask=None,
                    bbox_preds=None):
        """Generate targets of primitive head.

        Args:
            points (list[torch.Tensor]): Points of each batch.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth \
                bboxes of each batch.
            gt_labels_3d (list[torch.Tensor]): Labels of each batch.
            pts_semantic_mask (None | list[torch.Tensor]): Point-wise semantic
                label of each batch.
            pts_instance_mask (None | list[torch.Tensor]): Point-wise instance
                label of each batch.
            bbox_preds (dict): Predictions from forward of primitive head.

        Returns:
            tuple[torch.Tensor]: Targets of primitive head.
        """
        for index in range(len(gt_labels_3d)):
            if len(gt_labels_3d[index]) == 0:
                fake_box = gt_bboxes_3d[index].tensor.new_zeros(
                    1, gt_bboxes_3d[index].tensor.shape[-1])
                gt_bboxes_3d[index] = gt_bboxes_3d[index].new_box(fake_box)
                gt_labels_3d[index] = gt_labels_3d[index].new_zeros(1)

        if pts_semantic_mask is None:
            pts_semantic_mask = [None for i in range(len(gt_labels_3d))]
            pts_instance_mask = [None for i in range(len(gt_labels_3d))]

        (point_mask, point_sem,
         point_offset) = multi_apply(self.get_targets_single, points,
                                     gt_bboxes_3d, gt_labels_3d,
                                     pts_semantic_mask, pts_instance_mask)

        point_mask = torch.stack(point_mask)
        point_sem = torch.stack(point_sem)
        point_offset = torch.stack(point_offset)

        batch_size = point_mask.shape[0]
        num_proposal = bbox_preds['aggregated_points_' +
                                  self.primitive_mode].shape[1]
        num_seed = bbox_preds['seed_points'].shape[1]
        seed_inds = bbox_preds['seed_indices'].long()
        seed_inds_expand = seed_inds.view(batch_size, num_seed,
                                          1).repeat(1, 1, 3)
        seed_gt_votes = torch.gather(point_offset, 1, seed_inds_expand)
        seed_gt_votes += bbox_preds['seed_points']
        gt_primitive_center = seed_gt_votes.view(batch_size * num_proposal, 1,
                                                 3)

        seed_inds_expand_sem = seed_inds.view(batch_size, num_seed, 1).repeat(
            1, 1, 4 + self.num_dims)
        seed_gt_sem = torch.gather(point_sem, 1, seed_inds_expand_sem)
        gt_primitive_semantic = seed_gt_sem[:, :, 3:3 + self.num_dims].view(
            batch_size * num_proposal, 1, self.num_dims).contiguous()

        gt_sem_cls_label = seed_gt_sem[:, :, -1].long()

        gt_votes_mask = torch.gather(point_mask, 1, seed_inds)

        return (point_mask, point_offset, gt_primitive_center,
                gt_primitive_semantic, gt_sem_cls_label, gt_votes_mask)

    def get_targets_single(self,
                           points,
                           gt_bboxes_3d,
                           gt_labels_3d,
                           pts_semantic_mask=None,
                           pts_instance_mask=None):
        """Generate targets of primitive head for single batch.

        Args:
            points (torch.Tensor): Points of each batch.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): Ground truth \
                boxes of each batch.
            gt_labels_3d (torch.Tensor): Labels of each batch.
            pts_semantic_mask (None | torch.Tensor): Point-wise semantic
                label of each batch.
            pts_instance_mask (None | torch.Tensor): Point-wise instance
                label of each batch.

        Returns:
            tuple[torch.Tensor]: Targets of primitive head.
        """
        gt_bboxes_3d = gt_bboxes_3d.to(points.device)
        num_points = points.shape[0]

        point_mask = points.new_zeros(num_points)
        # Offset to the primitive center
        point_offset = points.new_zeros([num_points, 3])
        # Semantic information of primitive center
        point_sem = points.new_zeros([num_points, 3 + self.num_dims + 1])

        # Generate pts_semantic_mask and pts_instance_mask when they are None
        if pts_semantic_mask is None or pts_instance_mask is None:
            points2box_mask = gt_bboxes_3d.points_in_boxes(points)
            assignment = points2box_mask.argmax(1)
            background_mask = points2box_mask.max(1)[0] == 0

            if pts_semantic_mask is None:
                pts_semantic_mask = gt_labels_3d[assignment]
                pts_semantic_mask[background_mask] = self.num_classes

            if pts_instance_mask is None:
                pts_instance_mask = assignment
                pts_instance_mask[background_mask] = gt_labels_3d.shape[0]

        instance_flag = torch.nonzero(
            pts_semantic_mask != self.num_classes).squeeze(1)
        instance_labels = pts_instance_mask[instance_flag].unique()

        with_yaw = gt_bboxes_3d.with_yaw
        for i, i_instance in enumerate(instance_labels):
            indices = instance_flag[pts_instance_mask[instance_flag] ==
                                    i_instance]
            coords = points[indices, :3]
            cur_cls_label = pts_semantic_mask[indices][0]

            # Bbox Corners
            cur_corners = gt_bboxes_3d.corners[i]

            plane_lower_temp = points.new_tensor(
                [0, 0, 1, -cur_corners[7, -1]])
            upper_points = cur_corners[[1, 2, 5, 6]]
            refined_distance = (upper_points * plane_lower_temp[:3]).sum(dim=1)

            if self.check_horizon(upper_points) and \
                    plane_lower_temp[0] + plane_lower_temp[1] < \
                    self.train_cfg['lower_thresh']:
                plane_lower = points.new_tensor(
                    [0, 0, 1, plane_lower_temp[-1]])
                plane_upper = points.new_tensor(
                    [0, 0, 1, -torch.mean(refined_distance)])
            else:
                raise NotImplementedError('Only horizontal plane is support!')

            if self.check_dist(plane_upper, upper_points) is False:
                raise NotImplementedError(
                    'Mean distance to plane should be lower than thresh!')

            # Get the boundary points here
            point2plane_dist, selected = self.match_point2plane(
                plane_lower, coords)

            # Get bottom four lines
            if self.primitive_mode == 'line':
                point2line_matching = self.match_point2line(
                    coords[selected], cur_corners, with_yaw, mode='bottom')

                point_mask, point_offset, point_sem = \
                    self._assign_primitive_line_targets(point_mask,
                                                        point_offset,
                                                        point_sem,
                                                        coords[selected],
                                                        indices[selected],
                                                        cur_cls_label,
                                                        point2line_matching,
                                                        cur_corners,
                                                        [1, 1, 0, 0],
                                                        with_yaw,
                                                        mode='bottom')

            # Set the surface labels here
            if self.primitive_mode == 'z' and \
                    selected.sum() > self.train_cfg['num_point'] and \
                    point2plane_dist[selected].var() < \
                    self.train_cfg['var_thresh']:

                point_mask, point_offset, point_sem = \
                    self._assign_primitive_surface_targets(point_mask,
                                                           point_offset,
                                                           point_sem,
                                                           coords[selected],
                                                           indices[selected],
                                                           cur_cls_label,
                                                           cur_corners,
                                                           with_yaw,
                                                           mode='bottom')

            # Get the boundary points here
            point2plane_dist, selected = self.match_point2plane(
                plane_upper, coords)

            # Get top four lines
            if self.primitive_mode == 'line':
                point2line_matching = self.match_point2line(
                    coords[selected], cur_corners, with_yaw, mode='top')

                point_mask, point_offset, point_sem = \
                    self._assign_primitive_line_targets(point_mask,
                                                        point_offset,
                                                        point_sem,
                                                        coords[selected],
                                                        indices[selected],
                                                        cur_cls_label,
                                                        point2line_matching,
                                                        cur_corners,
                                                        [1, 1, 0, 0],
                                                        with_yaw,
                                                        mode='top')

            if self.primitive_mode == 'z' and \
                    selected.sum() > self.train_cfg['num_point'] and \
                    point2plane_dist[selected].var() < \
                    self.train_cfg['var_thresh']:

                point_mask, point_offset, point_sem = \
                    self._assign_primitive_surface_targets(point_mask,
                                                           point_offset,
                                                           point_sem,
                                                           coords[selected],
                                                           indices[selected],
                                                           cur_cls_label,
                                                           cur_corners,
                                                           with_yaw,
                                                           mode='top')

            # Get left two lines
            plane_left_temp = self._get_plane_fomulation(
                cur_corners[2] - cur_corners[3],
                cur_corners[3] - cur_corners[0], cur_corners[0])

            right_points = cur_corners[[4, 5, 7, 6]]
            plane_left_temp /= torch.norm(plane_left_temp[:3])
            refined_distance = (right_points * plane_left_temp[:3]).sum(dim=1)

            if plane_left_temp[2] < self.train_cfg['lower_thresh']:
                plane_left = plane_left_temp
                plane_right = points.new_tensor([
                    plane_left_temp[0], plane_left_temp[1], plane_left_temp[2],
                    -refined_distance.mean()
                ])
            else:
                raise NotImplementedError(
                    'Normal vector of the plane should be horizontal!')

            # Get the boundary points here
            point2plane_dist, selected = self.match_point2plane(
                plane_left, coords)

            # Get left four lines
            if self.primitive_mode == 'line':
                point2line_matching = self.match_point2line(
                    coords[selected], cur_corners, with_yaw, mode='left')
                point_mask, point_offset, point_sem = \
                    self._assign_primitive_line_targets(
                        point_mask, point_offset, point_sem,
                        coords[selected], indices[selected], cur_cls_label,
                        point2line_matching[2:], cur_corners, [2, 2],
                        with_yaw, mode='left')

            if self.primitive_mode == 'xy' and \
                    selected.sum() > self.train_cfg['num_point'] and \
                    point2plane_dist[selected].var() < \
                    self.train_cfg['var_thresh']:

                point_mask, point_offset, point_sem = \
                    self._assign_primitive_surface_targets(
                        point_mask, point_offset, point_sem,
                        coords[selected], indices[selected], cur_cls_label,
                        cur_corners, with_yaw, mode='left')

            # Get the boundary points here
            point2plane_dist, selected = self.match_point2plane(
                plane_right, coords)

            # Get right four lines
            if self.primitive_mode == 'line':
                point2line_matching = self.match_point2line(
                    coords[selected], cur_corners, with_yaw, mode='right')

                point_mask, point_offset, point_sem = \
                    self._assign_primitive_line_targets(
                        point_mask, point_offset, point_sem,
                        coords[selected], indices[selected], cur_cls_label,
                        point2line_matching[2:], cur_corners, [2, 2],
                        with_yaw, mode='right')

            if self.primitive_mode == 'xy' and \
                    selected.sum() > self.train_cfg['num_point'] and \
                    point2plane_dist[selected].var() < \
                    self.train_cfg['var_thresh']:

                point_mask, point_offset, point_sem = \
                    self._assign_primitive_surface_targets(
                        point_mask, point_offset, point_sem,
                        coords[selected], indices[selected], cur_cls_label,
                        cur_corners, with_yaw, mode='right')

            plane_front_temp = self._get_plane_fomulation(
                cur_corners[0] - cur_corners[4],
                cur_corners[4] - cur_corners[5], cur_corners[5])

            back_points = cur_corners[[3, 2, 7, 6]]
            plane_front_temp /= torch.norm(plane_front_temp[:3])
            refined_distance = (back_points * plane_front_temp[:3]).sum(dim=1)

            if plane_front_temp[2] < self.train_cfg['lower_thresh']:
                plane_front = plane_front_temp
                plane_back = points.new_tensor([
                    plane_front_temp[0], plane_front_temp[1],
                    plane_front_temp[2], -torch.mean(refined_distance)
                ])
            else:
                raise NotImplementedError(
                    'Normal vector of the plane should be horizontal!')

            # Get the boundary points here
            point2plane_dist, selected = self.match_point2plane(
                plane_front, coords)

            if self.primitive_mode == 'xy' and \
                    selected.sum() > self.train_cfg['num_point'] and \
                    (point2plane_dist[selected]).var() < \
                    self.train_cfg['var_thresh']:

                point_mask, point_offset, point_sem = \
                    self._assign_primitive_surface_targets(
                        point_mask, point_offset, point_sem,
                        coords[selected], indices[selected], cur_cls_label,
                        cur_corners, with_yaw, mode='front')

            # Get the boundary points here
            point2plane_dist, selected = self.match_point2plane(
                plane_back, coords)

            if self.primitive_mode == 'xy' and \
                    selected.sum() > self.train_cfg['num_point'] and \
                    point2plane_dist[selected].var() < \
                    self.train_cfg['var_thresh']:

                point_mask, point_offset, point_sem = \
                    self._assign_primitive_surface_targets(
                        point_mask, point_offset, point_sem,
                        coords[selected], indices[selected], cur_cls_label,
                        cur_corners, with_yaw, mode='back')

        return (point_mask, point_sem, point_offset)

    def primitive_decode_scores(self, predictions, aggregated_points):
        """Decode predicted parts to primitive head.

        Args:
            predictions (torch.Tensor): primitive pridictions of each batch.
            aggregated_points (torch.Tensor): The aggregated points
                of vote stage.

        Returns:
            Dict: Predictions of primitive head, including center,
                semantic size and semantic scores.
        """

        ret_dict = {}
        pred_transposed = predictions.transpose(2, 1)

        center = aggregated_points + pred_transposed[:, :, 0:3]
        ret_dict['center_' + self.primitive_mode] = center

        if self.primitive_mode in ['z', 'xy']:
            ret_dict['size_residuals_' + self.primitive_mode] = \
                pred_transposed[:, :, 3:3 + self.num_dims]

        ret_dict['sem_cls_scores_' + self.primitive_mode] = \
            pred_transposed[:, :, 3 + self.num_dims:]

        return ret_dict

    def check_horizon(self, points):
        """Check whether is a horizontal plane.

        Args:
            points (torch.Tensor): Points of input.

        Returns:
            Bool: Flag of result.
        """
        return (points[0][-1] == points[1][-1]) and \
               (points[1][-1] == points[2][-1]) and \
               (points[2][-1] == points[3][-1])

    def check_dist(self, plane_equ, points):
        """Whether the mean of points to plane distance is lower than thresh.

        Args:
            plane_equ (torch.Tensor): Plane to be checked.
            points (torch.Tensor): Points to be checked.

        Returns:
            Tuple: Flag of result.
        """
        return (points[:, 2] +
                plane_equ[-1]).sum() / 4.0 < self.train_cfg['lower_thresh']

    def point2line_dist(self, points, pts_a, pts_b):
        """Calculate the distance from point to line.

        Args:
            points (torch.Tensor): Points of input.
            pts_a (torch.Tensor): Point on the specific line.
            pts_b (torch.Tensor): Point on the specific line.

        Returns:
            torch.Tensor: Distance between each point to line.
        """
        line_a2b = pts_b - pts_a
        line_a2pts = points - pts_a
        length = (line_a2pts * line_a2b.view(1, 3)).sum(1) / \
            line_a2b.norm()
        dist = (line_a2pts.norm(dim=1)**2 - length**2).sqrt()

        return dist

    def match_point2line(self, points, corners, with_yaw, mode='bottom'):
        """Match points to corresponding line.

        Args:
            points (torch.Tensor): Points of input.
            corners (torch.Tensor): Eight corners of a bounding box.
            with_yaw (Bool): Whether the boundind box is with rotation.
            mode (str, optional): Specify which line should be matched,
                available mode are ('bottom', 'top', 'left', 'right').
                Defaults to 'bottom'.

        Returns:
            Tuple: Flag of matching correspondence.
        """
        if with_yaw:
            corners_pair = {
                'bottom': [[0, 3], [4, 7], [0, 4], [3, 7]],
                'top': [[1, 2], [5, 6], [1, 5], [2, 6]],
                'left': [[0, 1], [3, 2], [0, 1], [3, 2]],
                'right': [[4, 5], [7, 6], [4, 5], [7, 6]]
            }
            selected_list = []
            for pair_index in corners_pair[mode]:
                selected = self.point2line_dist(
                    points, corners[pair_index[0]], corners[pair_index[1]]) \
                    < self.train_cfg['line_thresh']
                selected_list.append(selected)
        else:
            xmin, ymin, _ = corners.min(0)[0]
            xmax, ymax, _ = corners.max(0)[0]
            sel1 = torch.abs(points[:, 0] -
                             xmin) < self.train_cfg['line_thresh']
            sel2 = torch.abs(points[:, 0] -
                             xmax) < self.train_cfg['line_thresh']
            sel3 = torch.abs(points[:, 1] -
                             ymin) < self.train_cfg['line_thresh']
            sel4 = torch.abs(points[:, 1] -
                             ymax) < self.train_cfg['line_thresh']
            selected_list = [sel1, sel2, sel3, sel4]
        return selected_list

    def match_point2plane(self, plane, points):
        """Match points to plane.

        Args:
            plane (torch.Tensor): Equation of the plane.
            points (torch.Tensor): Points of input.

        Returns:
            Tuple: Distance of each point to the plane and
                flag of matching correspondence.
        """
        point2plane_dist = torch.abs((points * plane[:3]).sum(dim=1) +
                                     plane[-1])
        min_dist = point2plane_dist.min()
        selected = torch.abs(point2plane_dist -
                             min_dist) < self.train_cfg['dist_thresh']
        return point2plane_dist, selected

    def compute_primitive_loss(self, primitive_center, primitive_semantic,
                               semantic_scores, num_proposal,
                               gt_primitive_center, gt_primitive_semantic,
                               gt_sem_cls_label, gt_primitive_mask):
        """Compute loss of primitive module.

        Args:
            primitive_center (torch.Tensor): Pridictions of primitive center.
            primitive_semantic (torch.Tensor): Pridictions of primitive
                semantic.
            semantic_scores (torch.Tensor): Pridictions of primitive
                semantic scores.
            num_proposal (int): The number of primitive proposal.
            gt_primitive_center (torch.Tensor): Ground truth of
                primitive center.
            gt_votes_sem (torch.Tensor): Ground truth of primitive semantic.
            gt_sem_cls_label (torch.Tensor): Ground truth of primitive
                semantic class.
            gt_primitive_mask (torch.Tensor): Ground truth of primitive mask.

        Returns:
            Tuple: Loss of primitive module.
        """
        batch_size = primitive_center.shape[0]
        vote_xyz_reshape = primitive_center.view(batch_size * num_proposal, -1,
                                                 3)

        center_loss = self.center_loss(
            vote_xyz_reshape,
            gt_primitive_center,
            dst_weight=gt_primitive_mask.view(batch_size * num_proposal, 1))[1]

        if self.primitive_mode != 'line':
            size_xyz_reshape = primitive_semantic.view(
                batch_size * num_proposal, -1, self.num_dims).contiguous()
            size_loss = self.semantic_reg_loss(
                size_xyz_reshape,
                gt_primitive_semantic,
                dst_weight=gt_primitive_mask.view(batch_size * num_proposal,
                                                  1))[1]
        else:
            size_loss = center_loss.new_tensor(0.0)

        # Semantic cls loss
        sem_cls_loss = self.semantic_cls_loss(
            semantic_scores, gt_sem_cls_label, weight=gt_primitive_mask)

        return center_loss, size_loss, sem_cls_loss

    def get_primitive_center(self, pred_flag, center):
        """Generate primitive center from predictions.

        Args:
            pred_flag (torch.Tensor): Scores of primitive center.
            center (torch.Tensor): Pridictions of primitive center.

        Returns:
            Tuple: Primitive center and the prediction indices.
        """
        ind_normal = F.softmax(pred_flag, dim=1)
        pred_indices = (ind_normal[:, 1, :] >
                        self.surface_thresh).detach().float()
        selected = (ind_normal[:, 1, :] <=
                    self.surface_thresh).detach().float()
        offset = torch.ones_like(center) * self.upper_thresh
        center = center + offset * selected.unsqueeze(-1)
        return center, pred_indices

    def _assign_primitive_line_targets(self,
                                       point_mask,
                                       point_offset,
                                       point_sem,
                                       coords,
                                       indices,
                                       cls_label,
                                       point2line_matching,
                                       corners,
                                       center_axises,
                                       with_yaw,
                                       mode='bottom'):
        """Generate targets of line primitive.

        Args:
            point_mask (torch.Tensor): Tensor to store the ground
                truth of mask.
            point_offset (torch.Tensor): Tensor to store the ground
                truth of offset.
            point_sem (torch.Tensor): Tensor to store the ground
                truth of semantic.
            coords (torch.Tensor): The selected points.
            indices (torch.Tensor): Indices of the selected points.
            cls_label (int): Class label of the ground truth bounding box.
            point2line_matching (torch.Tensor): Flag indicate that
                matching line of each point.
            corners (torch.Tensor): Corners of the ground truth bounding box.
            center_axises (list[int]): Indicate in which axis the line center
                should be refined.
            with_yaw (Bool): Whether the boundind box is with rotation.
            mode (str, optional): Specify which line should be matched,
                available mode are ('bottom', 'top', 'left', 'right').
                Defaults to 'bottom'.

        Returns:
            Tuple: Targets of the line primitive.
        """
        corners_pair = {
            'bottom': [[0, 3], [4, 7], [0, 4], [3, 7]],
            'top': [[1, 2], [5, 6], [1, 5], [2, 6]],
            'left': [[0, 1], [3, 2]],
            'right': [[4, 5], [7, 6]]
        }
        corners_pair = corners_pair[mode]
        assert len(corners_pair) == len(point2line_matching) == len(
            center_axises)
        for line_select, center_axis, pair_index in zip(
                point2line_matching, center_axises, corners_pair):
            if line_select.sum() > self.train_cfg['num_point_line']:
                point_mask[indices[line_select]] = 1.0

                if with_yaw:
                    line_center = (corners[pair_index[0]] +
                                   corners[pair_index[1]]) / 2
                else:
                    line_center = coords[line_select].mean(dim=0)
                    line_center[center_axis] = corners[:, center_axis].mean()

                point_offset[indices[line_select]] = \
                    line_center - coords[line_select]
                point_sem[indices[line_select]] = \
                    point_sem.new_tensor([line_center[0], line_center[1],
                                          line_center[2], cls_label])
        return point_mask, point_offset, point_sem

    def _assign_primitive_surface_targets(self,
                                          point_mask,
                                          point_offset,
                                          point_sem,
                                          coords,
                                          indices,
                                          cls_label,
                                          corners,
                                          with_yaw,
                                          mode='bottom'):
        """Generate targets for primitive z and primitive xy.

        Args:
            point_mask (torch.Tensor): Tensor to store the ground
                truth of mask.
            point_offset (torch.Tensor): Tensor to store the ground
                truth of offset.
            point_sem (torch.Tensor): Tensor to store the ground
                truth of semantic.
            coords (torch.Tensor): The selected points.
            indices (torch.Tensor): Indices of the selected points.
            cls_label (int): Class label of the ground truth bounding box.
            corners (torch.Tensor): Corners of the ground truth bounding box.
            with_yaw (Bool): Whether the boundind box is with rotation.
            mode (str, optional): Specify which line should be matched,
                available mode are ('bottom', 'top', 'left', 'right',
                'front', 'back').
                Defaults to 'bottom'.

        Returns:
            Tuple: Targets of the center primitive.
        """
        point_mask[indices] = 1.0
        corners_pair = {
            'bottom': [0, 7],
            'top': [1, 6],
            'left': [0, 1],
            'right': [4, 5],
            'front': [0, 1],
            'back': [3, 2]
        }
        pair_index = corners_pair[mode]
        if self.primitive_mode == 'z':
            if with_yaw:
                center = (corners[pair_index[0]] +
                          corners[pair_index[1]]) / 2.0
                center[2] = coords[:, 2].mean()
                point_sem[indices] = point_sem.new_tensor([
                    center[0], center[1],
                    center[2], (corners[4] - corners[0]).norm(),
                    (corners[3] - corners[0]).norm(), cls_label
                ])
            else:
                center = point_mask.new_tensor([
                    corners[:, 0].mean(), corners[:, 1].mean(),
                    coords[:, 2].mean()
                ])
                point_sem[indices] = point_sem.new_tensor([
                    center[0], center[1], center[2],
                    corners[:, 0].max() - corners[:, 0].min(),
                    corners[:, 1].max() - corners[:, 1].min(), cls_label
                ])
        elif self.primitive_mode == 'xy':
            if with_yaw:
                center = coords.mean(0)
                center[2] = (corners[pair_index[0], 2] +
                             corners[pair_index[1], 2]) / 2.0
                point_sem[indices] = point_sem.new_tensor([
                    center[0], center[1], center[2],
                    corners[pair_index[1], 2] - corners[pair_index[0], 2],
                    cls_label
                ])
            else:
                center = point_mask.new_tensor([
                    coords[:, 0].mean(), coords[:, 1].mean(),
                    corners[:, 2].mean()
                ])
                point_sem[indices] = point_sem.new_tensor([
                    center[0], center[1], center[2],
                    corners[:, 2].max() - corners[:, 2].min(), cls_label
                ])
        point_offset[indices] = center - coords
        return point_mask, point_offset, point_sem

    def _get_plane_fomulation(self, vector1, vector2, point):
        """Compute the equation of the plane.

        Args:
            vector1 (torch.Tensor): Parallel vector of the plane.
            vector2 (torch.Tensor): Parallel vector of the plane.
            point (torch.Tensor): Point on the plane.

        Returns:
            torch.Tensor: Equation of the plane.
        """
        surface_norm = torch.cross(vector1, vector2)
        surface_dis = -torch.dot(surface_norm, point)
        plane = point.new_tensor(
            [surface_norm[0], surface_norm[1], surface_norm[2], surface_dis])
        return plane
