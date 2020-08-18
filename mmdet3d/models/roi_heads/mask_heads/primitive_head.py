import torch
from mmcv.cnn import ConvModule
from torch import nn as nn

from mmdet3d.models.builder import build_loss
from mmdet3d.models.model_utils import VoteModule
from mmdet3d.ops import PointSAModule, furthest_point_sample
from mmdet.core import multi_apply
from mmdet.models import HEADS


@HEADS.register_module()
class PrimitiveHead(nn.Module):
    r"""Bbox head of `H3dnet <https://arxiv.org/abs/2006.05682>`_.

    Args:
        num_dim (int): The dimension of primitive semantic information.
        num_classes (int): The number of class.
        primitive_mode (str): The mode of primitive module,
            avaliable mode ['z', 'xy', 'line'].
        bbox_coder (:obj:`BaseBBoxCoder`): Bbox coder for encoding and
            decoding boxes.
        train_cfg (dict): Config for training.
        test_cfg (dict): Config for testing.
        vote_moudule_cfg (dict): Config of VoteModule for point-wise votes.
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
                 num_dim,
                 num_classes,
                 primitive_mode,
                 train_cfg=None,
                 test_cfg=None,
                 vote_moudule_cfg=None,
                 vote_aggregation_cfg=None,
                 feat_channels=(128, 128),
                 upper_thresh=100.0,
                 surface_thresh=0.5,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 objectness_loss=None,
                 center_loss=None,
                 semantic_loss=None):
        super(PrimitiveHead, self).__init__()
        assert primitive_mode in ['z', 'xy', 'line']
        # The dimension of primitive semantic information.
        self.num_dim = num_dim
        self.num_classes = num_classes
        self.primitive_mode = primitive_mode
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.gt_per_seed = vote_moudule_cfg['gt_per_seed']
        self.num_proposal = vote_aggregation_cfg['num_point']
        self.upper_thresh = upper_thresh
        self.surface_thresh = surface_thresh

        self.objectness_loss = build_loss(objectness_loss)
        self.center_loss = build_loss(center_loss)
        self.semantic_loss = build_loss(semantic_loss)

        assert vote_aggregation_cfg['mlp_channels'][0] == vote_moudule_cfg[
            'in_channels']

        # Primitive existence flag prediction
        self.flag_conv = ConvModule(
            vote_moudule_cfg['conv_channels'][-1],
            vote_moudule_cfg['conv_channels'][-1] // 2,
            1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=True,
            inplace=True)
        self.flag_pred = torch.nn.Conv1d(
            vote_moudule_cfg['conv_channels'][-1] // 2, 2, 1)

        self.vote_module = VoteModule(**vote_moudule_cfg)
        self.vote_aggregation = PointSAModule(**vote_aggregation_cfg)

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

        conv_out_channel = 3 + num_dim + num_classes
        self.conv_pred.add_module('conv_out',
                                  nn.Conv1d(prev_channel, conv_out_channel, 1))

        self.softmax_normal = nn.Softmax(dim=1)

    def init_weights(self):
        """Initialize weights of VoteHead."""
        pass

    def forward(self, feat_dict, sample_mod):
        """Forward pass.

        Args:
            feat_dict (dict): Feature dict from backbone.
            sample_mod (str): Sample mode for vote aggregation layer.
                valid modes are "vote", "seed" and "random".

        Returns:
            dict: Predictions of primitive head.
        """
        assert sample_mod in ['vote', 'seed', 'random']

        seed_points = feat_dict['fp_xyz_net0'][-1]
        seed_features = feat_dict['hd_feature']
        results = {}

        primitive_flag = self.flag_conv(seed_features)
        primitive_flag = self.flag_pred(primitive_flag)

        results['pred_flag_' + self.primitive_mode] = primitive_flag

        # 1. generate vote_points from seed_points
        vote_points, vote_features = self.vote_module(seed_points,
                                                      seed_features)
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
            sample_indices = seed_points.new_tensor(
                torch.randint(0, num_seed, (batch_size, self.num_proposal)),
                dtype=torch.int32)
        else:
            raise NotImplementedError

        vote_aggregation_ret = self.vote_aggregation(vote_points,
                                                     vote_features,
                                                     sample_indices)
        aggregated_points, features, aggregated_indices = vote_aggregation_ret
        results['aggregated_points_' + self.primitive_mode] = aggregated_points
        results['aggregated_features_' + self.primitive_mode] = features
        results['aggregated_indices_' +
                self.primitive_mode] = aggregated_indices

        # 3. predict bbox and score
        predictions = self.conv_pred(features)

        # 4. decode predictions
        primitive_center, decode_res = self.primitive_decode_scores(
            predictions, aggregated_points, mode=self.primitive_mode)
        results.update(decode_res)

        center, pred_ind = self.get_primitive_center(primitive_flag,
                                                     primitive_center)
        # print(center.sum(), pred_ind.sum())
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
            dict: Losses of Votenet.
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
        center_loss, size_loss, sem_cls_loss = self.compute_primitivesem_loss(
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
        """Generate targets of vote head.

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
        valid_gt_masks = list()
        gt_num = list()
        for index in range(len(gt_labels_3d)):
            if len(gt_labels_3d[index]) == 0:
                fake_box = gt_bboxes_3d[index].tensor.new_zeros(
                    1, gt_bboxes_3d[index].tensor.shape[-1])
                gt_bboxes_3d[index] = gt_bboxes_3d[index].new_box(fake_box)
                gt_labels_3d[index] = gt_labels_3d[index].new_zeros(1)
                valid_gt_masks.append(gt_labels_3d[index].new_zeros(1))
                gt_num.append(1)
            else:
                valid_gt_masks.append(gt_labels_3d[index].new_ones(
                    gt_labels_3d[index].shape))
                gt_num.append(gt_labels_3d[index].shape[0])

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
            1, 1, 4 + self.num_dim)
        seed_gt_sem = torch.gather(point_sem, 1, seed_inds_expand_sem)
        gt_primitive_semantic = seed_gt_sem[:, :, 3:3 + self.num_dim].view(
            batch_size * num_proposal, 1, self.num_dim).contiguous()

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
        """Generate targets of vote head for single batch.

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
        point_offset = points.new_zeros([num_points, 3])
        point_sem = points.new_zeros([num_points, 3 + self.num_dim + 1])

        instance_flag = torch.nonzero(
            pts_semantic_mask != self.num_classes).squeeze(1)
        instance_labels = pts_instance_mask[instance_flag].unique()

        for i, i_instance in enumerate(instance_labels):
            ind = instance_flag[pts_instance_mask[instance_flag] == i_instance]
            x = points[ind, :3]

            # Corners
            corners = gt_bboxes_3d.corners[i][[0, 1, 3, 2, 4, 5, 7, 6]]
            xmin, ymin, zmin = corners.min(0)[0]
            xmax, ymax, zmax = corners.max(0)[0]

            plane_lower_temp = points.new_tensor([0, 0, 1, -corners[6, -1]])
            para_points = corners[[1, 3, 5, 7]]
            newd = torch.sum(para_points * plane_lower_temp[:3], 1)
            if self.check_upright(para_points) and \
                    plane_lower_temp[0] + plane_lower_temp[1] < \
                    self.train_cfg['lower_thresh']:
                plane_lower = points.new_tensor(
                    [0, 0, 1, plane_lower_temp[-1]])
                plane_upper = points.new_tensor([0, 0, 1, -torch.mean(newd)])
            else:
                raise NotImplementedError
                # print('error with upright')

            if self.check_z(plane_upper, para_points) is False:
                raise NotImplementedError

            # Get the boundary points here
            alldist = torch.abs(
                torch.sum(x * plane_lower[:3], 1) + plane_lower[-1])
            mind = alldist.min()
            sel = torch.abs(alldist - mind) < self.train_cfg['dist_thresh']

            # Get lower four lines
            if self.primitive_mode == 'line':
                point2_lines_matching = self.match_point2line(
                    x[sel], xmin, xmax, ymin, ymax)
                for idx, line_select in enumerate(point2_lines_matching):
                    if torch.sum(line_select) > \
                            self.train_cfg['num_point_line']:
                        point_mask[ind[sel][line_select]] = 1.0
                        line_center = torch.mean(x[sel][line_select], axis=0)
                        if idx < 2:
                            line_center[1] = (ymin + ymax) / 2.0
                        else:
                            line_center[0] = (xmin + xmax) / 2.0
                        point_offset[ind[sel][line_select]] = \
                            line_center - x[sel][line_select]
                        point_sem[ind[sel][line_select]] = \
                            points.new_tensor([line_center[0], line_center[1],
                                               line_center[2],
                                               pts_semantic_mask[ind][0]])

            # Set the surface labels here
            if self.primitive_mode == 'z':
                if torch.sum(sel) > self.train_cfg['num_point'] and torch.var(
                        alldist[sel]) < self.train_cfg['var_thresh']:
                    center = points.new_tensor([(xmin + xmax) / 2.0,
                                                (ymin + ymax) / 2.0,
                                                torch.mean(x[sel][:, 2])])
                    sel_global = ind[sel]
                    point_mask[sel_global] = 1.0
                    point_sem[sel_global] = points.new_tensor([
                        center[0], center[1], center[2], xmax - xmin,
                        ymax - ymin, (pts_semantic_mask[ind][0])
                    ])
                    point_offset[sel_global] = center - x[sel]

            # Get the boundary points here
            alldist = torch.abs(
                torch.sum(x * plane_upper[:3], 1) + plane_upper[-1])
            mind = alldist.min()
            sel = torch.abs(alldist - mind) < self.train_cfg['dist_thresh']

            # Get upper four lines
            if self.primitive_mode == 'line':
                point2_lines_matching = self.match_point2line(
                    x[sel], xmin, xmax, ymin, ymax)
                for idx, line_select in enumerate(point2_lines_matching):
                    if torch.sum(line_select) > \
                            self.train_cfg['num_point_line']:
                        point_mask[ind[sel][line_select]] = 1.0
                        line_center = torch.mean(x[sel][line_select], axis=0)
                        if idx < 2:
                            line_center[1] = (ymin + ymax) / 2.0
                        else:
                            line_center[0] = (xmin + xmax) / 2.0
                        point_offset[ind[sel][line_select]] = \
                            line_center - x[sel][line_select]
                        point_sem[ind[sel][line_select]] = \
                            points.new_tensor([line_center[0], line_center[1],
                                               line_center[2],
                                               pts_semantic_mask[ind][0]])

            if self.primitive_mode == 'z':
                if torch.sum(sel) > self.train_cfg['num_point'] and torch.var(
                        alldist[sel]) < self.train_cfg['var_thresh']:
                    center = points.new_tensor([(xmin + xmax) / 2.0,
                                                (ymin + ymax) / 2.0,
                                                torch.mean(x[sel][:, 2])])
                    sel_global = ind[sel]
                    point_mask[sel_global] = 1.0
                    point_sem[sel_global] = points.new_tensor([
                        center[0], center[1], center[2], xmax - xmin,
                        ymax - ymin, (pts_semantic_mask[ind][0])
                    ])
                    point_offset[sel_global] = center - x[sel]

            # Get left two lines
            vec1 = corners[3] - corners[2]
            vec2 = corners[2] - corners[0]
            surface_norm = torch.cross(vec1, vec2)
            surface_dis = -torch.dot(surface_norm, corners[0])
            plane_left_temp = points.new_tensor([
                surface_norm[0], surface_norm[1], surface_norm[2], surface_dis
            ])

            para_points = corners[[4, 5, 6, 7]]
            # Normalize xy here
            plane_left_temp /= torch.norm(plane_left_temp[:3])
            newd = torch.sum(para_points * plane_left_temp[:3], 1)
            if plane_left_temp[2] < self.train_cfg['lower_thresh']:
                plane_left = plane_left_temp
                plane_right = points.new_tensor([
                    plane_left_temp[0], plane_left_temp[1], plane_left_temp[2],
                    -torch.mean(newd)
                ])
            else:
                raise NotImplementedError
                # print('error with upright')

            # Get the boundary points here
            alldist = torch.abs(
                torch.sum(x * plane_left[:3], 1) + plane_left[-1])
            mind = alldist.min()
            sel = torch.abs(alldist - mind) < self.train_cfg['dist_thresh']

            # Get upper four lines
            if self.primitive_mode == 'line':
                _, _, line_sel1, line_sel2 = self.match_point2line(
                    x[sel], xmin, xmax, ymin, ymax)
                for idx, line_select in enumerate([line_sel1, line_sel2]):
                    if torch.sum(line_select) > \
                            self.train_cfg['num_point_line']:
                        point_mask[ind[sel][line_select]] = 1.0
                        line_center = torch.mean(x[sel][line_select], axis=0)
                        line_center[2] = (zmin + zmax) / 2.0
                        point_offset[ind[sel][line_select]] = \
                            line_center - x[sel][line_select]
                        point_sem[ind[sel][line_select]] = \
                            points.new_tensor([line_center[0], line_center[1],
                                               line_center[2],
                                               pts_semantic_mask[ind][0]])

            if self.primitive_mode == 'xy':
                if torch.sum(sel) > self.train_cfg['num_point'] and torch.var(
                        alldist[sel]) < self.train_cfg['var_thresh']:
                    center = points.new_tensor([
                        torch.mean(x[sel][:, 0]),
                        torch.mean(x[sel][:, 1]), (zmin + zmax) / 2.0
                    ])
                    sel_global = ind[sel]
                    point_mask[sel_global] = 1.0
                    point_sem[sel_global] = points.new_tensor([
                        center[0], center[1], center[2], zmax - zmin,
                        (pts_semantic_mask[ind][0])
                    ])
                    point_offset[sel_global] = center - x[sel]

            # Get the boundary points here
            alldist = torch.abs(
                torch.sum(x * plane_right[:3], 1) + plane_right[-1])
            mind = alldist.min()
            sel = torch.abs(alldist - mind) < self.train_cfg['dist_thresh']

            if self.primitive_mode == 'line':
                _, _, line_sel1, line_sel2 = self.match_point2line(
                    x[sel], xmin, xmax, ymin, ymax)
                for idx, line_select in enumerate([line_sel1, line_sel2]):
                    if torch.sum(line_select) > \
                            self.train_cfg['num_point_line']:
                        point_mask[ind[sel][line_select]] = 1.0
                        line_center = torch.mean(x[sel][line_select], axis=0)
                        line_center[2] = (zmin + zmax) / 2.0
                        point_offset[ind[sel][line_select]] = \
                            line_center - x[sel][line_select]
                        point_sem[ind[sel][line_select]] = \
                            points.new_tensor([line_center[0], line_center[1],
                                               line_center[2],
                                               pts_semantic_mask[ind][0]])

            if self.primitive_mode == 'xy':
                if torch.sum(sel) > self.train_cfg['num_point'] and torch.var(
                        alldist[sel]) < self.train_cfg['var_thresh']:
                    center = points.new_tensor([
                        torch.mean(x[sel][:, 0]),
                        torch.mean(x[sel][:, 1]), (zmin + zmax) / 2.0
                    ])
                    sel_global = ind[sel]
                    point_mask[sel_global] = 1.0
                    point_sem[sel_global] = points.new_tensor([
                        center[0], center[1], center[2], zmax - zmin,
                        (pts_semantic_mask[ind][0])
                    ])
                    point_offset[sel_global] = center - x[sel]

            # Get the boundary points here
            vec1 = corners[0] - corners[4]
            vec2 = corners[4] - corners[5]
            surface_norm = torch.cross(vec1, vec2)
            surface_dis = -torch.dot(surface_norm, corners[5])
            plane_front_temp = points.new_tensor([
                surface_norm[0], surface_norm[1], surface_norm[2], surface_dis
            ])

            para_points = corners[[2, 3, 6, 7]]
            plane_front_temp /= torch.norm(plane_front_temp[:3])
            newd = torch.sum(para_points * plane_front_temp[:3], 1)
            if plane_front_temp[2] < self.train_cfg['lower_thresh']:
                plane_front = plane_front_temp
                plane_back = points.new_tensor([
                    plane_front_temp[0], plane_front_temp[1],
                    plane_front_temp[2], -torch.mean(newd)
                ])
            else:
                raise NotImplementedError
                # print('error with upright')

            # Get the boundary points here
            alldist = torch.abs(
                torch.sum(x * plane_front[:3], 1) + plane_front[-1])
            mind = alldist.min()
            sel = torch.abs(alldist - mind) < self.train_cfg['dist_thresh']
            if self.primitive_mode == 'xy':
                if torch.sum(sel) > self.train_cfg['num_point'] and torch.var(
                        alldist[sel]) < self.train_cfg['var_thresh']:
                    center = points.new_tensor([
                        torch.mean(x[sel][:, 0]),
                        torch.mean(x[sel][:, 1]), (zmin + zmax) / 2.0
                    ])
                    sel_global = ind[sel]
                    point_mask[sel_global] = 1.0
                    point_sem[sel_global] = points.new_tensor([
                        center[0], center[1], center[2], zmax - zmin,
                        (pts_semantic_mask[ind][0])
                    ])
                    point_offset[sel_global] = center - x[sel]

            # Get the boundary points here
            alldist = torch.abs(
                torch.sum(x * plane_back[:3], 1) + plane_back[-1])
            mind = alldist.min()
            sel = torch.abs(alldist - mind) < self.train_cfg['dist_thresh']
            if self.primitive_mode == 'xy':
                if torch.sum(sel) > self.train_cfg['num_point'] and torch.var(
                        alldist[sel]) < self.train_cfg['var_thresh']:
                    center = points.new_tensor([
                        torch.mean(x[sel][:, 0]),
                        torch.mean(x[sel][:, 1]), (zmin + zmax) / 2.0
                    ])
                    sel_global = ind[sel]
                    point_mask[sel_global] = 1.0
                    point_sem[sel_global] = points.new_tensor([
                        center[0], center[1], center[2], zmax - zmin,
                        (pts_semantic_mask[ind][0])
                    ])
                    point_offset[sel_global] = center - x[sel]

        return (point_mask, point_sem, point_offset)

    def primitive_decode_scores(self, preds, aggregated_points, mode='z'):
        """Decode the outputs of primitive module.

        Args:
            preds (torch.Tensor): primitive pridictions of each batch.
            aggregated_points (torch.Tensor): The aggregated points
                of vote stage.
            mode (string): The type of primitive module.

        Returns:
            Dict: Targets of center, size and semantic.
        """
        ret_dict = {}
        net_transposed = preds.transpose(2, 1)

        center = aggregated_points + net_transposed[:, :, 0:3]
        ret_dict['center_' + mode] = center

        if mode in ['z', 'xy']:
            ret_dict['size_residuals_' + mode] = net_transposed[:, :, 3:3 +
                                                                self.num_dim]

        ret_dict['sem_cls_scores_' + mode] = net_transposed[:, :,
                                                            3 + self.num_dim:]

        return center, ret_dict

    def check_upright(self, para_points):
        """Check whether is upright corrdinate.

        Args:
            para_points (torch.Tensor): Points of input.

        Returns:
            Bool: Flag of result.
        """
        return (para_points[0][-1] == para_points[1][-1]) and (
            para_points[1][-1]
            == para_points[2][-1]) and (para_points[2][-1]
                                        == para_points[3][-1])

    def check_z(self, plane_equ, para_points):
        return torch.sum(para_points[:, 2] +
                         plane_equ[-1]) / 4.0 < self.train_cfg['lower_thresh']

    def match_point2line(self, points, xmin, xmax, ymin, ymax):
        """Match points to corresponding edge.

        Args:
            points (torch.Tensor): Points of input.
            xmin (float): Min of X-axis.
            xmax (float): Max of X-axis.
            ymin (float): Min of Y-axis.
            ymax (float): Max of Y-axis.

        Returns:
            Tuple: Flag of matching correspondence.
        """
        sel1 = torch.abs(points[:, 0] - xmin) < self.train_cfg['line_thresh']
        sel2 = torch.abs(points[:, 0] - xmax) < self.train_cfg['line_thresh']
        sel3 = torch.abs(points[:, 1] - ymin) < self.train_cfg['line_thresh']
        sel4 = torch.abs(points[:, 1] - ymax) < self.train_cfg['line_thresh']
        return sel1, sel2, sel3, sel4

    def compute_primitivesem_loss(self, primitive_center, primitive_semantic,
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
        center_loss = center_loss.sum() / (
            torch.sum(gt_primitive_mask.float()) + 1e-6)

        if self.primitive_mode != 'line':
            size_xyz_reshape = primitive_semantic.view(
                batch_size * num_proposal, -1, self.num_dim).contiguous()
            size_loss = self.center_loss(
                size_xyz_reshape,
                gt_primitive_semantic,
                dst_weight=gt_primitive_mask.view(batch_size * num_proposal,
                                                  1))[1]
            size_loss = size_loss.sum() / (
                torch.sum(gt_primitive_mask.float()) + 1e-6)
        else:
            size_loss = torch.tensor(0).float().to(center_loss.device)

        # Semantic cls loss
        sem_cls_loss = self.semantic_loss(semantic_scores, gt_sem_cls_label)
        sem_cls_loss = torch.sum(sem_cls_loss * gt_primitive_mask.float()) / (
            torch.sum(gt_primitive_mask.float()) + 1e-6)

        return center_loss, size_loss, sem_cls_loss

    def get_primitive_center(self, pred_flag, center):
        ind_normal = self.softmax_normal(pred_flag)
        pred_indices = (ind_normal[:, 1, :] >
                        self.surface_thresh).detach().float()
        selected = (ind_normal[:, 1, :] <=
                    self.surface_thresh).detach().float()
        offset = torch.ones_like(center) * self.upper_thresh
        center = center + offset * selected.unsqueeze(-1)
        return center, pred_indices
