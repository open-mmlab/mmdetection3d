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
        num_dim (int): The dimension of primitive.
        num_classes (int): The number of class.
        primitive_mode (str): The mode of primitive.
        bbox_coder (:obj:`BaseBBoxCoder`): Bbox coder for encoding and
            decoding boxes.
        train_cfg (dict): Config for training.
        test_cfg (dict): Config for testing.
        vote_moudule_cfg (dict): Config of VoteModule for point-wise votes.
        vote_aggregation_cfg (dict): Config of vote aggregation layer.
        feat_channels (tuple[int]): Convolution channels of
            prediction layer.
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
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 objectness_loss=None,
                 center_loss=None,
                 semantic_loss=None):
        super(PrimitiveHead, self).__init__()
        self.num_dim = num_dim
        self.num_classes = num_classes
        self.primitive_mode = primitive_mode
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.gt_per_seed = vote_moudule_cfg['gt_per_seed']
        self.num_proposal = vote_aggregation_cfg['num_point']

        self.objectness_loss = build_loss(objectness_loss)
        self.center_loss = build_loss(center_loss)
        self.semantic_loss = build_loss(semantic_loss)

        assert vote_aggregation_cfg['mlp_channels'][0] == vote_moudule_cfg[
            'in_channels']

        # Existence flag prediction
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

    def init_weights(self):
        """Initialize weights of VoteHead."""
        pass

    def forward(self, feat_dict, sample_mod):
        """Forward pass.

        Note:
            The forward of VoteHead is devided into 4 steps:

                1. Generate vote_points from seed_points.
                2. Aggregate vote_points.
                3. Predict primitive cue and score.
                4. Decode predictions.

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

        net_flag = self.flag_conv(seed_features)
        net_flag = self.flag_pred(net_flag)

        results['pred_flag_' + self.primitive_mode] = net_flag

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
        newcenter, decode_res = self.primitive_decode_scores(
            predictions, results, self.num_classes, mode=self.primitive_mode)
        results.update(decode_res)

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
                                   pts_semantic_mask, pts_instance_mask)

        (point_mask, point_sem, point_offset) = targets

        losses = {}
        flag_loss = self.compute_flag_loss(
            bbox_preds, point_mask, mode=self.primitive_mode)
        losses['flag_loss_' + self.primitive_mode] = flag_loss

        # calculate vote loss
        vote_loss = self.vote_module.get_loss(
            bbox_preds['seed_points'],
            bbox_preds['vote_' + self.primitive_mode],
            bbox_preds['seed_indices'], point_mask, point_offset)
        losses['vote_loss_' + self.primitive_mode] = vote_loss

        center_loss, size_loss, sem_cls_loss = self.compute_primitivesem_loss(
            bbox_preds,
            point_mask,
            point_offset,
            point_sem,
            mode=self.primitive_mode)
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
            bbox_preds (torch.Tensor): Predictions of primitive head.

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

        return (point_mask, point_sem, point_offset)

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
        device = points.device

        if self.primitive_mode == 'z':
            point_boundary_mask_z = torch.zeros(num_points).to(device)
            point_boundary_offset_z = torch.zeros([num_points, 3]).to(device)
            point_boundary_sem_z = torch.zeros(
                [num_points, 3 + self.num_dim + 1]).to(device)
        elif self.primitive_mode == 'xy':
            point_boundary_mask_xy = torch.zeros(num_points).to(device)
            point_boundary_offset_xy = torch.zeros([num_points, 3]).to(device)
            point_boundary_sem_xy = torch.zeros(
                [num_points, 3 + self.num_dim + 1]).to(device)
        elif self.primitive_mode == 'line':
            point_line_mask = torch.zeros(num_points).to(device)
            point_line_offset = torch.zeros([num_points, 3]).to(device)
            point_line_sem = torch.zeros([num_points, 3 + 1]).to(device)
        else:
            NotImplementedError

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

            # Get lower four lines
            plane_lower_temp = torch.as_tensor([0, 0, 1,
                                                -corners[6, -1]]).to(device)
            para_points = corners[[1, 3, 5, 7]]
            newd = torch.sum(para_points * plane_lower_temp[:3], 1)
            if self.check_upright(para_points) and \
                    plane_lower_temp[0] + plane_lower_temp[1] < \
                    self.train_cfg['lower_thresh']:
                plane_lower = torch.as_tensor([0, 0, 1, plane_lower_temp[-1]
                                               ]).to(device)
                plane_upper = torch.as_tensor([0, 0, 1,
                                               -torch.mean(newd)]).to(device)
            else:
                import pdb
                pdb.set_trace()
                print('error with upright')
            if self.check_z(plane_upper, para_points) is False:
                import pdb
                pdb.set_trace()

            # Get the boundary points here
            alldist = torch.abs(
                torch.sum(x * plane_lower[:3], 1) + plane_lower[-1])
            mind = alldist.min()
            sel = torch.abs(alldist - mind) < self.train_cfg['dist_thresh']

            # Get lower four lines
            line_sel1, line_sel2, line_sel3, line_sel4 = self.get_linesel(
                x[sel], xmin, xmax, ymin, ymax)
            if self.primitive_mode == 'line':
                if torch.sum(line_sel1) > self.train_cfg['num_point_line']:
                    point_line_mask[ind[sel][line_sel1]] = 1.0
                    linecenter = torch.mean(x[sel][line_sel1], axis=0)
                    linecenter[1] = (ymin + ymax) / 2.0
                    point_line_offset[
                        ind[sel][line_sel1]] = linecenter - x[sel][line_sel1]
                    point_line_sem[ind[sel][line_sel1]] = torch.as_tensor([
                        linecenter[0], linecenter[1], linecenter[2],
                        (pts_semantic_mask[ind][0])
                    ]).to(device)
                if torch.sum(line_sel2) > self.train_cfg['num_point_line']:
                    point_line_mask[ind[sel][line_sel2]] = 1.0
                    linecenter = torch.mean(x[sel][line_sel2], axis=0)
                    linecenter[1] = (ymin + ymax) / 2.0
                    point_line_offset[
                        ind[sel][line_sel2]] = linecenter - x[sel][line_sel2]
                    point_line_sem[ind[sel][line_sel2]] = torch.as_tensor([
                        linecenter[0], linecenter[1], linecenter[2],
                        (pts_semantic_mask[ind][0])
                    ]).to(device)
                if torch.sum(line_sel3) > self.train_cfg['num_point_line']:
                    point_line_mask[ind[sel][line_sel3]] = 1.0
                    linecenter = torch.mean(x[sel][line_sel3], axis=0)
                    linecenter[0] = (xmin + xmax) / 2.0
                    point_line_offset[
                        ind[sel][line_sel3]] = linecenter - x[sel][line_sel3]
                    point_line_sem[ind[sel][line_sel3]] = torch.as_tensor([
                        linecenter[0], linecenter[1], linecenter[2],
                        pts_semantic_mask[ind][0]
                    ]).to(device)
                if torch.sum(line_sel4) > self.train_cfg['num_point_line']:
                    point_line_mask[ind[sel][line_sel4]] = 1.0
                    linecenter = torch.mean(x[sel][line_sel4], axis=0)
                    linecenter[0] = (xmin + xmax) / 2.0
                    point_line_offset[
                        ind[sel][line_sel4]] = linecenter - x[sel][line_sel4]
                    point_line_sem[ind[sel][line_sel4]] = torch.as_tensor([
                        linecenter[0], linecenter[1], linecenter[2],
                        (pts_semantic_mask[ind][0])
                    ]).to(device)

            # Set the surface labels here
            if self.primitive_mode == 'z':
                if torch.sum(sel) > self.train_cfg['num_point'] and torch.var(
                        alldist[sel]) < self.train_cfg['var_thresh']:
                    center = torch.as_tensor([
                        (xmin + xmax) / 2.0, (ymin + ymax) / 2.0,
                        torch.mean(x[sel][:, 2])
                    ]).to(device)
                    sel_global = ind[sel]
                    point_boundary_mask_z[sel_global] = 1.0
                    point_boundary_sem_z[sel_global] = torch.as_tensor([
                        center[0], center[1], center[2], xmax - xmin,
                        ymax - ymin, (pts_semantic_mask[ind][0])
                    ]).to(device)
                    point_boundary_offset_z[sel_global] = center - x[sel]

            # Get the boundary points here
            alldist = torch.abs(
                torch.sum(x * plane_upper[:3], 1) + plane_upper[-1])
            mind = alldist.min()
            sel = torch.abs(alldist - mind) < self.train_cfg['dist_thresh']

            # Get upper four lines
            line_sel1, line_sel2, line_sel3, line_sel4 = self.get_linesel(
                x[sel], xmin, xmax, ymin, ymax)

            if self.primitive_mode == 'line':
                if torch.sum(line_sel1) > self.train_cfg['num_point_line']:
                    point_line_mask[ind[sel][line_sel1]] = 1.0
                    linecenter = torch.mean(x[sel][line_sel1], axis=0)
                    linecenter[1] = (ymin + ymax) / 2.0
                    point_line_offset[
                        ind[sel][line_sel1]] = linecenter - x[sel][line_sel1]
                    point_line_sem[ind[sel][line_sel1]] = torch.as_tensor([
                        linecenter[0], linecenter[1], linecenter[2],
                        (pts_semantic_mask[ind][0])
                    ]).to(device)
                if torch.sum(line_sel2) > self.train_cfg['num_point_line']:
                    point_line_mask[ind[sel][line_sel2]] = 1.0
                    linecenter = torch.mean(x[sel][line_sel2], axis=0)
                    linecenter[1] = (ymin + ymax) / 2.0
                    point_line_offset[
                        ind[sel][line_sel2]] = linecenter - x[sel][line_sel2]
                    point_line_sem[ind[sel][line_sel2]] = torch.as_tensor([
                        linecenter[0], linecenter[1], linecenter[2],
                        (pts_semantic_mask[ind][0])
                    ]).to(device)
                if torch.sum(line_sel3) > self.train_cfg['num_point_line']:
                    point_line_mask[ind[sel][line_sel3]] = 1.0
                    linecenter = torch.mean(x[sel][line_sel3], axis=0)
                    linecenter[0] = (xmin + xmax) / 2.0
                    point_line_offset[
                        ind[sel][line_sel3]] = linecenter - x[sel][line_sel3]
                    point_line_sem[ind[sel][line_sel3]] = torch.as_tensor([
                        linecenter[0], linecenter[1], linecenter[2],
                        (pts_semantic_mask[ind][0])
                    ]).to(device)
                if torch.sum(line_sel4) > self.train_cfg['num_point_line']:
                    point_line_mask[ind[sel][line_sel4]] = 1.0
                    linecenter = torch.mean(x[sel][line_sel4], axis=0)
                    linecenter[0] = (xmin + xmax) / 2.0
                    point_line_offset[
                        ind[sel][line_sel4]] = linecenter - x[sel][line_sel4]
                    point_line_sem[ind[sel][line_sel4]] = torch.as_tensor([
                        linecenter[0], linecenter[1], linecenter[2],
                        (pts_semantic_mask[ind][0])
                    ]).to(device)

            if self.primitive_mode == 'z':
                if torch.sum(sel) > self.train_cfg['num_point'] and torch.var(
                        alldist[sel]) < self.train_cfg['var_thresh']:
                    center = torch.as_tensor([
                        (xmin + xmax) / 2.0, (ymin + ymax) / 2.0,
                        torch.mean(x[sel][:, 2])
                    ]).to(device)
                    sel_global = ind[sel]
                    point_boundary_mask_z[sel_global] = 1.0
                    point_boundary_sem_z[sel_global] = torch.as_tensor([
                        center[0], center[1], center[2], xmax - xmin,
                        ymax - ymin, (pts_semantic_mask[ind][0])
                    ]).to(device)
                    point_boundary_offset_z[sel_global] = center - x[sel]

            # Get left two lines
            v1 = corners[3] - corners[2]
            v2 = corners[2] - corners[0]
            cp = torch.cross(v1, v2)
            d = -torch.dot(cp, corners[0])
            a, b, c = cp
            plane_left_temp = torch.as_tensor([a, b, c, d]).to(device)
            para_points = corners[[4, 5, 6, 7]]
            # Normalize xy here
            plane_left_temp /= torch.norm(plane_left_temp[:3])
            newd = torch.sum(para_points * plane_left_temp[:3], 1)
            if plane_left_temp[2] < self.train_cfg['lower_thresh']:
                plane_left = plane_left_temp
                plane_right = torch.as_tensor([
                    plane_left_temp[0], plane_left_temp[1], plane_left_temp[2],
                    -torch.mean(newd)
                ]).to(device)
            else:
                import pdb
                pdb.set_trace()
                print('error with upright')

            # Get the boundary points here
            alldist = torch.abs(
                torch.sum(x * plane_left[:3], 1) + plane_left[-1])
            mind = alldist.min()
            sel = torch.abs(alldist - mind) < self.train_cfg['dist_thresh']

            # Get upper four lines
            line_sel1, line_sel2 = self.get_linesel2(
                x[sel], ymin, ymax, zmin, zmax, axis=1)
            if self.primitive_mode == 'line':
                if torch.sum(line_sel1) > self.train_cfg['num_point_line']:
                    point_line_mask[ind[sel][line_sel1]] = 1.0
                    linecenter = torch.mean(x[sel][line_sel1], axis=0)
                    linecenter[2] = (zmin + zmax) / 2.0
                    point_line_offset[
                        ind[sel][line_sel1]] = linecenter - x[sel][line_sel1]
                    point_line_sem[ind[sel][line_sel1]] = torch.as_tensor([
                        linecenter[0], linecenter[1], linecenter[2],
                        (pts_semantic_mask[ind][0])
                    ]).to(device)
                if torch.sum(line_sel2) > self.train_cfg['num_point_line']:
                    point_line_mask[ind[sel][line_sel2]] = 1.0
                    linecenter = torch.mean(x[sel][line_sel2], axis=0)
                    linecenter[2] = (zmin + zmax) / 2.0
                    point_line_offset[
                        ind[sel][line_sel2]] = linecenter - x[sel][line_sel2]
                    point_line_sem[ind[sel][line_sel2]] = torch.as_tensor([
                        linecenter[0], linecenter[1], linecenter[2],
                        (pts_semantic_mask[ind][0])
                    ]).to(device)

            if self.primitive_mode == 'xy':
                if torch.sum(sel) > self.train_cfg['num_point'] and torch.var(
                        alldist[sel]) < self.train_cfg['var_thresh']:
                    center = torch.as_tensor([
                        torch.mean(x[sel][:, 0]),
                        torch.mean(x[sel][:, 1]), (zmin + zmax) / 2.0
                    ]).to(device)
                    sel_global = ind[sel]
                    point_boundary_mask_xy[sel_global] = 1.0
                    point_boundary_sem_xy[sel_global] = torch.as_tensor([
                        center[0], center[1], center[2], zmax - zmin,
                        (pts_semantic_mask[ind][0])
                    ]).to(device)
                    point_boundary_offset_xy[sel_global] = center - x[sel]

            # Get the boundary points here
            alldist = torch.abs(
                torch.sum(x * plane_right[:3], 1) + plane_right[-1])
            mind = alldist.min()
            sel = torch.abs(alldist - mind) < self.train_cfg['dist_thresh']
            line_sel1, line_sel2 = self.get_linesel2(
                x[sel], ymin, ymax, zmin, zmax, axis=1)
            if self.primitive_mode == 'line':
                if torch.sum(line_sel1) > self.train_cfg['num_point_line']:
                    point_line_mask[ind[sel][line_sel1]] = 1.0
                    linecenter = torch.mean(x[sel][line_sel1], axis=0)
                    linecenter[2] = (zmin + zmax) / 2.0
                    point_line_offset[
                        ind[sel][line_sel1]] = linecenter - x[sel][line_sel1]
                    point_line_sem[ind[sel][line_sel1]] = torch.as_tensor([
                        linecenter[0], linecenter[1], linecenter[2],
                        (pts_semantic_mask[ind][0])
                    ]).to(device)
                if torch.sum(line_sel2) > self.train_cfg['num_point_line']:
                    point_line_mask[ind[sel][line_sel2]] = 1.0
                    linecenter = torch.mean(x[sel][line_sel2], axis=0)
                    linecenter[2] = (zmin + zmax) / 2.0
                    point_line_offset[
                        ind[sel][line_sel2]] = linecenter - x[sel][line_sel2]
                    point_line_sem[ind[sel][line_sel2]] = torch.as_tensor([
                        linecenter[0], linecenter[1], linecenter[2],
                        (pts_semantic_mask[ind][0])
                    ]).to(device)

            if self.primitive_mode == 'xy':
                if torch.sum(sel) > self.train_cfg['num_point'] and torch.var(
                        alldist[sel]) < self.train_cfg['var_thresh']:
                    center = torch.as_tensor([
                        torch.mean(x[sel][:, 0]),
                        torch.mean(x[sel][:, 1]), (zmin + zmax) / 2.0
                    ]).to(device)
                    sel_global = ind[sel]
                    point_boundary_mask_xy[sel_global] = 1.0
                    point_boundary_sem_xy[sel_global] = torch.as_tensor([
                        center[0], center[1], center[2], zmax - zmin,
                        (pts_semantic_mask[ind][0])
                    ]).to(device)
                    point_boundary_offset_xy[sel_global] = center - x[sel]

            # Get the boundary points here
            v1 = corners[0] - corners[4]
            v2 = corners[4] - corners[5]
            cp = torch.cross(v1, v2)
            d = -torch.dot(cp, corners[5])
            a, b, c = cp
            plane_front_temp = torch.as_tensor([a, b, c, d]).to(device)
            para_points = corners[[2, 3, 6, 7]]
            plane_front_temp /= torch.norm(plane_front_temp[:3])
            newd = torch.sum(para_points * plane_front_temp[:3], 1)
            if plane_front_temp[2] < self.train_cfg['lower_thresh']:
                plane_front = plane_front_temp
                plane_back = torch.as_tensor([
                    plane_front_temp[0], plane_front_temp[1],
                    plane_front_temp[2], -torch.mean(newd)
                ]).to(device)
            else:
                import pdb
                pdb.set_trace()
                print('error with upright')

            # Get the boundary points here
            alldist = torch.abs(
                torch.sum(x * plane_front[:3], 1) + plane_front[-1])
            mind = alldist.min()
            sel = torch.abs(alldist - mind) < self.train_cfg['dist_thresh']
            if self.primitive_mode == 'xy':
                if torch.sum(sel) > self.train_cfg['num_point'] and torch.var(
                        alldist[sel]) < self.train_cfg['var_thresh']:
                    center = torch.as_tensor([
                        torch.mean(x[sel][:, 0]),
                        torch.mean(x[sel][:, 1]), (zmin + zmax) / 2.0
                    ]).to(device)
                    sel_global = ind[sel]
                    point_boundary_mask_xy[sel_global] = 1.0
                    point_boundary_sem_xy[sel_global] = torch.as_tensor([
                        center[0], center[1], center[2], zmax - zmin,
                        (pts_semantic_mask[ind][0])
                    ]).to(device)
                    point_boundary_offset_xy[sel_global] = center - x[sel]
            # Get the boundary points here
            alldist = torch.abs(
                torch.sum(x * plane_back[:3], 1) + plane_back[-1])
            mind = alldist.min()
            sel = torch.abs(alldist - mind) < self.train_cfg['dist_thresh']
            if self.primitive_mode == 'xy':
                if torch.sum(sel) > self.train_cfg['num_point'] and torch.var(
                        alldist[sel]) < self.train_cfg['var_thresh']:
                    center = torch.as_tensor([
                        torch.mean(x[sel][:, 0]),
                        torch.mean(x[sel][:, 1]), (zmin + zmax) / 2.0
                    ]).to(device)
                    sel_global = ind[sel]
                    point_boundary_mask_xy[sel_global] = 1.0
                    point_boundary_sem_xy[sel_global] = torch.as_tensor([
                        center[0], center[1], center[2], zmax - zmin,
                        (pts_semantic_mask[ind][0])
                    ]).to(device)
                    point_boundary_offset_xy[sel_global] = center - x[sel]

        if self.primitive_mode == 'z':
            return (point_boundary_mask_z, point_boundary_sem_z,
                    point_boundary_offset_z)
        elif self.primitive_mode == 'xy':
            return (point_boundary_mask_xy, point_boundary_sem_xy,
                    point_boundary_offset_xy)
        elif self.primitive_mode == 'line':
            return (point_line_mask, point_line_sem, point_line_offset)
        else:
            NotImplementedError

    def primitive_decode_scores(self, net, end_points, num_class, mode=''):
        net_transposed = net.transpose(2, 1)  # (batch_size, 1024, ..)

        base_xyz = end_points['aggregated_points_' +
                              mode]  # (batch_size, num_proposal, 3)
        center = base_xyz + net_transposed[:, :, 0:
                                           3]  # (batch_size, num_proposal, 3)
        end_points['center_' + mode] = center

        if mode in ['z', 'xy']:
            end_points['size_residuals_' + mode] = net_transposed[:, :, 3:3 +
                                                                  self.num_dim]

        end_points['sem_cls_scores_' + mode] = net_transposed[:, :, 3 +
                                                              self.num_dim:]

        return center, end_points

    def check_upright(self, para_points):
        return (para_points[0][-1] == para_points[1][-1]) and (
            para_points[1][-1]
            == para_points[2][-1]) and (para_points[2][-1]
                                        == para_points[3][-1])

    def check_z(self, plane_equ, para_points):
        return torch.sum(para_points[:, 2] +
                         plane_equ[-1]) / 4.0 < self.train_cfg['lower_thresh']

    def get_linesel(self, points, xmin, xmax, ymin, ymax):
        sel1 = torch.abs(points[:, 0] - xmin) < self.train_cfg['line_thresh']
        sel2 = torch.abs(points[:, 0] - xmax) < self.train_cfg['line_thresh']
        sel3 = torch.abs(points[:, 1] - ymin) < self.train_cfg['line_thresh']
        sel4 = torch.abs(points[:, 1] - ymax) < self.train_cfg['line_thresh']
        return sel1, sel2, sel3, sel4

    def get_linesel2(self, points, ymin, ymax, zmin, zmax, axis=0):
        sel3 = torch.abs(points[:, axis] -
                         ymin) < self.train_cfg['line_thresh']
        sel4 = torch.abs(points[:, axis] -
                         ymax) < self.train_cfg['line_thresh']
        return sel3, sel4

    def compute_flag_loss(self, end_points, point_mask, mode):
        # Compute existence flag for face and edge centers
        # Load ground truth votes and assign them to seed points
        seed_inds = end_points['seed_indices'].long()

        seed_gt_votes_mask = torch.gather(point_mask, 1, seed_inds).float()
        end_points['sem_mask'] = seed_gt_votes_mask

        sem_cls_label = torch.gather(point_mask, 1, seed_inds)
        end_points['sub_point_sem_cls_label_' + mode] = sem_cls_label

        pred_flag = end_points['pred_flag_' + mode]

        sem_loss = self.objectness_loss(pred_flag, sem_cls_label.long())

        return sem_loss

    def compute_primitivesem_loss(self,
                                  end_points,
                                  point_mask,
                                  point_offset,
                                  point_sem,
                                  mode=''):
        """Compute final geometric primitive center and semantic."""
        # Load ground truth votes and assign them to seed points
        batch_size = end_points['seed_points'].shape[0]
        num_seed = end_points['seed_points'].shape[1]  # B,num_seed,3
        vote_xyz = end_points['center_' + mode]  # B,num_seed*vote_factor,3
        seed_inds = end_points['seed_indices'].long()

        num_proposal = end_points['aggregated_points_' +
                                  mode].shape[1]  # B,num_seed,3

        seed_gt_votes_mask = torch.gather(point_mask, 1, seed_inds)
        seed_inds_expand = seed_inds.view(batch_size, num_seed,
                                          1).repeat(1, 1, 3)

        seed_inds_expand_sem = seed_inds.view(batch_size, num_seed, 1).repeat(
            1, 1, 4 + self.num_dim)

        seed_gt_votes = torch.gather(point_offset, 1, seed_inds_expand)
        seed_gt_sem = torch.gather(point_sem, 1, seed_inds_expand_sem)
        seed_gt_votes += end_points['seed_points']

        end_points['surface_center_gt_' + mode] = seed_gt_votes
        end_points['surface_sem_gt_' + mode] = seed_gt_sem
        end_points['surface_mask_gt_' + mode] = seed_gt_votes_mask

        # Compute the min of min of distance
        vote_xyz_reshape = vote_xyz.view(batch_size * num_proposal, -1, 3)
        seed_gt_votes_reshape = seed_gt_votes.view(batch_size * num_proposal,
                                                   1, 3)
        # A predicted vote to no where is not penalized as long as there is a
        # good vote near the GT vote.
        center_loss = self.center_loss(
            vote_xyz_reshape,
            seed_gt_votes_reshape,
            dst_weight=seed_gt_votes_mask.view(batch_size * num_proposal,
                                               1))[1]
        center_loss = center_loss.sum() / (
            torch.sum(seed_gt_votes_mask.float()) + 1e-6)

        # Compute the min of min of distance
        # Need to remove this soon
        if mode != 'line':
            size_xyz = end_points[
                'size_residuals_' +
                mode].contiguous()  # B,num_seed*vote_factor,3
            size_xyz_reshape = size_xyz.view(batch_size * num_proposal, -1,
                                             self.num_dim).contiguous()
            seed_gt_votes_reshape = seed_gt_sem[:, :, 3:3 + self.num_dim].view(
                batch_size * num_proposal, 1, self.num_dim).contiguous()
            # A predicted vote to no where is not penalized as long as
            # there is a good vote near the GT vote.
            size_loss = self.center_loss(
                size_xyz_reshape,
                seed_gt_votes_reshape,
                dst_weight=seed_gt_votes_mask.view(batch_size * num_proposal,
                                                   1))[1]
            size_loss = size_loss.sum() / (
                torch.sum(seed_gt_votes_mask.float()) + 1e-6)
        else:
            size_loss = torch.tensor(0).float().to(center_loss.device)

        # 3.4 Semantic cls loss
        sem_cls_label = seed_gt_sem[:, :, -1].long()
        end_points['supp_sem_' + mode] = sem_cls_label
        sem_cls_loss = self.semantic_loss(
            end_points['sem_cls_scores_' + mode].transpose(2, 1),
            sem_cls_label)
        sem_cls_loss = torch.sum(sem_cls_loss * seed_gt_votes_mask.float()) / (
            torch.sum(seed_gt_votes_mask.float()) + 1e-6)

        return center_loss, size_loss, sem_cls_loss
