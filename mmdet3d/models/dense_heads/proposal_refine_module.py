import torch
from mmcv.cnn import ConvModule
from torch import nn as nn

from mmdet3d.models.builder import build_loss
from mmdet3d.models.losses import chamfer_distance
from mmdet3d.ops import PointSAModule
from mmdet.core import multi_apply


def rotz_batch_pytorch(t):
    """Rotation about the y-axis.

    Args:
        t (torch.Tensor): Rotation angle

    Returns:
        torch.Tensor: Rotation matrix
    """
    input_shape = t.shape
    output = torch.zeros(tuple(list(input_shape) + [3, 3])).cuda()
    c = torch.cos(t)
    s = torch.sin(t)
    output[..., 0, 0] = c
    output[..., 0, 1] = -s
    output[..., 1, 0] = s
    output[..., 1, 1] = c
    output[..., 2, 2] = 1
    return output


def get_surface_line_points_batch_pytorch(obj_size, heading_angle, center):
    """Compute surface and line center of bounding boxes.

    Args:
        box_size (torch.Tensor): Size of bounding boxes.
        heading_angle (torch.Tensor): Clockwise, sunrgbd's angle is clockwise
        center (torch.Tensor): Center of bounding boxes.

    Returns:
        torch.Tensor: Surface and line center of bounding boxes.
    """
    # input_shape = heading_angle.shape
    R = rotz_batch_pytorch(
        -heading_angle.float()
    )  # Add the rotz here, clockwise to counter-clockwise

    offset_x = torch.zeros_like(obj_size)
    offset_y = torch.zeros_like(obj_size)
    offset_z = torch.zeros_like(obj_size)
    offset_x[:, :, 0] = 0.5  # obj_size[:,:,0] / 2.0
    offset_y[:, :, 1] = 0.5  # obj_size[:,:,1] / 2.0
    offset_z[:, :, 2] = 0.5  # obj_size[:,:,2] / 2.0

    obj_upper_surface_center = offset_z * obj_size
    obj_lower_surface_center = -offset_z * obj_size

    obj_front_surface_center = offset_y * obj_size
    obj_back_surface_center = -offset_y * obj_size

    obj_left_surface_center = offset_x * obj_size
    obj_right_surface_center = -offset_x * obj_size
    surface_3d = torch.cat((obj_upper_surface_center, obj_lower_surface_center,
                            obj_front_surface_center, obj_back_surface_center,
                            obj_left_surface_center, obj_right_surface_center),
                           dim=1)

    # Get the object line center here
    obj_line_center_0 = offset_z * obj_size + offset_x * obj_size
    obj_line_center_1 = offset_z * obj_size - offset_x * obj_size
    obj_line_center_2 = offset_z * obj_size + offset_y * obj_size
    obj_line_center_3 = offset_z * obj_size - offset_y * obj_size

    obj_line_center_4 = -offset_z * obj_size + offset_x * obj_size
    obj_line_center_5 = -offset_z * obj_size - offset_x * obj_size
    obj_line_center_6 = -offset_z * obj_size + offset_y * obj_size
    obj_line_center_7 = -offset_z * obj_size - offset_y * obj_size

    obj_line_center_8 = offset_x * obj_size + offset_y * obj_size
    obj_line_center_9 = offset_x * obj_size - offset_y * obj_size
    obj_line_center_10 = -offset_x * obj_size + offset_y * obj_size
    obj_line_center_11 = -offset_x * obj_size - offset_y * obj_size
    line_3d = torch.cat(
        (obj_line_center_0, obj_line_center_1, obj_line_center_2,
         obj_line_center_3, obj_line_center_4, obj_line_center_5,
         obj_line_center_6, obj_line_center_7, obj_line_center_8,
         obj_line_center_9, obj_line_center_10, obj_line_center_11),
        dim=1)

    surface_rot = R.repeat(1, 6, 1, 1)
    surface_3d = torch.matmul(
        surface_3d.unsqueeze(-2), surface_rot.transpose(3, 2)).squeeze(-2)
    surface_center = center.repeat(1, 6, 1) + surface_3d

    line_rot = R.repeat(1, 12, 1, 1)
    line_3d = torch.matmul(line_3d.unsqueeze(-2),
                           line_rot.transpose(3, 2)).squeeze(-2)
    line_center = center.repeat(1, 12, 1) + line_3d

    return surface_center, line_center


class ProposalRefineModule(nn.Module):
    r"""Bbox head of `H3dnet <https://arxiv.org/abs/2006.05682>`_.

    Args:
        num_classes (int): The number of class.
        bbox_coder (:obj:`BaseBBoxCoder`): Bbox coder for encoding and
            decoding boxes.
        primitive_list (list[dict]): Configs of primitive head.
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
        dir_class_loss (dict): Config of direction classification loss.
        dir_res_loss (dict): Config of direction residual regression loss.
        size_class_loss (dict): Config of size classification loss.
        size_res_loss (dict): Config of size residual regression loss.
        semantic_loss (dict): Config of point-wise semantic segmentation loss.
    """

    def __init__(self,
                 num_class,
                 num_heading_bin,
                 num_size_cluster,
                 mean_size_arr,
                 num_proposal,
                 suface_matching_cfg,
                 line_matching_cfg,
                 primitive_refine_channels,
                 upper_thresh=100.0,
                 surface_thresh=0.5,
                 line_thresh=0.5,
                 train_cfg=None,
                 with_angle=False,
                 cues_objectness_loss=None,
                 cues_semantic_loss=None,
                 proposal_objectness_loss=None,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d')):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.with_angle = with_angle
        self.upper_thresh = upper_thresh
        self.surface_thresh = surface_thresh
        self.line_thresh = line_thresh
        self.train_cfg = train_cfg

        self.cues_objectness_loss = build_loss(cues_objectness_loss)
        self.cues_semantic_loss = build_loss(cues_semantic_loss)
        self.proposal_objectness_loss = build_loss(proposal_objectness_loss)

        assert suface_matching_cfg['mlp_channels'][-1] == \
            line_matching_cfg['mlp_channels'][-1]
        # surface center matching
        self.match_surface_center = PointSAModule(**suface_matching_cfg)
        # line center matching
        self.match_line_center = PointSAModule(**line_matching_cfg)

        # Compute the matching scores
        matching_feat_dims = suface_matching_cfg['mlp_channels'][-1]
        self.matching_conv = ConvModule(
            matching_feat_dims,
            matching_feat_dims,
            1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=True,
            inplace=True)
        self.matching_pred = torch.nn.Conv1d(matching_feat_dims, 2, 1)

        # Compute the semantic matching scores
        self.matching_sem_conv = ConvModule(
            matching_feat_dims,
            matching_feat_dims,
            1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=True,
            inplace=True)
        self.matching_sem_pred = torch.nn.Conv1d(matching_feat_dims, 2, 1)

        # Surface feature aggregation
        self.surface_feat_aggregation = list()
        for k in range(2):
            self.surface_feat_aggregation.append(
                ConvModule(
                    matching_feat_dims,
                    matching_feat_dims,
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    bias=True,
                    inplace=True))
        self.surface_feat_aggregation = nn.Sequential(
            *self.surface_feat_aggregation)

        # Line feature aggregation
        self.line_feat_aggregation = list()
        for k in range(2):
            self.line_feat_aggregation.append(
                ConvModule(
                    matching_feat_dims,
                    matching_feat_dims,
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    bias=True,
                    inplace=True))
        self.line_feat_aggregation = nn.Sequential(*self.line_feat_aggregation)

        # Final object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class +
        # residual(num_size_cluster*4)
        prev_channel = 18 * matching_feat_dims
        self.proposal_pred = nn.ModuleList()
        for k in range(len(primitive_refine_channels)):
            self.proposal_pred.append(
                ConvModule(
                    prev_channel,
                    primitive_refine_channels[k],
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    bias=True,
                    inplace=False))
            prev_channel = primitive_refine_channels[k]
        conv_out_channel = (2 + 3 + num_heading_bin * 2 +
                            num_size_cluster * 4 + self.num_class)
        self.proposal_pred.append(nn.Conv1d(prev_channel, conv_out_channel, 1))

        self.softmax_normal = torch.nn.Softmax(dim=1)

    def forward(self, end_points):
        """Forward pass.

        Args:
            end_points (dict): Feature dict from backbone and primitive head.

        Returns:
            dict: Predictions of refined bbox.
        """
        ret_dict = {}
        center_z = end_points['center_z']
        z_feature = end_points['aggregated_features_z']
        center_xy = end_points['center_xy']
        xy_feature = end_points['aggregated_features_xy']
        center_line = end_points['center_line']
        line_feature = end_points['aggregated_features_line']
        center_vote = end_points['center']
        size_vote = end_points['size_res']
        sizescore_vote = end_points['sem_scores']
        original_feature = end_points['aggregated_features']

        batch_size = original_feature.shape[0]
        object_proposal = original_feature.shape[2]

        # Create surface center here
        # Extract surface points and features here
        ind_normal_z = self.softmax_normal(end_points['pred_flag_z'])
        ret_dict['pred_z_ind'] = (ind_normal_z[:, 1, :] >
                                  self.surface_thresh).detach().float()
        z_sel = (ind_normal_z[:, 1, :] <= self.surface_thresh).detach().float()
        offset = torch.ones_like(center_z) * self.upper_thresh
        z_center = center_z + offset * z_sel.unsqueeze(-1)
        z_sem = end_points['sem_cls_scores_z']

        ind_normal_xy = self.softmax_normal(end_points['pred_flag_xy'])
        ret_dict['pred_xy_ind'] = (ind_normal_xy[:, 1, :] >
                                   self.surface_thresh).detach().float()
        xy_sel = (ind_normal_xy[:, 1, :] <=
                  self.surface_thresh).detach().float()
        offset = torch.ones_like(center_xy) * self.upper_thresh
        xy_center = center_xy + offset * xy_sel.unsqueeze(-1)
        xy_sem = end_points['sem_cls_scores_xy']

        surface_center_pred = torch.cat((z_center, xy_center), dim=1)
        ret_dict['surface_center_pred'] = surface_center_pred
        ret_dict['surface_sem_pred'] = torch.cat((z_sem, xy_sem), dim=1)
        surface_center_feature_pred = torch.cat((z_feature, xy_feature), dim=2)
        surface_center_feature_pred = torch.cat((torch.zeros(
            (batch_size, 6, surface_center_feature_pred.shape[2])).cuda(),
                                                 surface_center_feature_pred),
                                                dim=1)

        # Extract line points and features here
        ind_normal_line = self.softmax_normal(end_points['pred_flag_line'])
        ret_dict['pred_line_ind'] = (ind_normal_line[:, 1, :] >
                                     self.line_thresh).detach().float()
        line_sel = (ind_normal_line[:, 1, :] <=
                    self.surface_thresh).detach().float()
        offset = torch.ones_like(center_line) * self.upper_thresh
        line_center = center_line + offset * line_sel.unsqueeze(-1)
        ret_dict['line_center_pred'] = line_center
        ret_dict['line_sem_pred'] = end_points['sem_cls_scores_line']
        ret_dict['aggregated_vote_xyzopt'] = end_points['aggregated_points']

        # Extract the object center here
        obj_center = center_vote.contiguous()
        size_residual = size_vote.contiguous()
        pred_size_class = torch.argmax(sizescore_vote.contiguous(), -1)
        pred_size_residual = torch.gather(
            size_vote.contiguous(), 2,
            pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3))
        mean_size_class_batched = torch.ones_like(
            size_residual) * torch.as_tensor(
                self.mean_size_arr).cuda().unsqueeze(0).unsqueeze(0)
        pred_size_avg = torch.gather(
            mean_size_class_batched, 2,
            pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3))
        obj_size = (pred_size_avg.squeeze(2) +
                    pred_size_residual.squeeze(2)).detach()

        pred_heading_class = torch.argmax(end_points['dir_class'].detach(),
                                          -1)  # B,num_proposal
        pred_heading_residual = torch.gather(
            end_points['dir_res'].detach(), 2,
            pred_heading_class.unsqueeze(-1))  # B,num_proposal,1
        pred_heading_residual.squeeze_(2)

        if not self.with_angle:
            pred_heading = torch.zeros_like(pred_heading_class)
        else:
            raise NotImplementedError
            # TO-DO
            # config = SunrgbdDatasetConfig()
            # pred_heading = pred_heading_class.float() * (2 * np.pi / float(
            #     config.num_heading_bin)) + pred_heading_residual

        obj_surface_center, obj_line_center = \
            get_surface_line_points_batch_pytorch(
                obj_size, pred_heading, obj_center)
        obj_surface_feature = original_feature.repeat(1, 1, 6)
        ret_dict['surface_center_object'] = obj_surface_center
        # Add an indicator for different surfaces
        obj_upper_indicator = torch.zeros(
            (batch_size, object_proposal, 6)).cuda()
        obj_upper_indicator[:, :, 0] = 1
        obj_lower_indicator = torch.zeros(
            (batch_size, object_proposal, 6)).cuda()
        obj_lower_indicator[:, :, 1] = 1
        obj_front_indicator = torch.zeros(
            (batch_size, object_proposal, 6)).cuda()
        obj_front_indicator[:, :, 2] = 1
        obj_back_indicator = torch.zeros(
            (batch_size, object_proposal, 6)).cuda()
        obj_back_indicator[:, :, 3] = 1
        obj_left_indicator = torch.zeros(
            (batch_size, object_proposal, 6)).cuda()
        obj_left_indicator[:, :, 4] = 1
        obj_right_indicator = torch.zeros(
            (batch_size, object_proposal, 6)).cuda()
        obj_right_indicator[:, :, 5] = 1
        obj_surface_indicator = torch.cat(
            (obj_upper_indicator, obj_lower_indicator, obj_front_indicator,
             obj_back_indicator, obj_left_indicator, obj_right_indicator),
            dim=1).transpose(2, 1).contiguous()
        obj_surface_feature = torch.cat(
            (obj_surface_indicator, obj_surface_feature), dim=1)

        obj_line_feature = original_feature.repeat(1, 1, 12)
        ret_dict['line_center_object'] = obj_line_center
        # Add an indicator for different lines
        obj_line_indicator0 = torch.zeros(
            (batch_size, 12, object_proposal)).cuda()
        obj_line_indicator0[:, 0, :] = 1
        obj_line_indicator1 = torch.zeros(
            (batch_size, 12, object_proposal)).cuda()
        obj_line_indicator1[:, 1, :] = 1
        obj_line_indicator2 = torch.zeros(
            (batch_size, 12, object_proposal)).cuda()
        obj_line_indicator2[:, 2, :] = 1
        obj_line_indicator3 = torch.zeros(
            (batch_size, 12, object_proposal)).cuda()
        obj_line_indicator3[:, 3, :] = 1

        obj_line_indicator4 = torch.zeros(
            (batch_size, 12, object_proposal)).cuda()
        obj_line_indicator4[:, 4, :] = 1
        obj_line_indicator5 = torch.zeros(
            (batch_size, 12, object_proposal)).cuda()
        obj_line_indicator5[:, 5, :] = 1
        obj_line_indicator6 = torch.zeros(
            (batch_size, 12, object_proposal)).cuda()
        obj_line_indicator6[:, 6, :] = 1
        obj_line_indicator7 = torch.zeros(
            (batch_size, 12, object_proposal)).cuda()
        obj_line_indicator7[:, 7, :] = 1

        obj_line_indicator8 = torch.zeros(
            (batch_size, 12, object_proposal)).cuda()
        obj_line_indicator8[:, 8, :] = 1
        obj_line_indicator9 = torch.zeros(
            (batch_size, 12, object_proposal)).cuda()
        obj_line_indicator9[:, 9, :] = 1
        obj_line_indicator10 = torch.zeros(
            (batch_size, 12, object_proposal)).cuda()
        obj_line_indicator10[:, 10, :] = 1
        obj_line_indicator11 = torch.zeros(
            (batch_size, 12, object_proposal)).cuda()
        obj_line_indicator11[:, 11, :] = 1

        obj_line_indicator = torch.cat(
            (obj_line_indicator0, obj_line_indicator1, obj_line_indicator2,
             obj_line_indicator3, obj_line_indicator4, obj_line_indicator5,
             obj_line_indicator6, obj_line_indicator7, obj_line_indicator8,
             obj_line_indicator9, obj_line_indicator10, obj_line_indicator11),
            dim=2)
        obj_line_feature = torch.cat((obj_line_indicator, obj_line_feature),
                                     dim=1)

        surface_xyz, surface_features, _ = self.match_surface_center(
            surface_center_pred,
            surface_center_feature_pred,
            target_xyz=obj_surface_center)
        line_feature = torch.cat((torch.zeros(
            (batch_size, 12, line_feature.shape[2])).cuda(), line_feature),
                                 dim=1)

        line_xyz, line_features, _ = self.match_line_center(
            line_center, line_feature, target_xyz=obj_line_center)

        combine_features = torch.cat(
            (surface_features.contiguous(), line_features.contiguous()), dim=2)

        match_features = self.matching_conv(combine_features)
        match_score = self.matching_pred(match_features)
        ret_dict['match_scores'] = match_score.transpose(2, 1).contiguous()

        match_features_sem = self.matching_sem_conv(combine_features)
        match_score_sem = self.matching_sem_pred(match_features_sem)
        ret_dict['match_scores_sem'] = match_score_sem.transpose(
            2, 1).contiguous()

        surface_features = self.surface_feat_aggregation(surface_features)

        line_features = self.line_feat_aggregation(line_features)

        surface_features = surface_features.view(batch_size, -1, 6,
                                                 object_proposal).contiguous()
        line_features = line_features.view(batch_size, -1, 12,
                                           object_proposal).contiguous()

        # Combine all surface and line features
        surface_pool_feature = surface_features.view(
            batch_size, -1, object_proposal).contiguous()
        line_pool_feature = line_features.view(batch_size, -1,
                                               object_proposal).contiguous()

        combine_feature = torch.cat((surface_pool_feature, line_pool_feature),
                                    dim=1)

        net = self.proposal_pred[0](combine_feature)
        net += original_feature
        for conv_module in self.proposal_pred[1:]:
            net = conv_module(net)

        return net, ret_dict

    def loss(self,
             bbox_preds,
             preds_dict,
             points,
             gt_bboxes_3d,
             gt_labels_3d,
             pts_semantic_mask=None,
             pts_instance_mask=None,
             img_metas=None,
             gt_bboxes_ignore=None):
        """Compute loss.

        Args:
            bbox_preds (torch.Tensor): Predictions of the bounding boxes.
            preds_dict (dict): Predictions from forward of vote head.
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
                                   preds_dict)

        (cues_objectness_label, cues_sem_label, proposal_objectness_label,
         cues_mask, cues_match_mask, proposal_objectness_mask,
         cues_matching_label, obj_surface_center, obj_line_center) = targets

        # match scores for each geometric primitive
        objectness_scores = preds_dict['match_scores']
        # match scores for the semantics of primitives
        objectness_scores_sem = preds_dict['match_scores_sem']

        objectness_loss = self.cues_objectness_loss(
            objectness_scores.transpose(2, 1), cues_objectness_label)
        objectness_loss = torch.sum(objectness_loss * cues_mask) / (
            torch.sum(cues_mask) + 1e-6)

        objectness_loss_sem = self.cues_semantic_loss(
            objectness_scores_sem.transpose(2, 1), cues_sem_label)
        objectness_loss_sem = torch.sum(objectness_loss_sem * cues_mask) / (
            torch.sum(cues_mask) + 1e-6)

        objectness_scores = preds_dict['obj_scores_opt']
        objectness_loss_refine = self.proposal_objectness_loss(
            objectness_scores.transpose(2, 1), proposal_objectness_label)
        objectness_loss_refine1 = torch.sum(
            objectness_loss_refine *
            cues_match_mask) / (torch.sum(cues_match_mask) + 1e-6) * 0.5
        objectness_loss_refine2 = torch.sum(
            objectness_loss_refine * proposal_objectness_mask) / (
                torch.sum(proposal_objectness_mask) + 1e-6) * 0.5

        # Get the object surface center here
        obj_size = bbox_preds[:, :, 3:6]
        pred_heading = bbox_preds[:, :, 6]
        obj_center = bbox_preds[:, :, 0:3]
        pred_obj_surface_center, pred_obj_line_center = \
            get_surface_line_points_batch_pytorch(
                obj_size, pred_heading, obj_center)
        source_point = torch.cat(
            (pred_obj_surface_center, pred_obj_line_center), 1)

        target_point = torch.cat((obj_surface_center, obj_line_center), 1)
        dist_match = torch.sqrt(
            torch.sum((source_point - target_point)**2, dim=-1) + 1e-6)
        primitive_centroid_reg_loss = torch.sum(
            dist_match * cues_matching_label) / (
                torch.sum(cues_matching_label) + 1e-6)

        losses = dict(
            objectness_loss=objectness_loss,
            objectness_loss_sem=objectness_loss_sem,
            objectness_loss_refine1=objectness_loss_refine1,
            objectness_loss_refine2=objectness_loss_refine2,
            primitive_centroid_reg_loss=primitive_centroid_reg_loss)
        return losses

    def get_targets(self,
                    points,
                    gt_bboxes_3d,
                    gt_labels_3d,
                    pts_semantic_mask=None,
                    pts_instance_mask=None,
                    bbox_preds=None):
        """Generate targets of proposal module.

        Args:
            points (list[torch.Tensor]): Points of each batch.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth \
                bboxes of each batch.
            gt_labels_3d (list[torch.Tensor]): Labels of each batch.
            pts_semantic_mask (None | list[torch.Tensor]): Point-wise semantic
                label of each batch.
            pts_instance_mask (None | list[torch.Tensor]): Point-wise instance
                label of each batch.
            bbox_preds (torch.Tensor): Bounding box predictions of vote head.

        Returns:
            tuple[torch.Tensor]: Targets of proposal module.
        """
        # find empty example
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

        aggregated_points = [
            bbox_preds['aggregated_points'][i]
            for i in range(len(gt_labels_3d))
        ]

        surface_center_pred = [
            bbox_preds['surface_center_pred'][i]
            for i in range(len(gt_labels_3d))
        ]

        line_center_pred = [
            bbox_preds['line_center_pred'][i]
            for i in range(len(gt_labels_3d))
        ]

        surface_center_object = [
            bbox_preds['surface_center_object'][i]
            for i in range(len(gt_labels_3d))
        ]

        line_center_object = [
            bbox_preds['line_center_object'][i]
            for i in range(len(gt_labels_3d))
        ]

        surface_sem_pred = [
            bbox_preds['surface_sem_pred'][i]
            for i in range(len(gt_labels_3d))
        ]

        line_sem_pred = [
            bbox_preds['line_sem_pred'][i] for i in range(len(gt_labels_3d))
        ]

        (cues_objectness_label, cues_sem_label, proposal_objectness_label,
         cues_mask, cues_match_mask, proposal_objectness_mask,
         cues_matching_label,
         obj_surface_center, obj_line_center) = multi_apply(
             self.get_targets_single, points, gt_bboxes_3d, gt_labels_3d,
             pts_semantic_mask, pts_instance_mask, aggregated_points,
             surface_center_pred, line_center_pred, surface_center_object,
             line_center_object, surface_sem_pred, line_sem_pred)

        cues_objectness_label = torch.stack(cues_objectness_label)
        cues_sem_label = torch.stack(cues_sem_label)
        proposal_objectness_label = torch.stack(proposal_objectness_label)
        cues_mask = torch.stack(cues_mask)
        cues_match_mask = torch.stack(cues_match_mask)
        proposal_objectness_mask = torch.stack(proposal_objectness_mask)
        cues_matching_label = torch.stack(cues_matching_label)
        obj_surface_center = torch.stack(obj_surface_center)
        obj_line_center = torch.stack(obj_line_center)

        return (cues_objectness_label, cues_sem_label,
                proposal_objectness_label, cues_mask, cues_match_mask,
                proposal_objectness_mask, cues_matching_label,
                obj_surface_center, obj_line_center)

    def get_targets_single(self,
                           points,
                           gt_bboxes_3d,
                           gt_labels_3d,
                           pts_semantic_mask=None,
                           pts_instance_mask=None,
                           aggregated_points=None,
                           pred_surface_center=None,
                           pred_line_center=None,
                           pred_obj_surface_center=None,
                           pred_obj_line_center=None,
                           pred_surface_sem=None,
                           pred_line_sem=None):
        """Generate targets for primitive cues for single batch.

        Args:
            points (torch.Tensor): Points of each batch.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): Ground truth \
                boxes of each batch.
            gt_labels_3d (torch.Tensor): Labels of each batch.
            pts_semantic_mask (None | torch.Tensor): Point-wise semantic
                label of each batch.
            pts_instance_mask (None | torch.Tensor): Point-wise instance
                label of each batch.
            aggregated_points (torch.Tensor): Aggregated points from
                vote aggregation layer.
            pred_surface_center (torch.Tensor): Prediction of surface center.
            pred_line_center (torch.Tensor): Prediction of line center.
            pred_obj_surface_center (torch.Tensor): Objectness prediction \
                of surface center.
            pred_obj_line_center (torch.Tensor): Objectness prediction of \
                line center.
            pred_surface_sem (torch.Tensor): Semantic prediction of \
                surface center.
            pred_line_sem (torch.Tensor): Semantic prediction of line center.
        Returns:
            tuple[torch.Tensor]: Targets for primitive cues.
        """
        device = points.device
        gt_bboxes_3d = gt_bboxes_3d.to(device)
        K = aggregated_points.shape[0]
        gt_center = gt_bboxes_3d.gravity_center
        size_label = gt_bboxes_3d.dims
        heading_label = gt_bboxes_3d.yaw

        dist1, dist2, ind1, _ = chamfer_distance(
            aggregated_points.unsqueeze(0),
            gt_center.unsqueeze(0),
            reduction='none')
        # Set assignment
        object_assignment = ind1.squeeze(
            0)  # (B,K) with values in 0,1,...,K2-1

        # Generate objectness label and mask
        # objectness_label: 1 if pred object center is within
        # self.train_cfg['near_threshold'] of any GT object
        # objectness_mask: 0 if pred object center is in gray
        # zone (DONOTCARE), 1 otherwise
        euclidean_dist1 = torch.sqrt(dist1.squeeze(0) + 1e-6)
        proposal_objectness_label = torch.zeros(K, dtype=torch.long).cuda()
        proposal_objectness_mask = torch.zeros(K).cuda()

        obj_center = gt_center[object_assignment].unsqueeze(0)
        gt_size = size_label[object_assignment].unsqueeze(0)
        gt_heading = heading_label[object_assignment].unsqueeze(0)
        gt_sem = gt_labels_3d[object_assignment]

        # gt for primitive matching
        obj_surface_center, obj_line_center = \
            get_surface_line_points_batch_pytorch(
                gt_size, gt_heading, obj_center)

        surface_sem = torch.argmax(pred_surface_sem, dim=1).float()
        line_sem = torch.argmax(pred_line_sem, dim=1).float()

        dist_surface, _, surface_ind, _ = chamfer_distance(
            obj_surface_center, pred_surface_center.unsqueeze(0))
        dist_line, _, line_ind, _ = chamfer_distance(
            obj_line_center, pred_line_center.unsqueeze(0))

        surface_sel = pred_surface_center[surface_ind.squeeze(0)]
        line_sel = pred_line_center[line_ind.squeeze(0)]
        surface_sel_sem = surface_sem[surface_ind.squeeze(0)]
        line_sel_sem = line_sem[line_ind.squeeze(0)]

        surface_sel_sem_gt = gt_sem.repeat(6).float()
        line_sel_sem_gt = gt_sem.repeat(12).float()

        euclidean_dist_surface = torch.sqrt(dist_surface.squeeze(0) + 1e-6)
        euclidean_dist_line = torch.sqrt(dist_line.squeeze(0) + 1e-6)
        objectness_label_surface = torch.zeros(K * 6, dtype=torch.long).cuda()
        objectness_mask_surface = torch.zeros(K * 6).cuda()
        objectness_label_line = torch.zeros(K * 12, dtype=torch.long).cuda()
        objectness_mask_line = torch.zeros(K * 12).cuda()
        objectness_label_surface_sem = torch.zeros(
            K * 6, dtype=torch.long).cuda()
        objectness_label_line_sem = torch.zeros(
            K * 12, dtype=torch.long).cuda()

        euclidean_dist_obj_surface = torch.sqrt(
            torch.sum((pred_obj_surface_center - surface_sel)**2, dim=-1) +
            1e-6)
        euclidean_dist_obj_line = torch.sqrt(
            torch.sum((pred_obj_line_center - line_sel)**2, dim=-1) + 1e-6)

        # Objectness score just with centers
        proposal_objectness_label[
            euclidean_dist1 < self.train_cfg['near_threshold']] = 1
        proposal_objectness_mask[
            euclidean_dist1 < self.train_cfg['near_threshold']] = 1
        proposal_objectness_mask[
            euclidean_dist1 > self.train_cfg['far_threshold']] = 1

        objectness_label_surface[
            (euclidean_dist_obj_surface <
             self.train_cfg['label_surface_threshold']) *
            (euclidean_dist_surface <
             self.train_cfg['mask_surface_threshold'])] = 1
        objectness_label_surface_sem[
            (euclidean_dist_obj_surface <
             self.train_cfg['label_surface_threshold']) *
            (euclidean_dist_surface < self.train_cfg['mask_surface_threshold'])
            * (surface_sel_sem == surface_sel_sem_gt)] = 1

        objectness_label_line[
            (euclidean_dist_obj_line < self.train_cfg['label_line_threshold'])
            *
            (euclidean_dist_line < self.train_cfg['mask_line_threshold'])] = 1
        objectness_label_line_sem[
            (euclidean_dist_obj_line < self.train_cfg['label_line_threshold'])
            * (euclidean_dist_line < self.train_cfg['mask_line_threshold']) *
            (line_sel_sem == line_sel_sem_gt)] = 1

        objectness_label_surface_obj = proposal_objectness_label.repeat(6)
        objectness_mask_surface_obj = proposal_objectness_mask.repeat(6)
        objectness_label_line_obj = proposal_objectness_label.repeat(12)
        objectness_mask_line_obj = proposal_objectness_mask.repeat(12)

        objectness_mask_surface = objectness_mask_surface_obj
        objectness_mask_line = objectness_mask_line_obj

        cues_objectness_label = torch.cat(
            (objectness_label_surface, objectness_label_line), 0)
        cues_sem_label = torch.cat(
            (objectness_label_surface_sem, objectness_label_line_sem), 0)
        cues_mask = torch.cat((objectness_mask_surface, objectness_mask_line),
                              0)

        objectness_label_surface *= objectness_label_surface_obj
        objectness_label_line *= objectness_label_line_obj
        cues_matching_label = torch.cat(
            (objectness_label_surface, objectness_label_line), 0)

        objectness_label_surface_sem *= objectness_label_surface_obj
        objectness_label_line_sem *= objectness_label_line_obj

        cues_match_mask = (torch.sum(cues_objectness_label.view(18, K), dim=0)
                           >= 1).float()

        obj_surface_center = obj_surface_center.squeeze(0)
        obj_line_center = obj_line_center.squeeze(0)

        return (cues_objectness_label, cues_sem_label,
                proposal_objectness_label, cues_mask, cues_match_mask,
                proposal_objectness_mask, cues_matching_label,
                obj_surface_center, obj_line_center)
