import numpy as np
import torch
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch import nn as nn

from mmdet3d.core.bbox.structures import rotation_3d_in_axis, xywhr2xyxyr
from mmdet3d.models.builder import build_loss
from mmdet3d.ops import build_sa_module
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu, nms_normal_gpu
from mmdet.core import build_bbox_coder, multi_apply
from mmdet.models import HEADS


@HEADS.register_module()
class PointRCNNBboxHead(BaseModule):
    """PointRCNN ROI head.

    Args:
        num_classes (int): The number of classes to prediction.
        in_channels (int): Input channels of RCNN
            convolution layer
        conv_channels (list(int)): Out channels of each
            pointrcnn convolution layer.
    """

    def __init__(
            self,
            num_classes,
            in_channels,
            mlp_channels,
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='ReLU'),
            bbox_codesize=7,
            conv_channels=(512, 512),
            num_points=(128, 32, 1),
            radius=(0.2, 0.4, 100),
            num_samples=(64, 64, 64),
            sa_channels=((128, 128, 128), (128, 128, 256), (256, 256, 512)),
            bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
            sa_cfg=dict(type='PointSAModule', pool_mod='max', use_xyz=True),
            loss_bbox=dict(
                type='SmoothL1Loss',
                beta=1.0 / 9.0,
                reduction='sum',
                loss_weight=1.0),
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                reduction='sum',
                loss_weight=1.0),
            init_cfg=None,
            pretrained=None):
        super(PointRCNNBboxHead, self).__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.bbox_codesize = bbox_codesize
        self.num_sa = len(sa_channels)

        self.loss_bbox = build_loss(loss_bbox)
        self.loss_cls = build_loss(loss_cls)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)

        mlp_channels = [5] + mlp_channels
        shared_mlps = nn.Sequential()
        for i in range(len(mlp_channels) - 1):
            shared_mlps.add_module(
                f'layer{i}',
                ConvModule(
                    mlp_channels[i],
                    mlp_channels[i + 1],
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    inplace=False,
                    conv_cfg=dict(type='Conv2d')))
        self.xyz_up_layer = nn.Sequential(*shared_mlps)

        c_out = mlp_channels[-1]
        self.merge_down_layer = ConvModule(
            c_out * 2,
            c_out,
            kernel_size=(1, 1),
            stride=(1, 1),
            inplace=False,
            conv_cfg=dict(type='Conv2d'))

        pre_channels = c_out
        self.cls_layers = self.make_conv_layers(
            conv_channels=conv_channels,
            input_channels=pre_channels,
            output_channels=self.num_classes,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.reg_layers = self.make_conv_layers(
            conv_channels=conv_channels,
            input_channels=pre_channels,
            output_channels=self.bbox_codesize,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.SA_modules = nn.ModuleList()
        sa_in_channel = self.in_channels

        for sa_index in range(self.num_sa):
            cur_sa_mlps = list(sa_channels[sa_index])
            cur_sa_mlps = [sa_in_channel] + cur_sa_mlps
            sa_out_channel = cur_sa_mlps[-1]

            self.SA_modules.append(
                build_sa_module(
                    num_point=num_points[sa_index],
                    radius=radius[sa_index],
                    num_sample=num_samples[sa_index],
                    mlp_channels=cur_sa_mlps,
                    cfg=sa_cfg))
            sa_in_channel = sa_out_channel

        pre_channels = sa_in_channel
        self.cls_layers = self.make_conv_layers(
            conv_channels=conv_channels,
            input_channels=pre_channels,
            output_channels=self.num_classes,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.reg_layers = self.make_conv_layers(
            conv_channels=conv_channels,
            input_channels=pre_channels,
            output_channels=self.bbox_coder.code_size,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    @staticmethod
    def make_conv_layers(conv_channels, input_channels, output_channels,
                         conv_cfg, norm_cfg, act_cfg):
        prev_channels = input_channels
        conv_list = list()
        for k in range(len(conv_channels)):
            conv_list.append(
                ConvModule(
                    prev_channels,
                    conv_channels[k],
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    bias=True))
            prev_channels = conv_channels[k]
        conv_list.append(
            ConvModule(
                prev_channels,
                output_channels,
                1,
                padding=0,
                conv_cfg=conv_cfg,
                act_cfg=None,
                bias=True,
                inplace=False))
        return nn.Sequential(*conv_list)

    def forward(self, feats):
        input_data = feats.clone().detach()
        xyz_input = input_data[..., 0:5].transpose(
            1, 2).unsqueeze(dim=3).contiguous().clone().detach()
        xyz_features = self.xyz_up_layer(xyz_input)
        rpn_features = input_data[..., 5:].transpose(1, 2).unsqueeze(dim=3)
        merged_features = torch.cat((xyz_features, rpn_features), dim=1)
        merged_features = self.merge_down_layer(merged_features)
        l_xyz, l_features = [input_data[..., 0:3].contiguous()], \
                            [merged_features.squeeze(dim=3)]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features, cur_indices = \
                self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        shared_features = l_features[-1]
        rcnn_reg = self.reg_layers(shared_features).transpose(
            1, 2).contiguous().squeeze(dim=1)  # (B, C)
        rcnn_cls = self.cls_layers(shared_features).transpose(
            1, 2).contiguous().squeeze(dim=1)

        return (rcnn_cls, rcnn_reg)

    def loss(self, cls_score, bbox_pred, rois, labels, bbox_targets,
             pos_gt_bboxes, reg_mask, label_weights, bbox_weights):
        losses = dict()
        rcnn_batch_size = cls_score.shape[0]
        # calculate class loss
        cls_flat = cls_score.view(-1)
        loss_cls = self.loss_cls(cls_flat, labels, label_weights)
        losses['loss_cls'] = loss_cls

        # calculate regression loss
        pos_inds = (reg_mask > 0)
        if pos_inds.any() == 0:
            # fake a part loss
            losses['loss_bbox'] = loss_cls.new_tensor(0)
        else:
            pos_bbox_pred = bbox_pred.view(rcnn_batch_size,
                                           -1)[pos_inds].clone()
            bbox_weights_flat = bbox_weights[pos_inds].view(-1, 1).repeat(
                1, pos_bbox_pred.shape[-1])
            loss_bbox = self.loss_bbox(
                pos_bbox_pred.unsqueeze(dim=0),
                bbox_targets.unsqueeze(dim=0).detach(),
                bbox_weights_flat.unsqueeze(dim=0))
            losses['loss_bbox'] = loss_bbox
        return losses

    def get_targets(self, sampling_results, rcnn_train_cfg, concat=True):
        """Generate targets.

        Args:
            sampling_results (list[:obj:`SamplingResult`]):
                Sampled results from rois.
            rcnn_train_cfg (:obj:`ConfigDict`): Training config of rcnn.
            concat (bool): Whether to concatenate targets between batches.

        Returns:
            tuple[torch.Tensor]: Targets of boxes and class prediction.
        """
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        iou_list = [res.iou for res in sampling_results]
        targets = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            pos_gt_bboxes_list,
            iou_list,
            cfg=rcnn_train_cfg)
        (label, bbox_targets, pos_gt_bboxes, reg_mask, label_weights,
         bbox_weights) = targets

        if concat:
            label = torch.cat(label, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            pos_gt_bboxes = torch.cat(pos_gt_bboxes, 0)
            reg_mask = torch.cat(reg_mask, 0)

            label_weights = torch.cat(label_weights, 0)
            label_weights /= torch.clamp(label_weights.sum(), min=1.0)

            bbox_weights = torch.cat(bbox_weights, 0)
            bbox_weights /= torch.clamp(bbox_weights.sum(), min=1.0)

        return (label, bbox_targets, pos_gt_bboxes, reg_mask, label_weights,
                bbox_weights)

    def _get_target_single(self, pos_bboxes, pos_gt_bboxes, ious, cfg):
        """Generate training targets for a single sample.

        Args:
            pos_bboxes (torch.Tensor): Positive boxes with shape
                (N, 7).
            pos_gt_bboxes (torch.Tensor): Ground truth boxes with shape
                (M, 7).
            ious (torch.Tensor): IoU between `pos_bboxes` and `pos_gt_bboxes`
                in shape (N, M).
            cfg (dict): Training configs.

        Returns:
            tuple[torch.Tensor]: Target for positive boxes.
                (label, bbox_targets, pos_gt_bboxes, reg_mask, label_weights,
                bbox_weights)
        """
        cls_pos_mask = ious > cfg.cls_pos_thr
        cls_neg_mask = ious < cfg.cls_neg_thr
        interval_mask = (cls_pos_mask == 0) & (cls_neg_mask == 0)
        # iou regression target
        label = (cls_pos_mask > 0).float()
        label[interval_mask] = ious[interval_mask] * 2 - 0.5
        # label weights
        label_weights = (label >= 0).float()

        # box regression target
        reg_mask = pos_bboxes.new_zeros(ious.size(0)).long()
        reg_mask[0:pos_gt_bboxes.size(0)] = 1
        bbox_weights = (reg_mask > 0).float()
        if reg_mask.bool().any():
            pos_gt_bboxes_ct = pos_gt_bboxes.clone().detach()
            roi_center = pos_bboxes[..., 0:3]
            roi_ry = pos_bboxes[..., 6] % (2 * np.pi)

            # canonical transformation
            pos_gt_bboxes_ct[..., 0:3] -= roi_center
            pos_gt_bboxes_ct[..., 6] -= roi_ry
            pos_gt_bboxes_ct[..., 0:3] = rotation_3d_in_axis(
                pos_gt_bboxes_ct[..., 0:3].unsqueeze(1), -roi_ry,
                axis=2).squeeze(1)

            # flip orientation if rois have opposite orientation
            ry_label = pos_gt_bboxes_ct[..., 6] % (2 * np.pi)  # 0 ~ 2pi
            opposite_flag = (ry_label > np.pi * 0.5) & (ry_label < np.pi * 1.5)
            ry_label[opposite_flag] = (ry_label[opposite_flag] + np.pi) % (
                2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
            flag = ry_label > np.pi
            ry_label[flag] = ry_label[flag] - np.pi * 2  # (-pi/2, pi/2)
            ry_label = torch.clamp(ry_label, min=-np.pi / 2, max=np.pi / 2)
            pos_gt_bboxes_ct[..., 6] = ry_label

            rois_anchor = pos_bboxes.clone().detach()
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            bbox_targets = self.bbox_coder.encode(rois_anchor,
                                                  pos_gt_bboxes_ct)
        else:
            # no fg bbox
            bbox_targets = pos_gt_bboxes.new_empty((0, 7))

        return (label, bbox_targets, pos_gt_bboxes, reg_mask, label_weights,
                bbox_weights)

    def get_bboxes(self,
                   rois,
                   bbox_pred,
                   class_labels,
                   class_pred,
                   img_metas,
                   cfg=None):
        """Generate bboxes from bbox head predictions.

        Args:
            rois (torch.Tensor): Roi bounding boxes.
            bbox_pred (torch.Tensor): Bounding boxes predictions
            class_labels (torch.Tensor): Label of classes
            class_pred (torch.Tensor): Score for nms.
            img_metas (list[dict]): Point cloud and image's meta info.
            cfg (:obj:`ConfigDict`): Testing config.

        Returns:
            list[tuple]: Decoded bbox, scores and labels after nms.
        """
        roi_batch_id = rois[..., 0]
        roi_boxes = rois[..., 1:]  # boxes without batch id
        batch_size = int(roi_batch_id.max().item() + 1)

        # decode boxes
        roi_ry = roi_boxes[..., 6].view(-1)
        roi_xyz = roi_boxes[..., 0:3].view(-1, 3)
        local_roi_boxes = roi_boxes.clone().detach()
        local_roi_boxes[..., 0:3] = 0
        rcnn_boxes3d = self.bbox_coder.decode(local_roi_boxes, bbox_pred)
        rcnn_boxes3d[..., 0:3] = rotation_3d_in_axis(
            rcnn_boxes3d[..., 0:3].unsqueeze(1), roi_ry, axis=2).squeeze(1)
        rcnn_boxes3d[:, 0:3] += roi_xyz

        # post processing
        result_list = []
        for batch_id in range(batch_size):
            cur_class_labels = class_labels[batch_id]
            cur_cls_score = class_pred[roi_batch_id == batch_id].view(-1)

            cur_rcnn_boxes3d = rcnn_boxes3d[roi_batch_id == batch_id]
            selected = self.multi_class_nms(
                cur_cls_score.unsqueeze(1), cur_rcnn_boxes3d, cfg.score_thr,
                cfg.nms_thr, img_metas[batch_id], cfg.use_rotate_nms)
            selected_bboxes = cur_rcnn_boxes3d[selected]
            selected_label_preds = cur_class_labels[selected]
            selected_scores = cur_cls_score[selected]

            result_list.append((img_metas[batch_id]['box_type_3d'](
                selected_bboxes,
                self.bbox_coder.code_size,
                origin=(0.5, 0.5, 0.5)), selected_scores,
                                selected_label_preds))
        return result_list

    def multi_class_nms(self,
                        box_probs,
                        box_preds,
                        score_thr,
                        nms_thr,
                        input_meta,
                        use_rotate_nms=True):
        """Multi-class NMS for box head.

        Note:
            This function has large overlap with the `box3d_multiclass_nms`
            implemented in `mmdet3d.core.post_processing`. We are considering
            merging these two functions in the future.

        Args:
            box_probs (torch.Tensor): Predicted boxes probabitilies in
                shape (N,).
            box_preds (torch.Tensor): Predicted boxes in shape (N, 7+C).
            score_thr (float): Threshold of scores.
            nms_thr (float): Threshold for NMS.
            input_meta (dict): Meta informations of the current sample.
            use_rotate_nms (bool, optional): Whether to use rotated nms.
                Defaults to True.

        Returns:
            torch.Tensor: Selected indices.
        """
        if use_rotate_nms:
            nms_func = nms_gpu
        else:
            nms_func = nms_normal_gpu

        assert box_probs.shape[
            1] == self.num_classes, f'box_probs shape: {str(box_probs.shape)}'
        selected_list = []
        selected_labels = []
        boxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d'](
            box_preds, self.bbox_coder.code_size).bev)

        score_thresh = score_thr if isinstance(
            score_thr, list) else [score_thr for x in range(self.num_classes)]
        nms_thresh = nms_thr if isinstance(
            nms_thr, list) else [nms_thr for x in range(self.num_classes)]
        for k in range(0, self.num_classes):
            class_scores_keep = box_probs[:, k] >= score_thresh[k]

            if class_scores_keep.int().sum() > 0:
                original_idxs = class_scores_keep.nonzero(
                    as_tuple=False).view(-1)
                cur_boxes_for_nms = boxes_for_nms[class_scores_keep]
                cur_rank_scores = box_probs[class_scores_keep, k]

                cur_selected = nms_func(cur_boxes_for_nms, cur_rank_scores,
                                        nms_thresh[k])

                if cur_selected.shape[0] == 0:
                    continue
                selected_list.append(original_idxs[cur_selected])
                selected_labels.append(
                    torch.full([cur_selected.shape[0]],
                               k + 1,
                               dtype=torch.int64,
                               device=box_preds.device))

        selected = torch.cat(
            selected_list, dim=0) if len(selected_list) > 0 else []
        return selected
