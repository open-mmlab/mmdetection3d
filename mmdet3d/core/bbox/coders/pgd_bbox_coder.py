# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from torch.nn import functional as F

from mmdet.core.bbox.builder import BBOX_CODERS
from .fcos3d_bbox_coder import FCOS3DBBoxCoder


@BBOX_CODERS.register_module()
class PGDBBoxCoder(FCOS3DBBoxCoder):
    """Bounding box coder for PGD."""

    def encode(self, gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels):
        # TODO: refactor the encoder codes in the FCOS3D and PGD head
        pass

    def decode_2d(self,
                  bbox,
                  scale,
                  stride,
                  max_regress_range,
                  training,
                  pred_keypoints=False,
                  pred_bbox2d=True):
        """Decode regressed 2D attributes.

        Args:
            bbox (torch.Tensor): Raw bounding box predictions in shape
                [N, C, H, W].
            scale (tuple[`Scale`]): Learnable scale parameters.
            stride (int): Stride for a specific feature level.
            max_regress_range (int): Maximum regression range for a specific
                feature level.
            training (bool): Whether the decoding is in the training
                procedure.
            pred_keypoints (bool, optional): Whether to predict keypoints.
                Defaults to False.
            pred_bbox2d (bool, optional): Whether to predict 2D bounding
                boxes. Defaults to False.

        Returns:
            torch.Tensor: Decoded boxes.
        """
        clone_bbox = bbox.clone()
        if pred_keypoints:
            scale_kpts = scale[3]
            # 2 dimension of offsets x 8 corners of a 3D bbox
            bbox[:, self.bbox_code_size:self.bbox_code_size + 16] = \
                torch.tanh(scale_kpts(clone_bbox[
                    :, self.bbox_code_size:self.bbox_code_size + 16]).float())

        if pred_bbox2d:
            scale_bbox2d = scale[-1]
            # The last four dimensions are offsets to four sides of a 2D bbox
            bbox[:, -4:] = scale_bbox2d(clone_bbox[:, -4:]).float()

        if self.norm_on_bbox:
            if pred_bbox2d:
                bbox[:, -4:] = F.relu(bbox.clone()[:, -4:])
            if not training:
                if pred_keypoints:
                    bbox[
                        :, self.bbox_code_size:self.bbox_code_size + 16] *= \
                           max_regress_range
                if pred_bbox2d:
                    bbox[:, -4:] *= stride
        else:
            if pred_bbox2d:
                bbox[:, -4:] = bbox.clone()[:, -4:].exp()
        return bbox

    def decode_prob_depth(self, depth_cls_preds, depth_range, depth_unit,
                          division, num_depth_cls):
        """Decode probabilistic depth map.

        Args:
            depth_cls_preds (torch.Tensor): Depth probabilistic map in shape
                [..., self.num_depth_cls] (raw output before softmax).
            depth_range (tuple[float]): Range of depth estimation.
            depth_unit (int): Unit of depth range division.
            division (str): Depth division method. Options include 'uniform',
                'linear', 'log', 'loguniform'.
            num_depth_cls (int): Number of depth classes.

        Returns:
            torch.Tensor: Decoded probabilistic depth estimation.
        """
        if division == 'uniform':
            depth_multiplier = depth_unit * \
                depth_cls_preds.new_tensor(
                    list(range(num_depth_cls))).reshape([1, -1])
            prob_depth_preds = (F.softmax(depth_cls_preds.clone(), dim=-1) *
                                depth_multiplier).sum(dim=-1)
            return prob_depth_preds
        elif division == 'linear':
            split_pts = depth_cls_preds.new_tensor(list(
                range(num_depth_cls))).reshape([1, -1])
            depth_multiplier = depth_range[0] + (
                depth_range[1] - depth_range[0]) / \
                (num_depth_cls * (num_depth_cls - 1)) * \
                (split_pts * (split_pts+1))
            prob_depth_preds = (F.softmax(depth_cls_preds.clone(), dim=-1) *
                                depth_multiplier).sum(dim=-1)
            return prob_depth_preds
        elif division == 'log':
            split_pts = depth_cls_preds.new_tensor(list(
                range(num_depth_cls))).reshape([1, -1])
            start = max(depth_range[0], 1)
            end = depth_range[1]
            depth_multiplier = (np.log(start) +
                                split_pts * np.log(end / start) /
                                (num_depth_cls - 1)).exp()
            prob_depth_preds = (F.softmax(depth_cls_preds.clone(), dim=-1) *
                                depth_multiplier).sum(dim=-1)
            return prob_depth_preds
        elif division == 'loguniform':
            split_pts = depth_cls_preds.new_tensor(list(
                range(num_depth_cls))).reshape([1, -1])
            start = max(depth_range[0], 1)
            end = depth_range[1]
            log_multiplier = np.log(start) + \
                split_pts * np.log(end / start) / (num_depth_cls - 1)
            prob_depth_preds = (F.softmax(depth_cls_preds.clone(), dim=-1) *
                                log_multiplier).sum(dim=-1).exp()
            return prob_depth_preds
        else:
            raise NotImplementedError
