# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple

import torch
from mmdet.models.task_modules import BaseBBoxCoder
from torch import Tensor

from mmdet3d.registry import TASK_UTILS


@TASK_UTILS.register_module()
class VoxelNeXtBBoxCoder(BaseBBoxCoder):
    """Bbox coder for CenterPoint.

    Args:
        pc_range (list[float]): Range of point cloud.
        out_size_factor (int): Downsample factor of the model.
        voxel_size (list[float]): Size of voxel.
        post_center_range (list[float], optional): Limit of the center.
            Default: None.
        max_num (int, optional): Max number to be kept. Default: 100.
        score_threshold (float, optional): Threshold to filter boxes
            based on score. Default: None.
        code_size (int, optional): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range: List[float],
                 out_size_factor: int,
                 voxel_size: List[float],
                 post_center_range: Optional[List[float]] = None,
                 max_num: int = 100,
                 score_threshold: Optional[float] = None,
                 code_size: int = 9) -> None:

        self.pc_range = pc_range
        self.out_size_factor = out_size_factor
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.code_size = code_size

    def gather_feat_idx(self, feats, inds, batch_size, batch_idx):
        feats_list = []
        dim = feats.size(-1)
        _inds = inds.unsqueeze(-1).expand(inds.size(0), inds.size(1), dim)

        for bs_idx in range(batch_size):
            batch_inds = batch_idx==bs_idx
            feat = feats[batch_inds]
            feats_list.append(feat.gather(0, _inds[bs_idx]))
        feats = torch.stack(feats_list)
        return feats

    def _topk_1d(self, scores, batch_size, batch_idx, obj, K=40, nuscenes=False):
        # scores: (N, num_classes)
        topk_score_list = []
        topk_inds_list = []
        topk_classes_list = []

        for bs_idx in range(batch_size):
            batch_inds = batch_idx==bs_idx
            if obj.shape[-1] == 1 and not nuscenes:
                score = scores[batch_inds].permute(1, 0)
                topk_scores, topk_inds = torch.topk(score, K)
                topk_score, topk_ind = torch.topk(obj[topk_inds.view(-1)].squeeze(-1), K) #torch.topk(topk_scores.view(-1), K)
            else:
                score = obj[batch_inds].permute(1, 0)
                topk_scores, topk_inds = torch.topk(score, min(K, score.shape[-1]))
                topk_score, topk_ind = torch.topk(topk_scores.view(-1), min(K, topk_scores.view(-1).shape[-1]))
                #topk_score, topk_ind = torch.topk(score.reshape(-1), K)

            topk_classes = (topk_ind // K).int()
            topk_inds = topk_inds.view(-1).gather(0, topk_ind)
            #print('topk_inds', topk_inds)

            if not obj is None and obj.shape[-1] == 1:
                topk_score_list.append(obj[batch_inds][topk_inds])
            else:
                topk_score_list.append(topk_score)
            topk_inds_list.append(topk_inds)
            topk_classes_list.append(topk_classes)

        topk_score = torch.stack(topk_score_list)
        topk_inds = torch.stack(topk_inds_list)
        topk_classes = torch.stack(topk_classes_list)

        return topk_score, topk_inds, topk_classes

    def encode(self):
        pass

    def decode(self,
               batch_size, 
               indices, 
               obj: Tensor,
               rot_sine: Tensor,
               rot_cosine: Tensor,
               hei: Tensor, 
               dim: Tensor, 
               vel: Tensor,
               iou: Tensor,
               reg: Optional[Tensor] = None, 
               add_features = None,
               task_id: int = -1) -> List[Dict[str, Tensor]]:
        """Decode bboxes.

        Args:
            heat (torch.Tensor): Heatmap with the shape of [B, N, W, H].
            rot_sine (torch.Tensor): Sine of rotation with the shape of
                [B, 1, W, H].
            rot_cosine (torch.Tensor): Cosine of rotation with the shape of
                [B, 1, W, H].
            hei (torch.Tensor): Height of the boxes with the shape
                of [B, 1, W, H].
            dim (torch.Tensor): Dim of the boxes with the shape of
                [B, 1, W, H].
            vel (torch.Tensor): Velocity with the shape of [B, 1, W, H].
            reg (torch.Tensor, optional): Regression value of the boxes in
                2D with the shape of [B, 2, W, H]. Default: None.
            task_id (int, optional): Index of task. Default: -1.

        Returns:
            list[dict]: Decoded boxes.
        """
    
        """
        K=self.max_num
        point_cloud_range即 self.pc_range
        voxel_size 即  self.voxel_size
        feature_map_stride -> self.out_size_factor
        score_thresh - > self.score_threshold
        post_center_limit_range -> self.post_center_range
        """

        batch_idx = indices[:, 0]
        spatial_indices = indices[:, 1:]
        scores, inds, class_ids = self._topk_1d(None, batch_size, batch_idx, obj, K=self.max_num, nuscenes=True)
        feature_map_stride = self.out_size_factor

        if reg is not None:
            reg = self.gather_feat_idx(reg, inds, batch_size, batch_idx)

        # rotation value and direction label
        rot_sin = self.gather_feat_idx(rot_sine, inds, batch_size, batch_idx)

        rot_cos = self.gather_feat_idx(rot_cosine, inds, batch_size, batch_idx)
        rot = torch.atan2(rot_sin, rot_cos) #angle

        # height in the bev
        hei = self.gather_feat_idx(hei, inds, batch_size, batch_idx)

        # dim of the box
        dim = self.gather_feat_idx(dim, inds, batch_size, batch_idx)

        #
        spatial_indices = self.gather_feat_idx(spatial_indices, inds, batch_size, batch_idx)

        if not add_features is None:
            add_features = [self.gather_feat_idx(add_feature, inds, batch_size, batch_idx) for add_feature in add_features]

        if not isinstance(feature_map_stride, int):
            feature_map_stride = self.gather_feat_idx(feature_map_stride.unsqueeze(-1), inds, batch_size, batch_idx)

        xs = (spatial_indices[:, :, -1:] + reg[:, :, 0:1]) * feature_map_stride * self.voxel_size[0] + self.pc_range[0]
        ys = (spatial_indices[:, :, -2:-1] + reg[:, :, 1:2]) * feature_map_stride * self.voxel_size[1] + self.pc_range[1]

        if vel is None:  # KITTI FORMAT
            final_box_preds = torch.cat([xs, ys, hei, dim, rot], dim=2)
        else:  # exist velocity, nuscene format
            vel = self.gather_feat_idx(vel, inds, batch_size, batch_idx)
            final_box_preds = torch.cat([xs, ys, hei, dim, rot, vel], dim=2)

        if not iou is None:
            iou = self.gather_feat_idx(iou, inds, batch_size, batch_idx)
            iou = torch.clamp(iou, min=0, max=1.)

        # class label
        class_ids = class_ids.view(batch_size, self.max_num).float()
        scores = scores.view(batch_size, self.max_num)

        final_scores = scores
        final_preds = class_ids

        if not add_features is None:
            add_features = [add_feature.view(batch_size, self.max_num, add_feature.shape[-1]) for add_feature in add_features]

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=scores.device) 
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(2)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(2)

            predictions_dicts = []
            for i in range(batch_size):
                cmask = mask[i, :]
                if self.score_threshold:
                    cmask &= thresh_mask[i]

                boxes3d = final_box_preds[i, cmask]
                scores = final_scores[i, cmask]
                labels = final_preds[i, cmask]
                add_features = [add_feature[self.max_num, cmask] for add_feature in add_features] if not add_features is None else None
                iou = iou[self.max_num, cmask] if not iou is None else None
                predictions_dict = {
                    'bboxes': boxes3d,
                    'scores': scores,
                    'labels': labels,
                    'ious': iou,
                    'features': add_features,
                }

                predictions_dicts.append(predictions_dict)
        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')

        return predictions_dicts
