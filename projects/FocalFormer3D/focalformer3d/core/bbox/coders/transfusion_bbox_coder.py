# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates.
# All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FocalFormer3D/blob/main/LICENSE

import torch
# --- Base class import ---
from mmdet.models.task_modules.coders import BaseBBoxCoder

from mmdet3d.registry import TASK_UTILS


@TASK_UTILS.register_module()
class TransFusionBBoxCoder(BaseBBoxCoder):

    def __init__(self,
                 pc_range,
                 out_size_factor,
                 voxel_size,
                 post_center_range=None,
                 score_threshold=None,
                 code_size=8):
        super().__init__()
        self.pc_range = pc_range
        self.out_size_factor = out_size_factor
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.score_threshold = score_threshold
        self.code_size = code_size

    def encode(self, dst_boxes):

        targets = torch.zeros([dst_boxes.shape[0],
                               self.code_size]).to(dst_boxes.device)
        targets[:, 0] = ((dst_boxes[:, 0] - self.pc_range[0]) /
                         (self.out_size_factor * self.voxel_size[0]))
        targets[:, 1] = ((dst_boxes[:, 1] - self.pc_range[1]) /
                         (self.out_size_factor * self.voxel_size[1]))
        targets[:, 3] = (dst_boxes[:, 3] + 1e-6).log()
        targets[:, 4] = (dst_boxes[:, 4] + 1e-6).log()
        targets[:, 5] = (dst_boxes[:, 5] + 1e-6).log()
        # bottom center to gravity center
        targets[:, 2] = dst_boxes[:, 2] + dst_boxes[:, 5] * 0.5
        targets[:, 6] = torch.sin(dst_boxes[:, 6])
        targets[:, 7] = torch.cos(dst_boxes[:, 6])
        if self.code_size == 10:
            targets[:, 8:10] = dst_boxes[:, 7:]

        return targets

    def encode_center(self, center):
        assert center.shape[1] == 2, \
            (f'[TransFusionBBoxCoder.encode_center] Expected '
             f'center.shape[1]==2, got {center.shape[1]}')

        center = center.clone()
        center[:, 0] = ((center[:, 0] - self.pc_range[0]) /
                        (self.out_size_factor * self.voxel_size[0]))
        center[:, 1] = ((center[:, 1] - self.pc_range[1]) /
                        (self.out_size_factor * self.voxel_size[1]))
        return center

    def decode_center(self, center):
        assert center.shape[1] == 2, \
            (f'[TransFusionBBoxCoder.decode_center] Expected '
             f'center.shape[1]==2, got {center.shape[1]}')

        center = center.clone()
        center[:, 0, :] = (
            center[:, 0, :] * self.out_size_factor * self.voxel_size[0] +
            self.pc_range[0])
        center[:, 1, :] = (
            center[:, 1, :] * self.out_size_factor * self.voxel_size[1] +
            self.pc_range[1])
        return center

    def decode_box(self, rot, dim, center, height, vel):

        # change size to real world metric
        center[:, 0, :] = (
            center[:, 0, :] * self.out_size_factor * self.voxel_size[0] +
            self.pc_range[0])
        center[:, 1, :] = (
            center[:, 1, :] * self.out_size_factor * self.voxel_size[1] +
            self.pc_range[1])
        dim[:, 0, :] = dim[:, 0, :].exp()
        dim[:, 1, :] = dim[:, 1, :].exp()
        dim[:, 2, :] = dim[:, 2, :].exp()
        # gravity center to bottom center
        height = height - dim[:, 2:3, :] * 0.5
        rots, rotc = rot[:, 0:1, :], rot[:, 1:2, :]
        rot = torch.atan2(rots, rotc)

        if vel is None:
            final_box_preds = torch.cat([center, height, dim, rot],
                                        dim=1).permute(0, 2, 1)
        else:
            final_box_preds = torch.cat([center, height, dim, rot, vel],
                                        dim=1).permute(0, 2, 1)

        return final_box_preds

    def decode(self, heatmap, rot, dim, center, height, vel, filter=False):
        """Decode bboxes.

        Args:
            heatmap (torch.Tensor): [B, num_cls, num_proposals].
            rot (torch.Tensor): [B, 1, num_proposals].
            dim (torch.Tensor): [B, 3, num_proposals].
            center (torch.Tensor): [B, 2, num_proposals] (feature map metric).
            height (torch.Tensor): [B, 2, num_proposals] (real world metric).
            vel (torch.Tensor): [B, 2, num_proposals].
            filter: if False, return all boxes without score/range filtering.

        Returns:
            list[dict]: Decoded boxes.
        """

        # class label
        final_preds = heatmap.max(1, keepdims=False).indices
        final_scores = heatmap.max(1, keepdims=False).values

        # change size to real world metric
        center[:, 0, :] = (
            center[:, 0, :] * self.out_size_factor * self.voxel_size[0] +
            self.pc_range[0])
        center[:, 1, :] = (
            center[:, 1, :] * self.out_size_factor * self.voxel_size[1] +
            self.pc_range[1])
        dim[:, 0, :] = dim[:, 0, :].exp()
        dim[:, 1, :] = dim[:, 1, :].exp()
        dim[:, 2, :] = dim[:, 2, :].exp()
        # gravity center to bottom center
        height = height - dim[:, 2:3, :] * 0.5
        rots, rotc = rot[:, 0:1, :], rot[:, 1:2, :]
        rot = torch.atan2(rots, rotc)

        if vel is None:
            final_box_preds = torch.cat([center, height, dim, rot],
                                        dim=1).permute(0, 2, 1)
        else:
            final_box_preds = torch.cat([center, height, dim, rot, vel],
                                        dim=1).permute(0, 2, 1)

        predictions_dicts = []
        for i in range(heatmap.shape[0]):
            boxes3d = final_box_preds[i]
            scores = final_scores[i]
            labels = final_preds[i]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels
            }
            predictions_dicts.append(predictions_dict)

        if filter is False:
            return predictions_dicts

        # --- Filtered path ---

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=heatmap.device)
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(2)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(2)

            predictions_dicts = []
            for i in range(heatmap.shape[0]):
                cmask = mask[i, :]
                if self.score_threshold:
                    cmask &= thresh_mask[i]

                boxes3d = final_box_preds[i, cmask]
                scores = final_scores[i, cmask]
                labels = final_preds[i, cmask]
                predictions_dict = {
                    'bboxes': boxes3d,
                    'scores': scores,
                    'labels': labels
                }
                predictions_dicts.append(predictions_dict)
        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')

        return predictions_dicts
