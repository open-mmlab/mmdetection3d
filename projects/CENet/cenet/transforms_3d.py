# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import numpy as np
from mmcv.transforms import BaseTransform

from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class SemkittiRangeView(BaseTransform):
    """Convert Semantickitti point cloud dataset to range image."""

    def __init__(self,
                 H: int = 64,
                 W: int = 2048,
                 fov_up: float = 3.0,
                 fov_down: float = -25.0,
                 means: Sequence[float] = (11.71279, -0.1023471, 0.4952,
                                           -1.0545, 0.2877),
                 stds: Sequence[float] = (10.24, 12.295865, 9.4287, 0.8643,
                                          0.1450),
                 ignore_index: int = 19) -> None:
        self.H = H
        self.W = W
        self.fov_up = fov_up / 180.0 * np.pi
        self.fov_down = fov_down / 180.0 * np.pi
        self.fov = abs(self.fov_down) + abs(self.fov_up)
        self.means = np.array(means, dtype=np.float32)
        self.stds = np.array(stds, dtype=np.float32)
        self.ignore_index = ignore_index

    def transform(self, results: dict) -> dict:
        points_numpy = results['points'].numpy()

        proj_image = np.full((self.H, self.W, 5), -1, dtype=np.float32)
        proj_idx = np.full((self.H, self.W), -1, dtype=np.int64)

        # get depth of all points
        depth = np.linalg.norm(points_numpy[:, :3], 2, axis=1)

        # get angles of all points
        yaw = -np.arctan2(points_numpy[:, 1], points_numpy[:, 0])
        pitch = np.arcsin(points_numpy[:, 2] / depth)

        # get projection in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)
        proj_y = 1.0 - (pitch + abs(self.fov_down)) / self.fov

        # scale to image size using angular resolution
        proj_x *= self.W
        proj_y *= self.H

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int64)

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int64)

        results['proj_x'] = proj_x
        results['proj_y'] = proj_y
        results['unproj_range'] = depth

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        proj_idx[proj_y[order], proj_x[order]] = indices[order]
        proj_image[proj_y[order], proj_x[order], 0] = depth[order]
        proj_image[proj_y[order], proj_x[order], 1:] = points_numpy[order]
        proj_mask = (proj_idx > 0).astype(np.int32)
        results['proj_range'] = proj_image[..., 0]

        proj_image = (proj_image -
                      self.means[None, None, :]) / self.stds[None, None, :]
        proj_image = proj_image * proj_mask[..., None].astype(np.float32)
        results['img'] = proj_image

        if 'pts_semantic_mask' in results:
            proj_sem_label = np.full((self.H, self.W),
                                     self.ignore_index,
                                     dtype=np.int64)
            proj_sem_label[proj_y[order],
                           proj_x[order]] = results['pts_semantic_mask'][order]
            results['gt_semantic_seg'] = proj_sem_label
        return results
