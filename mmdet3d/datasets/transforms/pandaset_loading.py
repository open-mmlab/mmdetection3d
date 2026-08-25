# Copyright (c) OpenMMLab. All rights reserved.
"""On-the-fly PandaSet point cloud loading transform for MMDetection3D.

Reads the original PandaSet .pkl files directly, applies ego transform, filters
by sensor ID, and returns points in the standard format expected by the
PointPillars (or any LiDAR-based) model.
"""

import numpy as np
import pandas as pd
from mmcv.transforms import BaseTransform

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.points import get_points_type


@TRANSFORMS.register_module()
class LoadPointsFromPandaSet(BaseTransform):
    """Load points from PandaSet .pkl files with on-the-fly ego transform.

    Required Keys:
        - lidar_points (dict):
            - lidar_path (str): path to the .pkl file
            - pandaset_sensor_id (int): 0=Pandar64, 1=PandarGT
            - pandaset_pose (list): 4x4 world-to-ego pose matrix

    Added Keys:
        - points (BasePoints): transformed point cloud

    Args:
        coord_type (str): Coordinate type ('LIDAR').
        use_dim (int): Number of point dimensions to use. Default 4 (x,y,z,i).
    """

    def __init__(self, coord_type='LIDAR', use_dim=4):
        self.coord_type = coord_type
        self.use_dim = use_dim

    def transform(self, results: dict) -> dict:
        lidar_info = results['lidar_points']
        pkl_path = lidar_info['lidar_path']
        sensor_id = lidar_info['pandaset_sensor_id']
        pose_mat = np.array(lidar_info['pandaset_pose'])

        lidar_df = pd.read_pickle(pkl_path)
        sensor_pts = lidar_df[lidar_df['d'] == sensor_id]

        world_xyz = sensor_pts[['x', 'y', 'z']].values.astype(np.float64)
        intensity = sensor_pts['i'].values.astype(np.float32) / 255.0

        # Transform from world frame to ego/lidar frame
        inv_pose = np.linalg.inv(pose_mat)
        pts_h = np.hstack([world_xyz, np.ones((len(world_xyz), 1))])
        ego_xyz = (inv_pose @ pts_h.T).T[:, :3].astype(np.float32)

        points = np.column_stack([ego_xyz, intensity])

        if self.use_dim < 4:
            points = points[:, :self.use_dim]

        points_class = get_points_type(self.coord_type)
        points = points_class(points, points_dim=points.shape[-1])
        results['points'] = points

        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'coord_type={self.coord_type}, use_dim={self.use_dim})')


@TRANSFORMS.register_module()
class NormalizePointsIntensity(BaseTransform):
    """Normalize the intensity channel of loaded points.

    The nuScenes PointPillars model uses timestamp (≈0) as its 4th channel,
    not intensity. When feeding PandaSet .bin files (intensity in [0, 255]),
    we normalize to [0, 1] to keep values near zero and compatible with the
    model's learned batch normalization statistics.

    Args:
        intensity_dim (int): Index of the intensity channel. Default 3.
        scale (float): Divisor for normalization. Default 255.0.
    """

    def __init__(self, intensity_dim=3, scale=255.0):
        self.intensity_dim = intensity_dim
        self.scale = scale

    def transform(self, results: dict) -> dict:
        points = results['points']
        points.tensor[:, self.intensity_dim] /= self.scale
        results['points'] = points
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'intensity_dim={self.intensity_dim}, scale={self.scale})')
