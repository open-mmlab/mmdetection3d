import mmengine
import numpy as np

from mmdet3d.structures import (Box3DMode, CameraInstance3DBoxes,
                                LiDARInstance3DBoxes)

val_infos = mmengine.load('data/waymo_mini/kitti_format/waymo_infos_val.pkl')
old_val_infos = mmengine.load(
    'data/waymo_mini/kitti_format/old_pkl/waymo_infos_val.pkl')
instance = np.array(
    val_infos['data_list'][0]['cam_sync_instances'][0]['bbox_3d'])[np.newaxis,
                                                                   ...]
instance = LiDARInstance3DBoxes(instance)
lidar2cam = np.array(
    val_infos['data_list'][0]['images']['CAM_FRONT']['lidar2cam'])
cam_instance = instance.convert_to(Box3DMode.CAM, lidar2cam)

old_instance = np.array(old_val_infos['data_list'][0]['cam_sync_instances'][0]
                        ['bbox_3d'])[np.newaxis, ...]

old_lidar2cam = np.array(
    old_val_infos['data_list'][0]['images']['CAM_FRONT']['lidar2cam'])
old_instance = CameraInstance3DBoxes(old_instance)
pass
