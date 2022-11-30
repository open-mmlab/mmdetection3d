# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional, Tuple, Union

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box

from mmdet3d.structures import Box3DMode, CameraInstance3DBoxes, points_cam2img
from mmdet3d.structures.ops import box_np_ops

kitti_categories = ('Pedestrian', 'Cyclist', 'Car', 'Van', 'Truck',
                    'Person_sitting', 'Tram', 'Misc')

waymo_categories = ('Car', 'Pedestrian', 'Cyclist')

nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')
NuScenesNameMapping = {
    'movable_object.barrier': 'barrier',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.car': 'car',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.motorcycle': 'motorcycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
    'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck'
}
LyftNameMapping = {
    'bicycle': 'bicycle',
    'bus': 'bus',
    'car': 'car',
    'emergency_vehicle': 'emergency_vehicle',
    'motorcycle': 'motorcycle',
    'other_vehicle': 'other_vehicle',
    'pedestrian': 'pedestrian',
    'truck': 'truck',
    'animal': 'animal'
}


def get_nuscenes_2d_boxes(nusc: NuScenes, sample_data_token: str,
                          visibilities: List[str]) -> List[dict]:
    """Get the 2d / mono3d annotation records for a given `sample_data_token`
    of nuscenes dataset.

    Args:
        nusc (:obj:`NuScenes`): NuScenes class.
        sample_data_token (str): Sample data token belonging to a camera
            keyframe.
        visibilities (List[str]): Visibility filter.

    Return:
        List[dict]: List of 2d annotation record that belongs to the input
        `sample_data_token`.
    """

    # Get the sample data and the sample corresponding to that sample data.
    sd_rec = nusc.get('sample_data', sample_data_token)

    assert sd_rec[
        'sensor_modality'] == 'camera', 'Error: get_2d_boxes only works' \
        ' for camera sample_data!'
    if not sd_rec['is_key_frame']:
        raise ValueError(
            'The 2D re-projections are available only for keyframes.')

    s_rec = nusc.get('sample', sd_rec['sample_token'])

    # Get the calibrated sensor and ego pose
    # record to get the transformation matrices.
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    # Get all the annotation with the specified visibilties.
    ann_recs = [
        nusc.get('sample_annotation', token) for token in s_rec['anns']
    ]
    ann_recs = [
        ann_rec for ann_rec in ann_recs
        if (ann_rec['visibility_token'] in visibilities)
    ]

    repro_recs = []

    for ann_rec in ann_recs:
        # Augment sample_annotation with token information.
        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token

        # Get the box in global coordinates.
        box = nusc.get_box(ann_rec['token'])

        # Move them to the ego-pose frame.
        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)

        # Move them to the calibrated sensor frame.
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        # Filter out the corners that are not in front of the calibrated
        # sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = view_points(corners_3d, camera_intrinsic,
                                    True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners
        # does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y,
                                    'nuscenes')

        # if repro_rec is None, we do not append it into repre_recs
        if repro_rec is not None:
            loc = box.center.tolist()

            dim = box.wlh
            dim[[0, 1, 2]] = dim[[1, 2, 0]]  # convert wlh to our lhw
            dim = dim.tolist()

            rot = box.orientation.yaw_pitch_roll[0]
            rot = [-rot]  # convert the rot to our cam coordinate

            global_velo2d = nusc.box_velocity(box.token)[:2]
            global_velo3d = np.array([*global_velo2d, 0.0])
            e2g_r_mat = Quaternion(pose_rec['rotation']).rotation_matrix
            c2e_r_mat = Quaternion(cs_rec['rotation']).rotation_matrix
            cam_velo3d = global_velo3d @ np.linalg.inv(
                e2g_r_mat).T @ np.linalg.inv(c2e_r_mat).T
            velo = cam_velo3d[0::2].tolist()

            repro_rec['bbox_3d'] = loc + dim + rot
            repro_rec['velocity'] = velo

            center_3d = np.array(loc).reshape([1, 3])
            center_2d_with_depth = points_cam2img(
                center_3d, camera_intrinsic, with_depth=True)
            center_2d_with_depth = center_2d_with_depth.squeeze().tolist()
            repro_rec['center_2d'] = center_2d_with_depth[:2]
            repro_rec['depth'] = center_2d_with_depth[2]
            # normalized center2D + depth
            # if samples with depth < 0 will be removed
            if repro_rec['depth'] <= 0:
                continue

            ann_token = nusc.get('sample_annotation',
                                 box.token)['attribute_tokens']
            if len(ann_token) == 0:
                attr_name = 'None'
            else:
                attr_name = nusc.get('attribute', ann_token[0])['name']
            attr_id = nus_attributes.index(attr_name)
            # repro_rec['attribute_name'] = attr_name
            repro_rec['attr_label'] = attr_id

            repro_recs.append(repro_rec)

    return repro_recs


def get_kitti_style_2d_boxes(info: dict,
                             cam_idx: int = 2,
                             occluded: Tuple[int] = (0, 1, 2, 3),
                             annos: Optional[dict] = None,
                             mono3d: bool = True,
                             dataset: str = 'kitti') -> List[dict]:
    """Get the 2d / mono3d annotation records for a given info.

    This function is used to get 2D/Mono3D annotations when loading annotations
    from a kitti-style dataset class, such as KITTI and Waymo dataset.

    Args:
        info (dict): Information of the given sample data.
        cam_idx (int): Camera id which the 2d / mono3d annotations to obtain
            belong to. In KITTI, typically only CAM 2 will be used,
            and in Waymo, multi cameras could be used.
            Defaults to 2.
        occluded (Tuple[int]): Integer (0, 1, 2, 3) indicating occlusion state:
            0 = fully visible, 1 = partly occluded, 2 = largely occluded,
            3 = unknown, -1 = DontCare.
            Defaults to (0, 1, 2, 3).
        annos (dict, optional): Original annotations. Defaults to None.
        mono3d (bool): Whether to get boxes with mono3d annotation.
            Defaults to True.
        dataset (str): Dataset name of getting 2d bboxes.
            Defaults to 'kitti'.

    Return:
        List[dict]: List of 2d / mono3d annotation record that
        belongs to the input camera id.
    """
    # Get calibration information
    camera_intrinsic = info['calib'][f'P{cam_idx}']

    repro_recs = []
    # if no annotations in info (test dataset), then return
    if annos is None:
        return repro_recs

    # Get all the annotation with the specified visibilties.
    # filter the annotation bboxes by occluded attributes
    ann_dicts = annos
    mask = [(ocld in occluded) for ocld in ann_dicts['occluded']]
    for k in ann_dicts.keys():
        ann_dicts[k] = ann_dicts[k][mask]

    # convert dict of list to list of dict
    ann_recs = []
    for i in range(len(ann_dicts['occluded'])):
        ann_rec = {}
        for k in ann_dicts.keys():
            ann_rec[k] = ann_dicts[k][i]
        ann_recs.append(ann_rec)

    for ann_idx, ann_rec in enumerate(ann_recs):
        # Augment sample_annotation with token information.
        ann_rec['sample_annotation_token'] = \
            f"{info['image']['image_idx']}.{ann_idx}"
        ann_rec['sample_data_token'] = info['image']['image_idx']

        loc = ann_rec['location'][np.newaxis, :]
        dim = ann_rec['dimensions'][np.newaxis, :]
        rot = ann_rec['rotation_y'][np.newaxis, np.newaxis]

        # transform the center from [0.5, 1.0, 0.5] to [0.5, 0.5, 0.5]
        dst = np.array([0.5, 0.5, 0.5])
        src = np.array([0.5, 1.0, 0.5])
        loc = loc + dim * (dst - src)
        loc_3d = np.copy(loc)
        gt_bbox_3d = np.concatenate([loc, dim, rot], axis=1).astype(np.float32)

        # Filter out the corners that are not in front of the calibrated
        # sensor.
        corners_3d = box_np_ops.center_to_corner_box3d(
            gt_bbox_3d[:, :3],
            gt_bbox_3d[:, 3:6],
            gt_bbox_3d[:, 6], [0.5, 0.5, 0.5],
            axis=1)
        corners_3d = corners_3d[0].T  # (1, 8, 3) -> (3, 8)
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = view_points(corners_3d, camera_intrinsic,
                                    True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(
            corner_coords,
            imsize=(info['image']['image_shape'][1],
                    info['image']['image_shape'][0]))

        # Skip if the convex hull of the re-projected corners
        # does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y,
                                    dataset)

        # If mono3d=True, add 3D annotations in camera coordinates
        if mono3d and (repro_rec is not None):
            repro_rec['bbox_3d'] = np.concatenate(
                [loc_3d, dim, rot],
                axis=1).astype(np.float32).squeeze().tolist()
            repro_rec['velocity'] = -1  # no velocity in KITTI

            center_3d = np.array(loc).reshape([1, 3])
            center_2d_with_depth = points_cam2img(
                center_3d, camera_intrinsic, with_depth=True)
            center_2d_with_depth = center_2d_with_depth.squeeze().tolist()

            repro_rec['center_2d'] = center_2d_with_depth[:2]
            repro_rec['depth'] = center_2d_with_depth[2]
            # normalized center2D + depth
            # samples with depth < 0 will be removed
            if repro_rec['depth'] <= 0:
                continue
            repro_recs.append(repro_rec)

    return repro_recs


def convert_annos(info: dict, cam_idx: int) -> dict:
    """Convert front-cam anns to i-th camera (KITTI-style info)."""
    rect = info['calib']['R0_rect'].astype(np.float32)
    lidar2cam0 = info['calib']['Tr_velo_to_cam'].astype(np.float32)
    lidar2cami = info['calib'][f'Tr_velo_to_cam{cam_idx}'].astype(np.float32)
    annos = info['annos']
    converted_annos = copy.deepcopy(annos)
    loc = annos['location']
    dims = annos['dimensions']
    rots = annos['rotation_y']
    gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                  axis=1).astype(np.float32)
    # convert gt_bboxes_3d to velodyne coordinates
    gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(
        Box3DMode.LIDAR, np.linalg.inv(rect @ lidar2cam0), correct_yaw=True)
    # convert gt_bboxes_3d to cam coordinates
    gt_bboxes_3d = gt_bboxes_3d.convert_to(
        Box3DMode.CAM, rect @ lidar2cami, correct_yaw=True).tensor.numpy()
    converted_annos['location'] = gt_bboxes_3d[:, :3]
    converted_annos['dimensions'] = gt_bboxes_3d[:, 3:6]
    converted_annos['rotation_y'] = gt_bboxes_3d[:, 6]
    return converted_annos


def post_process_coords(
    corner_coords: List[int], imsize: Tuple[int] = (1600, 900)
) -> Union[Tuple[float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (List[int]): Corner coordinates of reprojected
            bounding box.
        imsize (Tuple[int]): Size of the image canvas.
            Defaults to (1600, 900).

    Return:
        Tuple[float] or None: Intersection of the convex hull of the 2D box
        corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(ann_rec: dict, x1: float, y1: float, x2: float, y2: float,
                    dataset: str) -> Union[dict, None]:
    """Generate one 2D annotation record given various information on top of
    the 2D bounding box coordinates.

    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        dataset (str): Name of dataset.

    Returns:
        dict or None: A sample 2d annotation record.

            - bbox_label (int): 2d box label id
            - bbox_label_3d (int): 3d box label id
            - bbox (List[float]): left x, top y, right x, bottom y of 2d box
            - bbox_3d_isvalid (bool): whether the box is valid
    """

    if dataset == 'nuscenes':
        cat_name = ann_rec['category_name']
        if cat_name not in NuScenesNameMapping:
            return None
        else:
            cat_name = NuScenesNameMapping[cat_name]
            categories = nus_categories
    else:
        if dataset == 'kitti':
            categories = kitti_categories
        elif dataset == 'waymo':
            categories = waymo_categories
        else:
            raise NotImplementedError('Unsupported dataset!')

        cat_name = ann_rec['name']
        if cat_name not in categories:
            return None

    rec = dict()
    rec['bbox_label'] = categories.index(cat_name)
    rec['bbox_label_3d'] = rec['bbox_label']
    rec['bbox'] = [x1, y1, x2, y2]
    rec['bbox_3d_isvalid'] = True

    return rec
