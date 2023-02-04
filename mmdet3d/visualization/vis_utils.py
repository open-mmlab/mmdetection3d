# Copyright (c) OpenMMLab. All rights reserved.
import copy

import numpy as np
import torch
import trimesh

from mmdet3d.structures import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                                DepthInstance3DBoxes, LiDARInstance3DBoxes)


def write_obj(points, out_filename):
    """Write points into ``obj`` format for meshlab visualization.

    Args:
        points (np.ndarray): Points in shape (N, dim).
        out_filename (str): Filename to be saved.
    """
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        if points.shape[1] == 6:
            c = points[i, 3:].astype(int)
            fout.write(
                'v %f %f %f %d %d %d\n' %
                (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))

        else:
            fout.write('v %f %f %f\n' %
                       (points[i, 0], points[i, 1], points[i, 2]))
    fout.close()


def write_oriented_bbox(scene_bbox, out_filename):
    """Export oriented (around Z axis) scene bbox to meshes.

    Args:
        scene_bbox(list[ndarray] or ndarray): xyz pos of center and
            3 lengths (x_size, y_size, z_size) and heading angle around Z axis.
            Y forward, X right, Z upward. heading angle of positive X is 0,
            heading angle of positive Y is 90 degrees.
        out_filename(str): Filename.
    """

    def heading2rotmat(heading_angle):
        rotmat = np.zeros((3, 3))
        rotmat[2, 2] = 1
        cosval = np.cos(heading_angle)
        sinval = np.sin(heading_angle)
        rotmat[0:2, 0:2] = np.array([[cosval, -sinval], [sinval, cosval]])
        return rotmat

    def convert_oriented_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:6]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3, 3] = 1.0
        trns[0:3, 0:3] = heading2rotmat(box[6])
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    if len(scene_bbox) == 0:
        scene_bbox = np.zeros((1, 7))
    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box))

    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to obj file
    trimesh.io.export.export_mesh(mesh_list, out_filename, file_type='obj')

    return


def to_depth_mode(points, bboxes):
    """Convert points and bboxes to Depth Coord and Depth Box mode."""
    if points is not None:
        points = Coord3DMode.convert_point(points.copy(), Coord3DMode.LIDAR,
                                           Coord3DMode.DEPTH)
    if bboxes is not None:
        bboxes = Box3DMode.convert(bboxes.clone(), Box3DMode.LIDAR,
                                   Box3DMode.DEPTH)
    return points, bboxes


# TODO: refactor lidar2img to img_meta
def proj_lidar_bbox3d_to_img(bboxes_3d: LiDARInstance3DBoxes,
                             input_meta: dict) -> np.array:
    """Project the 3D bbox on 2D plane.

    Args:
        bboxes_3d (:obj:`LiDARInstance3DBoxes`):
            3d bbox in lidar coordinate system to visualize.
        lidar2img (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
    """
    corners_3d = bboxes_3d.corners
    num_bbox = corners_3d.shape[0]
    pts_4d = np.concatenate(
        [corners_3d.reshape(-1, 3),
         np.ones((num_bbox * 8, 1))], axis=-1)
    lidar2img = copy.deepcopy(input_meta['lidar2img']).reshape(4, 4)
    if isinstance(lidar2img, torch.Tensor):
        lidar2img = lidar2img.cpu().numpy()
    pts_2d = pts_4d @ lidar2img.T

    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

    return imgfov_pts_2d


# TODO: remove third parameter in all functions here in favour of img_metas
def proj_depth_bbox3d_to_img(bboxes_3d: DepthInstance3DBoxes,
                             input_meta: dict) -> np.array:
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes_3d (:obj:`DepthInstance3DBoxes`, shape=[M, 7]):
            3d bbox in depth coordinate system to visualize.
        input_meta (dict): Used in coordinates transformation.
    """
    from mmdet3d.models import apply_3d_transformation
    from mmdet3d.structures import points_cam2img

    input_meta = copy.deepcopy(input_meta)
    corners_3d = bboxes_3d.corners
    num_bbox = corners_3d.shape[0]
    points_3d = corners_3d.reshape(-1, 3)

    # first reverse the data transformations
    xyz_depth = apply_3d_transformation(
        points_3d, 'DEPTH', input_meta, reverse=True)

    # project to 2d to get image coords (uv)
    uv_origin = points_cam2img(xyz_depth,
                               xyz_depth.new_tensor(input_meta['depth2img']))
    uv_origin = (uv_origin - 1).round()
    imgfov_pts_2d = uv_origin[..., :2].reshape(num_bbox, 8, 2).numpy()

    return imgfov_pts_2d


# project the camera bboxes 3d to image
def proj_camera_bbox3d_to_img(bboxes_3d: CameraInstance3DBoxes,
                              input_meta: dict) -> np.array:
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes_3d (:obj:`CameraInstance3DBoxes`, shape=[M, 7]):
            3d bbox in camera coordinate system to visualize.
        cam2img (np.array)): Camera intrinsic matrix,
            denoted as `K` in depth bbox coordinate system.
    """
    from mmdet3d.structures import points_cam2img

    cam2img = copy.deepcopy(input_meta['cam2img'])
    corners_3d = bboxes_3d.corners
    num_bbox = corners_3d.shape[0]
    points_3d = corners_3d.reshape(-1, 3)
    if not isinstance(cam2img, torch.Tensor):
        cam2img = torch.from_numpy(np.array(cam2img))

    assert (cam2img.shape == torch.Size([3, 3])
            or cam2img.shape == torch.Size([4, 4]))
    cam2img = cam2img.float().cpu()

    # project to 2d to get image coords (uv)
    uv_origin = points_cam2img(points_3d, cam2img)
    uv_origin = (uv_origin - 1).round()
    imgfov_pts_2d = uv_origin[..., :2].reshape(num_bbox, 8, 2).numpy()

    return imgfov_pts_2d
