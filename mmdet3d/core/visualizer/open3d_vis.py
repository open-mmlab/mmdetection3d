import cv2
import numpy as np
import open3d as o3d
import torch
from matplotlib import pyplot as plt
from open3d import geometry


def _draw_points(points,
                 vis,
                 points_size=2,
                 point_color=(0.5, 0.5, 0.5),
                 mode='xyz'):
    """Draw points on visualizer.

    Args:
        points (numpy.array | torch.tensor, shape=[N, 3+C]):
            points to visualize.
        vis (:obj:`open3d.visualization.Visualizer`): open3d visualizer.
        points_size (int): the size of points to show on visualizer.
        point_color (tuple[float]): the color of points.
    Returns:
        tuple: points, color of each point.
    """
    vis.get_render_option().point_size = points_size  # set points size
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    points = points.copy()
    pcd = geometry.PointCloud()
    if mode == 'xyz':
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        points_colors = np.tile(np.array(point_color), (points.shape[0], 1))
    elif mode == 'xyzrgb':
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        points_colors = points[:, 3:6]
    else:
        raise NotImplementedError

    pcd.colors = o3d.utility.Vector3dVector(points_colors)
    vis.add_geometry(pcd)

    return pcd, points_colors


def _draw_bboxes(bbox3d,
                 vis,
                 points_colors,
                 pcd=None,
                 bbox_color=(0, 1, 0),
                 points_in_box_color=(1, 0, 0),
                 rot_axis=2,
                 center_mode='lidar_bottom',
                 mode='xyz'):
    """Draw bbox and points in bbox on visualizer.

    Args:
        bbox3d (numpy.array | torch.tensor, shape=[M, 7]):
            3d bbox (x, y, z, dx, dy, dz, yaw) to visualize.
        vis (:obj:`open3d.visualization.Visualizer`): open3d visualizer.
        points_colors (numpy.array): color of each points.
        pcd (:obj:`open3d.geometry.PointCloud`): point cloud.
        bbox_color (tuple[float]): the color of bbox.
        points_in_box_color (tuple[float]):
            the color of points which are in bbox3d.
        rot_axis (int): rotation axis of bbox.
        bottom_center (bool): indicate the center of bbox is
            bottom center or gravity center.
    """
    if isinstance(bbox3d, torch.Tensor):
        bbox3d = bbox3d.cpu().numpy()
    bbox3d = bbox3d.copy()

    in_box_color = np.array(points_in_box_color)
    for i in range(len(bbox3d)):
        center = bbox3d[i, 0:3]
        dim = bbox3d[i, 3:6]
        yaw = np.zeros(3)
        # TODO: fix problem of current coordinate system
        # dim[0], dim[1] = dim[1], dim[0]  # for current coordinate
        # yaw[rot_axis] = -(bbox3d[i, 6] - 0.5 * np.pi)
        yaw[rot_axis] = bbox3d[i, 6]
        rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)
        if center_mode == 'lidar_bottom':
            center[rot_axis] += dim[
                rot_axis] / 2  # bottom center to gravity center
        elif center_mode == 'camera_bottom':
            center[rot_axis] -= dim[
                rot_axis] / 2  # bottom center to gravity center
        box3d = geometry.OrientedBoundingBox(center, rot_mat, dim)

        line_set = geometry.LineSet.create_from_oriented_bounding_box(box3d)
        line_set.paint_uniform_color(bbox_color)
        # draw bboxes on visualizer
        vis.add_geometry(line_set)

        # change the color of points which are in box
        if pcd is not None and mode == 'xyz':
            indices = box3d.get_point_indices_within_bounding_box(pcd.points)
            points_colors[indices] = in_box_color

    # update points colors
    pcd.colors = o3d.utility.Vector3dVector(points_colors)
    vis.update_geometry(pcd)


def show_pts_boxes(points,
                   bbox3d=None,
                   show=True,
                   mode='xyz',
                   save_path=None,
                   points_size=2,
                   point_color=(0.5, 0.5, 0.5),
                   bbox_color=(0, 1, 0),
                   points_in_box_color=(1, 0, 0),
                   rot_axis=2,
                   center_mode='lidar_bottom'):
    """open3d visualizer.

    Args:
        points (numpy.array | torch.tensor, shape=[N, 3+C]):
            points to visualize.
        bbox3d (numpy.array | torch.tensor, shape=[M, 7]):
            3d bbox (x, y, z, dx, dy, dz, yaw) to visualize.
        show (bool): whether to show the visualization results.
        save_path (str): path to save visualized results.
        points_size (int): the size of points to show on visualizer.
        point_color (tuple[float]): the color of points.
        bbox_color (tuple[float]): the color of bbox.
        points_in_box_color (tuple[float]):
            the color of points which are in bbox3d.
        rot_axis (int): rotation axis of bbox.
        center_mode (str): indicate the center of bbox is
            bottom center or gravity center.
    """
    # TODO: support rgb points
    # TODO: support score and class info
    # TODO: support image
    assert 0 <= rot_axis <= 2

    # init visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=[0, 0, 0])  # create coordinate frame
    vis.add_geometry(mesh_frame)

    # draw points
    pcd, points_colors = _draw_points(points, vis, points_size, point_color,
                                      mode)

    # draw boxes
    if bbox3d is not None:
        _draw_bboxes(bbox3d, vis, points_colors, pcd, bbox_color,
                     points_in_box_color, rot_axis, center_mode, mode)

    if show:
        vis.run()

    if save_path is not None:
        vis.capture_screen_image(save_path)

    vis.destroy_window()


def _draw_bboxes_ind(bbox3d,
                     vis,
                     indices=None,
                     points_colors=None,
                     pcd=None,
                     bbox_color=(0, 1, 0),
                     points_in_box_color=(1, 0, 0),
                     rot_axis=2,
                     bottom_center=True,
                     mode='xyz'):
    """Draw bbox and points in bbox on visualizer.

    Args:
        bbox3d (numpy.array | torch.tensor, shape=[M, 7]):
            3d bbox (x, y, z, dx, dy, dz, yaw) to visualize.
        vis (:obj:`open3d.visualization.Visualizer`): open3d visualizer.
        points_colors (numpy.array): color of each points.
        pcd (:obj:`open3d.geometry.PointCloud`): point cloud.
        bbox_color (tuple[float]): the color of bbox.
        points_in_box_color (tuple[float]):
            the color of points which are in bbox3d.
        rot_axis (int): rotation axis of bbox.
        bottom_center (bool): indicate the center of bbox is
            bottom center or gravity center.
    """
    if isinstance(bbox3d, torch.Tensor):
        bbox3d = bbox3d.cpu().numpy()
    if isinstance(indices, torch.Tensor):
        indices = indices.cpu().numpy()
    bbox3d = bbox3d.copy()

    in_box_color = np.array(points_in_box_color)
    for i in range(len(bbox3d)):
        center = bbox3d[i, 0:3]
        dim = bbox3d[i, 3:6]
        yaw = np.zeros(3)
        # TODO: fix problem of current coordinate system
        # dim[0], dim[1] = dim[1], dim[0]  # for current coordinate
        # yaw[rot_axis] = -(bbox3d[i, 6] - 0.5 * np.pi)
        yaw[rot_axis] = bbox3d[i, 6]
        rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)
        if bottom_center:
            center[2] += dim[2] / 2  # bottom center to gravity center
        box3d = geometry.OrientedBoundingBox(center, rot_mat, dim)

        line_set = geometry.LineSet.create_from_oriented_bounding_box(box3d)
        line_set.paint_uniform_color(bbox_color)
        # draw bboxes on visualizer
        vis.add_geometry(line_set)

        # change the color of points which are in box
        # if pcd is not None and mode == 'xyz':
        points_colors[indices[:, i].astype(np.bool)] = in_box_color

    # update points colors
    pcd.colors = o3d.utility.Vector3dVector(points_colors)
    vis.update_geometry(pcd)


def show_pts_index_boxes(points,
                         bbox3d=None,
                         show=True,
                         indices=None,
                         mode='xyz',
                         save_path=None,
                         points_size=2,
                         point_color=(0.5, 0.5, 0.5),
                         bbox_color=(0, 1, 0),
                         points_in_box_color=(1, 0, 0),
                         rot_axis=2,
                         bottom_center=True):
    """open3d visualizer.

    Args:
        points (numpy.array | torch.tensor, shape=[N, 3+C]):
            points to visualize.
        bbox3d (numpy.array | torch.tensor, shape=[M, 7]):
            3d bbox (x, y, z, dx, dy, dz, yaw) to visualize.
        show (bool): whether to show the visualization results.
        save_path (str): path to save visualized results.
        points_size (int): the size of points to show on visualizer.
        point_color (tuple[float]): the color of points.
        bbox_color (tuple[float]): the color of bbox.
        points_in_box_color (tuple[float]):
            the color of points which are in bbox3d.
        rot_axis (int): rotation axis of bbox.
        bottom_center (bool): indicate the center of bbox is
            bottom center or gravity center.
    """
    # TODO: support rgb points
    # TODO: support score and class info
    # TODO: support image
    assert 0 <= rot_axis <= 2

    # init visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=[0, 0, 0])  # create coordinate frame
    vis.add_geometry(mesh_frame)

    # draw points
    pcd, points_colors = _draw_points(points, vis, points_size, point_color,
                                      mode)

    # draw boxes
    if bbox3d is not None:
        _draw_bboxes_ind(bbox3d, vis, indices, points_colors, pcd, bbox_color,
                         points_in_box_color, rot_axis, bottom_center, mode)

    if show:
        vis.run()

    if save_path is not None:
        vis.capture_screen_image(save_path)

    vis.destroy_window()


def project_pts_on_img(points,
                       raw_img,
                       lidar2img_rt,
                       max_distance=70,
                       thickness=-1):
    """Project the 3D points cloud on 2D image.

    Args:
        points (numpy.array): 3D points cloud (x, y, z) to visualize.
        raw_img (numpy.array): The numpy array of image.
        lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        max_distance (float): the max distance of the points cloud.
        thickness (int, optional): The thickness of 2D points.
    """
    img = raw_img.copy()
    num_points = points.shape[0]
    pts_4d = np.concatenate([points[:, :3], np.ones((num_points, 1))], axis=-1)
    pts_2d = pts_4d @ lidar2img_rt.T

    # cam_points is Tensor of Nx4 whose last column is 1
    # transform camera coordinate to image coordinate
    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=99999)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]

    fov_inds = ((pts_2d[:, 0] < img.shape[1])
                & (pts_2d[:, 0] >= 0)
                & (pts_2d[:, 1] < img.shape[0])
                & (pts_2d[:, 1] >= 0))

    imgfov_pts_2d = pts_2d[fov_inds, :3]  # u, v, d

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pts_2d[i, 2]
        color = cmap[np.clip(int(max_distance * 10 / depth), 0, 255), :]
        cv2.circle(
            img,
            center=(int(np.round(imgfov_pts_2d[i, 0])),
                    int(np.round(imgfov_pts_2d[i, 1]))),
            radius=1,
            color=tuple(color),
            thickness=thickness,
        )
    cv2.imshow('project_pts_img', img)
    cv2.waitKey(100)


def project_bbox3d_on_img(bboxes3d,
                          raw_img,
                          lidar2img_rt,
                          color=(0, 255, 0),
                          thickness=1):
    """Project the 3D bbox on 2D image.

    Args:
        bboxes3d (numpy.array, shape=[M, 7]):
            3d bbox (x, y, z, dx, dy, dz, yaw) to visualize.
        raw_img (numpy.array): The numpy array of image.
        lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        color (tuple[int]): the color to draw bboxes.
        thickness (int, optional): The thickness of bboxes.
    """
    img = raw_img.copy()
    corners_3d = bboxes3d.corners
    num_bbox = corners_3d.shape[0]
    pts_4d = np.concatenate(
        [corners_3d.reshape(-1, 3),
         np.ones((num_bbox * 8, 1))], axis=-1)
    pts_2d = pts_4d @ lidar2img_rt.T

    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

    line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                    (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
    for i in range(num_bbox):
        corners = imgfov_pts_2d[i].astype(np.int)
        for start, end in line_indices:
            cv2.line(img, (corners[start, 0], corners[start, 1]),
                     (corners[end, 0], corners[end, 1]), color, thickness,
                     cv2.LINE_AA)

    cv2.imshow('project_bbox3d_img', img)
    cv2.waitKey(0)


class Visualizer(object):
    r"""Bbox head of `H3DNet <https://arxiv.org/abs/2006.05682>`_.

    Args:
        points (numpy.array, shape=[N, 3+C]): Points to visualize. The Points
            cloud is in mode of Coord3DMode.DEPTH (please refer to
            core.structures.coord_3d_mode).
        bbox3d (numpy.array, shape=[M, 7]): 3d bbox (x, y, z, dx, dy, dz, yaw)
            to visualize. The 3d bbox is in mode of Box3DMode.DEPTH with
            gravity_center (please refer to core.structures.box_3d_mode)
        save_path (str): path to save visualized results.
        points_size (int): the size of points to show on visualizer.
        point_color (tuple[float]): the color of points.
        bbox_color (tuple[float]): the color of bbox.
        points_in_box_color (tuple[float]):
            the color of points which are in bbox3d.
        rot_axis (int): rotation axis of bbox.
        center_mode (str): indicate the center of bbox is
            bottom center or gravity center.
    """

    def __init__(self,
                 points,
                 bbox3d=None,
                 mode='xyz',
                 save_path=None,
                 points_size=2,
                 point_color=(0.5, 0.5, 0.5),
                 bbox_color=(0, 1, 0),
                 points_in_box_color=(1, 0, 0),
                 rot_axis=2,
                 center_mode='lidar_bottom'):
        super(Visualizer, self).__init__()
        assert 0 <= rot_axis <= 2

        # init visualizer
        self.o3d_visualizer = o3d.visualization.Visualizer()
        self.o3d_visualizer.create_window()
        mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
            size=1, origin=[0, 0, 0])  # create coordinate frame
        self.o3d_visualizer.add_geometry(mesh_frame)

        self.points_size = points_size
        self.point_color = point_color
        self.bbox_color = bbox_color
        self.points_in_box_color = points_in_box_color
        self.rot_axis = rot_axis
        self.center_mode = center_mode
        self.mode = mode

        # draw points
        if points is not None:
            self.pcd, self.points_colors = _draw_points(
                points, self.o3d_visualizer, points_size, point_color, mode)

        # draw boxes
        if bbox3d is not None:
            _draw_bboxes(bbox3d, self.o3d_visualizer, self.points_colors,
                         self.pcd, bbox_color, points_in_box_color, rot_axis,
                         center_mode, mode)

    def add_bboxes(self, bbox3d, bbox_color=None, points_in_box_color=None):
        """Add bounding box to visualizer.

        Args:
            bbox3d (numpy.array, shape=[M, 7]):
                3d bbox(x, y, z, dx, dy, dz, yaw) to visualize. The 3d bbox
                is in mode of Box3DMode.DEPTH with
            gravity_center (please refer to core.structures.box_3d_mode)
            bbox_color (tuple[float]): the color of bbox.
            points_in_box_color (tuple[float]): the color of points which
                are in bbox3d.
        """
        if bbox_color is None:
            bbox_color = self.bbox_color
        if points_in_box_color is None:
            points_in_box_color = self.points_in_box_color
        _draw_bboxes(bbox3d, self.o3d_visualizer, self.points_colors, self.pcd,
                     bbox_color, points_in_box_color, self.rot_axis,
                     self.center_mode, self.mode)

    def show(self, save_path=None):
        """Visualize the points cloud."""
        self.o3d_visualizer.run()

        if save_path is not None:
            self.o3d_visualizer.capture_screen_image(save_path)

        self.o3d_visualizer.destroy_window()
        return


if __name__ == '__main__':
    from mmdet3d.datasets import KittiDataset

    data_root = 'data/kitti'
    ann_file = 'data/kitti/kitti_infos_train.pkl'
    classes = ['Pedestrian', 'Cyclist', 'Car']
    pts_prefix = 'velodyne_reduced'
    pipeline = [{
        'type': 'LoadPointsFromFile',
        'load_dim': 4,
        'use_dim': 4,
        'file_client_args': {
            'backend': 'disk'
        }
    }, {
        'type': 'LoadImageFromFile'
    }, {
        'type': 'LoadAnnotations3D',
        'with_bbox_3d': True,
        'with_label_3d': True,
        'file_client_args': {
            'backend': 'disk'
        }
    }]
    modality = {'use_lidar': True, 'use_camera': True}
    split = 'training'
    self = KittiDataset(
        data_root,
        ann_file,
        split,
        pts_prefix,
        pipeline,
        classes,
        modality,
        filter_empty_gt=False)

    infos = self.data_infos
    for i in range(500):
        data, info = self[i], infos[i]
        points, gt_bboxes = data['points'], data['gt_bboxes_3d']
        img = data['img']
        lidar2img_rt = data['lidar2img']
        project_pts_on_img(points, img, lidar2img_rt)
        project_bbox3d_on_img(gt_bboxes, img, lidar2img_rt)
