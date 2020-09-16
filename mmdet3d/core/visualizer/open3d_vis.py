import numpy as np
import open3d as o3d
import torch
from open3d import geometry


def _draw_points(points, vis, points_size=2, point_color=(0.5, 0.5, 0.5)):
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
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    points_colors = np.tile(np.array(point_color), (points.shape[0], 1))

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
                 bottom_center=True):
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
        dim[0], dim[1] = dim[1], dim[0]  # for current coordinate
        yaw[rot_axis] = -(bbox3d[i, 6] - 0.5 * np.pi)
        # yaw[rot_axis] = bbox3d[i, 6]
        rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)
        if bottom_center:
            center[2] += dim[2] / 2  # bottom center to gravity center
        box3d = geometry.OrientedBoundingBox(center, rot_mat, dim)

        line_set = geometry.LineSet.create_from_oriented_bounding_box(box3d)
        line_set.paint_uniform_color(bbox_color)
        # draw bboxes on visualizer
        vis.add_geometry(line_set)

        # change the color of points which are in box
        if pcd is not None:
            indices = box3d.get_point_indices_within_bounding_box(pcd.points)
            points_colors[indices] = in_box_color

    # update points colors
    pcd.colors = o3d.utility.Vector3dVector(points_colors)
    vis.update_geometry(pcd)


def show_pts_boxes(points,
                   bbox3d=None,
                   show=True,
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

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
        size=5, origin=[0, 0, 0])  # create coordinate frame
    vis.add_geometry(mesh_frame)

    # draw points
    pcd, points_colors = _draw_points(points, vis, points_size, point_color)

    # draw boxes
    if bbox3d is not None:
        _draw_bboxes(bbox3d, vis, points_colors, pcd, bbox_color,
                     points_in_box_color, rot_axis, bottom_center)

    if show:
        vis.run()

    if save_path is not None:
        vis.capture_screen_image(save_path)

    vis.destroy_window()


if __name__ == '__main__':
    pts_filename = '/home/SENSETIME/wuyuefeng/projects/' \
                   + 'pth_models/mmdetection3d/demo/kitti_000008.bin'
    points = np.fromfile(pts_filename, dtype=np.float32).reshape(-1, 4)[:, :3]
    points = torch.from_numpy(points)
    boxes3d = np.array([[0, 0, 0, 1, 1, 1, 0], [10, 0, 0, 5, 5, 5, 0],
                        [10, 10, 0, 1, 1, 1, 0], [10, 10, 0, 1, 1, 1, 0.8]])

    show_pts_boxes(points, boxes3d, save_path='./file.png')

    print()
