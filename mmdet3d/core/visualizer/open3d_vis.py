import numpy as np
import open3d as o3d
import torch
from open3d import geometry


def show(points,
         bbox3d=None,
         show=True,
         save_path=None,
         points_color=(0.5, 0.5, 0.5),
         bbox_color=(0, 1, 0),
         points_in_box_color=(1, 0, 0),
         rot_axis=2):
    """open3d visualizer.

    Args:
        points (numpy.array | torch.tensor, shape=[N, 3+C]):
            points to visualize.
        bbox3d (numpy.array | torch.tensor, shape=[M, 7]):
            3d bbox (x, y, z, dx, dy, dz, yaw) to visualize.
        show (bool): whether to show the visualization results.
        save_path (str): path to save visualized results.
        points_color (tuple[float]): the color of points.
        bbox_color (tuple[float]): the color of bbox.
        points_in_box_color (tuple[float]):
            the color of points which are in bbox3d.
        rot_axis (int): rotation axis of bbox.
    """
    assert 0 <= rot_axis <= 2

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 2  # set points size
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=5, origin=[0, 0, 0])  # create coordinate frame
    vis.add_geometry(mesh_frame)

    # show points
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    pcd = geometry.PointCloud()
    # TODO: support rgb points
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    colors = np.tile(np.array(points_color), (points.shape[0], 1))

    # show boxes
    if bbox3d is not None:
        if isinstance(bbox3d, torch.Tensor):
            bbox3d = bbox3d.cpu().numpy()

        in_box_color = np.array(points_in_box_color)
        for i in range(len(bbox3d)):
            yaw = np.zeros(3)
            yaw[rot_axis] = 180 * bbox3d[i, -1] / np.pi  # radian to angle
            rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)
            center = bbox3d[i, 0:3]
            dim = bbox3d[i, 3:6]
            box3d = geometry.OrientedBoundingBox(center, rot_mat, dim)

            line_set = geometry.LineSet.create_from_oriented_bounding_box(
                box3d)
            line_set.paint_uniform_color(bbox_color)
            vis.add_geometry(line_set)

            # change the color of points which are in box
            indices = box3d.get_point_indices_within_bounding_box(pcd.points)
            colors[indices] = in_box_color

    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(pcd)

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

    show(points, boxes3d, save_path='./file.png')

    print()
