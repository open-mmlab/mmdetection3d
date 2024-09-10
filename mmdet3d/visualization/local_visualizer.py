# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
import os
import sys
import time
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import mmcv
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from mmdet.visualization import DetLocalVisualizer, get_palette
from mmengine.dist import master_only
from mmengine.logging import print_log
from mmengine.structures import InstanceData
from mmengine.visualization import Visualizer as MMENGINE_Visualizer
from mmengine.visualization.utils import (check_type, color_val_matplotlib,
                                          tensor2ndarray)
from torch import Tensor

from mmdet3d.registry import VISUALIZERS
from mmdet3d.structures import (BaseInstance3DBoxes, Box3DMode,
                                CameraInstance3DBoxes, Coord3DMode,
                                DepthInstance3DBoxes, DepthPoints,
                                Det3DDataSample, LiDARInstance3DBoxes,
                                PointData, points_cam2img)
from .vis_utils import (proj_camera_bbox3d_to_img, proj_depth_bbox3d_to_img,
                        proj_lidar_bbox3d_to_img, to_depth_mode)

try:
    import open3d as o3d
    from open3d import geometry
    from open3d.visualization import Visualizer
except ImportError:
    o3d = geometry = Visualizer = None


@VISUALIZERS.register_module()
class Det3DLocalVisualizer(DetLocalVisualizer):
    """MMDetection3D Local Visualizer.

    - 3D detection and segmentation drawing methods

      - draw_bboxes_3d: draw 3D bounding boxes on point clouds
      - draw_proj_bboxes_3d: draw projected 3D bounding boxes on image
      - draw_seg_mask: draw segmentation mask via per-point colorization

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        points (np.ndarray, optional): Points to visualize with shape (N, 3+C).
            Defaults to None.
        image (np.ndarray, optional): The origin image to draw. The format
            should be RGB. Defaults to None.
        pcd_mode (int): The point cloud mode (coordinates): 0 represents LiDAR,
            1 represents CAMERA, 2 represents Depth. Defaults to 0.
        vis_backends (List[dict], optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
            Defaults to None.
        bbox_color (str or Tuple[int], optional): Color of bbox lines.
            The tuple of color should be in BGR order. Defaults to None.
        text_color (str or Tuple[int]): Color of texts. The tuple of color
            should be in BGR order. Defaults to (200, 200, 200).
        mask_color (str or Tuple[int], optional): Color of masks. The tuple of
            color should be in BGR order. Defaults to None.
        line_width (int or float): The linewidth of lines. Defaults to 3.
        frame_cfg (dict): The coordinate frame config while Open3D
            visualization initialization.
            Defaults to dict(size=1, origin=[0, 0, 0]).
        alpha (int or float): The transparency of bboxes or mask.
            Defaults to 0.8.
        multi_imgs_col (int): The number of columns in arrangement when showing
            multi-view images.

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from mmengine.structures import InstanceData
        >>> from mmdet3d.structures import (DepthInstance3DBoxes
        ...                                 Det3DDataSample)
        >>> from mmdet3d.visualization import Det3DLocalVisualizer

        >>> det3d_local_visualizer = Det3DLocalVisualizer()
        >>> image = np.random.randint(0, 256, size=(10, 12, 3)).astype('uint8')
        >>> points = np.random.rand(1000, 3)
        >>> gt_instances_3d = InstanceData()
        >>> gt_instances_3d.bboxes_3d = DepthInstance3DBoxes(
        ...     torch.rand((5, 7)))
        >>> gt_instances_3d.labels_3d = torch.randint(0, 2, (5,))
        >>> gt_det3d_data_sample = Det3DDataSample()
        >>> gt_det3d_data_sample.gt_instances_3d = gt_instances_3d
        >>> data_input = dict(img=image, points=points)
        >>> det3d_local_visualizer.add_datasample('3D Scene', data_input,
        ...                                       gt_det3d_data_sample)

        >>> from mmdet3d.structures import PointData
        >>> det3d_local_visualizer = Det3DLocalVisualizer()
        >>> points = np.random.rand(1000, 3)
        >>> gt_pts_seg = PointData()
        >>> gt_pts_seg.pts_semantic_mask = torch.randint(0, 10, (1000, ))
        >>> gt_det3d_data_sample = Det3DDataSample()
        >>> gt_det3d_data_sample.gt_pts_seg = gt_pts_seg
        >>> data_input = dict(points=points)
        >>> det3d_local_visualizer.add_datasample('3D Scene', data_input,
        ...                                       gt_det3d_data_sample,
        ...                                       vis_task='lidar_seg')
    """

    def __init__(
        self,
        name: str = 'visualizer',
        points: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
        pcd_mode: int = 0,
        vis_backends: Optional[List[dict]] = None,
        save_dir: Optional[str] = None,
        bbox_color: Optional[Union[str, Tuple[int]]] = None,
        text_color: Union[str, Tuple[int]] = (200, 200, 200),
        mask_color: Optional[Union[str, Tuple[int]]] = None,
        line_width: Union[int, float] = 3,
        frame_cfg: dict = dict(size=1, origin=[0, 0, 0]),
        alpha: Union[int, float] = 0.8,
        multi_imgs_col: int = 3,
        fig_show_cfg: dict = dict(figsize=(18, 12))
    ) -> None:
        super().__init__(
            name=name,
            image=image,
            vis_backends=vis_backends,
            save_dir=save_dir,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            line_width=line_width,
            alpha=alpha)
        if points is not None:
            self.set_points(points, pcd_mode=pcd_mode, frame_cfg=frame_cfg)
        self.multi_imgs_col = multi_imgs_col
        self.fig_show_cfg.update(fig_show_cfg)

        self.flag_pause = False
        self.flag_next = False
        self.flag_exit = False

    def _clear_o3d_vis(self) -> None:
        """Clear open3d vis."""

        if hasattr(self, 'o3d_vis'):
            del self.o3d_vis
            del self.points_colors
            del self.view_control
            if hasattr(self, 'pcd'):
                del self.pcd

    def _initialize_o3d_vis(self, show=True) -> Visualizer:
        """Initialize open3d vis according to frame_cfg.

        Args:
            frame_cfg (dict): The config to create coordinate frame in open3d
                vis.

        Returns:
            :obj:`o3d.visualization.Visualizer`: Created open3d vis.
        """
        if o3d is None or geometry is None:
            raise ImportError(
                'Please run "pip install open3d" to install open3d first.')
        glfw_key_escape = 256  # Esc
        glfw_key_space = 32  # Space
        glfw_key_right = 262  # Right
        o3d_vis = o3d.visualization.VisualizerWithKeyCallback()
        o3d_vis.register_key_callback(glfw_key_escape, self.escape_callback)
        o3d_vis.register_key_action_callback(glfw_key_space,
                                             self.space_action_callback)
        o3d_vis.register_key_callback(glfw_key_right, self.right_callback)
        if os.environ.get('DISPLAY', None) is not None and show:
            o3d_vis.create_window()
            self.view_control = o3d_vis.get_view_control()
        return o3d_vis

    @master_only
    def set_points(self,
                   points: np.ndarray,
                   pcd_mode: int = 0,
                   vis_mode: str = 'replace',
                   frame_cfg: dict = dict(size=1, origin=[0, 0, 0]),
                   points_color: Tuple[float] = (0.8, 0.8, 0.8),
                   points_size: int = 2,
                   mode: str = 'xyz') -> None:
        """Set the point cloud to draw.

        Args:
            points (np.ndarray): Points to visualize with shape (N, 3+C).
            pcd_mode (int): The point cloud mode (coordinates): 0 represents
                LiDAR, 1 represents CAMERA, 2 represents Depth. Defaults to 0.
            vis_mode (str): The visualization mode in Open3D:

                - 'replace': Replace the existing point cloud with input point
                  cloud.
                - 'add': Add input point cloud into existing point cloud.

                Defaults to 'replace'.
            frame_cfg (dict): The coordinate frame config for Open3D
                visualization initialization.
                Defaults to dict(size=1, origin=[0, 0, 0]).
            points_color (Tuple[float]): The color of points.
                Defaults to (1, 1, 1).
            points_size (int): The size of points to show on visualizer.
                Defaults to 2.
            mode (str): Indicate type of the input points, available mode
                ['xyz', 'xyzrgb']. Defaults to 'xyz'.
        """
        assert points is not None
        assert vis_mode in ('replace', 'add')
        check_type('points', points, np.ndarray)

        if not hasattr(self, 'o3d_vis'):
            self.o3d_vis = self._initialize_o3d_vis()

        # for now we convert points into depth mode for visualization
        if pcd_mode != Coord3DMode.DEPTH:
            points = Coord3DMode.convert(points, pcd_mode, Coord3DMode.DEPTH)

        if hasattr(self, 'pcd') and vis_mode != 'add':
            self.o3d_vis.remove_geometry(self.pcd)

        # set points size in Open3D
        render_option = self.o3d_vis.get_render_option()
        if render_option is not None:
            render_option.point_size = points_size
            render_option.background_color = np.asarray([0, 0, 0])

        points = points.copy()
        pcd = geometry.PointCloud()
        if mode == 'xyz':
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            points_colors = np.tile(
                np.array(points_color), (points.shape[0], 1))
        elif mode == 'xyzrgb':
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            points_colors = points[:, 3:6]
            # normalize to [0, 1] for Open3D drawing
            if not ((points_colors >= 0.0) & (points_colors <= 1.0)).all():
                points_colors /= 255.0
        else:
            raise NotImplementedError

        # create coordinate frame
        mesh_frame = geometry.TriangleMesh.create_coordinate_frame(**frame_cfg)
        self.o3d_vis.add_geometry(mesh_frame)

        pcd.colors = o3d.utility.Vector3dVector(points_colors)
        self.o3d_vis.add_geometry(pcd)
        self.pcd = pcd
        self.points_colors = points_colors

    # TODO: assign 3D Box color according to pred / GT labels
    # We draw GT / pred bboxes on the same point cloud scenes
    # for better detection performance comparison
    def draw_bboxes_3d(self,
                       bboxes_3d: BaseInstance3DBoxes,
                       bbox_color: Tuple[float] = (0, 1, 0),
                       points_in_box_color: Tuple[float] = (1, 0, 0),
                       rot_axis: int = 2,
                       center_mode: str = 'lidar_bottom',
                       mode: str = 'xyz') -> None:
        """Draw bbox on visualizer and change the color of points inside
        bbox3d.

        Args:
            bboxes_3d (:obj:`BaseInstance3DBoxes`): 3D bbox
                (x, y, z, x_size, y_size, z_size, yaw) to visualize.
            bbox_color (Tuple[float]): The color of 3D bboxes.
                Defaults to (0, 1, 0).
            points_in_box_color (Tuple[float]): The color of points inside 3D
                bboxes. Defaults to (1, 0, 0).
            rot_axis (int): Rotation axis of 3D bboxes. Defaults to 2.
            center_mode (str): Indicates the center of bbox is bottom center or
                gravity center. Available mode
                ['lidar_bottom', 'camera_bottom']. Defaults to 'lidar_bottom'.
            mode (str): Indicates the type of input points, available mode
                ['xyz', 'xyzrgb']. Defaults to 'xyz'.
        """
        # Before visualizing the 3D Boxes in point cloud scene
        # we need to convert the boxes to Depth mode
        check_type('bboxes', bboxes_3d, BaseInstance3DBoxes)

        if not isinstance(bboxes_3d, DepthInstance3DBoxes):
            bboxes_3d = bboxes_3d.convert_to(Box3DMode.DEPTH)

        # convert bboxes to numpy dtype
        bboxes_3d = tensor2ndarray(bboxes_3d.tensor)

        # in_box_color = np.array(points_in_box_color)

        for i in range(len(bboxes_3d)):
            center = bboxes_3d[i, 0:3]
            dim = bboxes_3d[i, 3:6]
            yaw = np.zeros(3)
            yaw[rot_axis] = bboxes_3d[i, 6]
            rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)

            if center_mode == 'lidar_bottom':
                # bottom center to gravity center
                center[rot_axis] += dim[rot_axis] / 2
            elif center_mode == 'camera_bottom':
                # bottom center to gravity center
                center[rot_axis] -= dim[rot_axis] / 2
            box3d = geometry.OrientedBoundingBox(center, rot_mat, dim)

            line_set = geometry.LineSet.create_from_oriented_bounding_box(
                box3d)
            line_set.paint_uniform_color(np.array(bbox_color[i]) / 255.)
            # draw bboxes on visualizer
            self.o3d_vis.add_geometry(line_set)

            # change the color of points which are in box
            if self.pcd is not None and mode == 'xyz':
                indices = box3d.get_point_indices_within_bounding_box(
                    self.pcd.points)
                self.points_colors[indices] = np.array(bbox_color[i]) / 255.

        # update points colors
        if self.pcd is not None:
            self.pcd.colors = o3d.utility.Vector3dVector(self.points_colors)
            self.o3d_vis.update_geometry(self.pcd)

    def set_bev_image(self,
                      bev_image: Optional[np.ndarray] = None,
                      bev_shape: int = 900) -> None:
        """Set the bev image to draw.

        Args:
            bev_image (np.ndarray, optional): The bev image to draw.
                Defaults to None.
            bev_shape (int): The bev image shape. Defaults to 900.
        """
        if bev_image is None:
            bev_image = np.zeros((bev_shape, bev_shape, 3), np.uint8)

        self._image = bev_image
        self.width, self.height = bev_image.shape[1], bev_image.shape[0]
        self._default_font_size = max(
            np.sqrt(self.height * self.width) // 90, 10)
        self.ax_save.cla()
        self.ax_save.axis(False)
        self.ax_save.imshow(bev_image, origin='lower')
        # plot camera view range
        x1 = np.linspace(0, self.width / 2)
        x2 = np.linspace(self.width / 2, self.width)
        self.ax_save.plot(
            x1,
            self.width / 2 - x1,
            ls='--',
            color='grey',
            linewidth=1,
            alpha=0.5)
        self.ax_save.plot(
            x2,
            x2 - self.width / 2,
            ls='--',
            color='grey',
            linewidth=1,
            alpha=0.5)
        self.ax_save.plot(
            self.width / 2,
            0,
            marker='+',
            markersize=16,
            markeredgecolor='red')

    # TODO: Support bev point cloud visualization
    @master_only
    def draw_bev_bboxes(self,
                        bboxes_3d: BaseInstance3DBoxes,
                        scale: int = 15,
                        edge_colors: Union[str, Tuple[int],
                                           List[Union[str, Tuple[int]]]] = 'o',
                        line_styles: Union[str, List[str]] = '-',
                        line_widths: Union[int, float, List[Union[int,
                                                                  float]]] = 1,
                        face_colors: Union[str, Tuple[int],
                                           List[Union[str,
                                                      Tuple[int]]]] = 'none',
                        alpha: Union[int, float] = 1) -> MMENGINE_Visualizer:
        """Draw projected 3D boxes on the image.

        Args:
            bboxes_3d (:obj:`BaseInstance3DBoxes`): 3D bbox
                (x, y, z, x_size, y_size, z_size, yaw) to visualize.
            scale (dict): Value to scale the bev bboxes for better
                visualization. Defaults to 15.
            edge_colors (str or Tuple[int] or List[str or Tuple[int]]):
                The colors of bboxes. ``colors`` can have the same length with
                lines or just single value. If ``colors`` is single value, all
                the lines will have the same colors. Refer to `matplotlib.
                colors` for full list of formats that are accepted.
                Defaults to 'o'.
            line_styles (str or List[str]): The linestyle of lines.
                ``line_styles`` can have the same length with texts or just
                single value. If ``line_styles`` is single value, all the lines
                will have the same linestyle. Reference to
                https://matplotlib.org/stable/api/collections_api.html?highlight=collection#matplotlib.collections.AsteriskPolygonCollection.set_linestyle
                for more details. Defaults to '-'.
            line_widths (int or float or List[int or float]): The linewidth of
                lines. ``line_widths`` can have the same length with lines or
                just single value. If ``line_widths`` is single value, all the
                lines will have the same linewidth. Defaults to 2.
            face_colors (str or Tuple[int] or List[str or Tuple[int]]):
                The face colors. Defaults to 'none'.
            alpha (int or float): The transparency of bboxes. Defaults to 1.
        """

        check_type('bboxes', bboxes_3d, BaseInstance3DBoxes)
        bev_bboxes = tensor2ndarray(bboxes_3d.bev)
        # scale the bev bboxes for better visualization
        bev_bboxes[:, :4] *= scale
        ctr, w, h, theta = np.split(bev_bboxes, [2, 3, 4], axis=-1)
        cos_value, sin_value = np.cos(theta), np.sin(theta)
        vec1 = np.concatenate([w / 2 * cos_value, w / 2 * sin_value], axis=-1)
        vec2 = np.concatenate([-h / 2 * sin_value, h / 2 * cos_value], axis=-1)
        pt1 = ctr + vec1 + vec2
        pt2 = ctr + vec1 - vec2
        pt3 = ctr - vec1 - vec2
        pt4 = ctr - vec1 + vec2
        poly = np.stack([pt1, pt2, pt3, pt4], axis=-2)
        # move the object along x-axis
        poly[:, :, 0] += self.width / 2
        poly = [p for p in poly]
        return self.draw_polygons(
            poly,
            alpha=alpha,
            edge_colors=edge_colors,
            line_styles=line_styles,
            line_widths=line_widths,
            face_colors=face_colors)

    @master_only
    def draw_points_on_image(self,
                             points: Union[np.ndarray, Tensor],
                             pts2img: np.ndarray,
                             sizes: Union[np.ndarray, int] = 3,
                             max_depth: Optional[float] = None) -> None:
        """Draw projected points on the image.

        Args:
            points (np.ndarray or Tensor): Points to draw.
            pts2img (np.ndarray): The transformation matrix from the coordinate
                of point cloud to image plane.
            sizes (np.ndarray or int): The marker size. Defaults to 10.
            max_depth (float): The max depth in the color map. Defaults to
                None.
        """
        check_type('points', points, (np.ndarray, Tensor))
        points = tensor2ndarray(points)
        assert self._image is not None, 'Please set image using `set_image`'
        projected_points = points_cam2img(points, pts2img, with_depth=True)
        depths = projected_points[:, 2]
        # Show depth adaptively consideing different scenes
        if max_depth is None:
            max_depth = depths.max()
        colors = (depths % max_depth) / max_depth
        # use colormap to obtain the render color
        color_map = plt.get_cmap('jet')
        self.ax_save.scatter(
            projected_points[:, 0],
            projected_points[:, 1],
            c=colors,
            cmap=color_map,
            s=sizes,
            alpha=0.7,
            edgecolors='none')

    # TODO: set bbox color according to palette
    @master_only
    def draw_proj_bboxes_3d(
            self,
            bboxes_3d: BaseInstance3DBoxes,
            input_meta: dict,
            edge_colors: Union[str, Tuple[int],
                               List[Union[str, Tuple[int]]]] = 'royalblue',
            line_styles: Union[str, List[str]] = '-',
            line_widths: Union[int, float, List[Union[int, float]]] = 2,
            face_colors: Union[str, Tuple[int],
                               List[Union[str, Tuple[int]]]] = 'royalblue',
            alpha: Union[int, float] = 0.4,
            img_size: Optional[Tuple] = None):
        """Draw projected 3D boxes on the image.

        Args:
            bboxes_3d (:obj:`BaseInstance3DBoxes`): 3D bbox
                (x, y, z, x_size, y_size, z_size, yaw) to visualize.
            input_meta (dict): Input meta information.
            edge_colors (str or Tuple[int] or List[str or Tuple[int]]):
                The colors of bboxes. ``colors`` can have the same length with
                lines or just single value. If ``colors`` is single value, all
                the lines will have the same colors. Refer to `matplotlib.
                colors` for full list of formats that are accepted.
                Defaults to 'royalblue'.
            line_styles (str or List[str]): The linestyle of lines.
                ``line_styles`` can have the same length with texts or just
                single value. If ``line_styles`` is single value, all the lines
                will have the same linestyle. Reference to
                https://matplotlib.org/stable/api/collections_api.html?highlight=collection#matplotlib.collections.AsteriskPolygonCollection.set_linestyle
                for more details. Defaults to '-'.
            line_widths (int or float or List[int or float]): The linewidth of
                lines. ``line_widths`` can have the same length with lines or
                just single value. If ``line_widths`` is single value, all the
                lines will have the same linewidth. Defaults to 2.
            face_colors (str or Tuple[int] or List[str or Tuple[int]]):
                The face colors. Defaults to 'royalblue'.
            alpha (int or float): The transparency of bboxes. Defaults to 0.4.
            img_size (tuple, optional): The size (w, h) of the image.
        """

        check_type('bboxes', bboxes_3d, BaseInstance3DBoxes)

        if isinstance(bboxes_3d, DepthInstance3DBoxes):
            proj_bbox3d_to_img = proj_depth_bbox3d_to_img
        elif isinstance(bboxes_3d, LiDARInstance3DBoxes):
            proj_bbox3d_to_img = proj_lidar_bbox3d_to_img
        elif isinstance(bboxes_3d, CameraInstance3DBoxes):
            proj_bbox3d_to_img = proj_camera_bbox3d_to_img
        else:
            raise NotImplementedError('unsupported box type!')

        edge_colors_norm = color_val_matplotlib(edge_colors)

        corners_2d = proj_bbox3d_to_img(bboxes_3d, input_meta)
        if img_size is not None:
            # Filter out the bbox where half of stuff is outside the image.
            # This is for the visualization of multi-view image.
            valid_point_idx = (corners_2d[..., 0] >= 0) & \
                        (corners_2d[..., 0] <= img_size[0]) & \
                        (corners_2d[..., 1] >= 0) & (corners_2d[..., 1] <= img_size[1])  # noqa: E501
            valid_bbox_idx = valid_point_idx.sum(axis=-1) >= 4
            corners_2d = corners_2d[valid_bbox_idx]
            filter_edge_colors = []
            filter_edge_colors_norm = []
            for i, color in enumerate(edge_colors):
                if valid_bbox_idx[i]:
                    filter_edge_colors.append(color)
                    filter_edge_colors_norm.append(edge_colors_norm[i])
            edge_colors = filter_edge_colors
            edge_colors_norm = filter_edge_colors_norm

        lines_verts_idx = [0, 1, 2, 3, 7, 6, 5, 4, 0, 3, 7, 4, 5, 1, 2, 6]
        lines_verts = corners_2d[:, lines_verts_idx, :]
        front_polys = corners_2d[:, 4:, :]
        codes = [Path.LINETO] * lines_verts.shape[1]
        codes[0] = Path.MOVETO
        pathpatches = []
        for i in range(len(corners_2d)):
            verts = lines_verts[i]
            pth = Path(verts, codes)
            pathpatches.append(PathPatch(pth))

        p = PatchCollection(
            pathpatches,
            facecolors='none',
            edgecolors=edge_colors_norm,
            linewidths=line_widths,
            linestyles=line_styles)

        self.ax_save.add_collection(p)

        # draw a mask on the front of project bboxes
        front_polys = [front_poly for front_poly in front_polys]
        return self.draw_polygons(
            front_polys,
            alpha=alpha,
            edge_colors=edge_colors,
            line_styles=line_styles,
            line_widths=line_widths,
            face_colors=edge_colors)

    @master_only
    def draw_seg_mask(self, seg_mask_colors: np.ndarray) -> None:
        """Add segmentation mask to visualizer via per-point colorization.

        Args:
            seg_mask_colors (np.ndarray): The segmentation mask with shape
                (N, 6), whose first 3 dims are point coordinates and last 3
                dims are converted colors.
        """
        # we can't draw the colors on existing points
        # in case gt and pred mask would overlap
        # instead we set a large offset along x-axis for each seg mask
        if hasattr(self, 'pcd'):
            offset = (np.array(self.pcd.points).max(0) -
                      np.array(self.pcd.points).min(0))[0] * 1.2
            mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
                size=1, origin=[offset, 0,
                                0])  # create coordinate frame for seg
            self.o3d_vis.add_geometry(mesh_frame)
        else:
            offset = 0
        seg_points = copy.deepcopy(seg_mask_colors)
        seg_points[:, 0] += offset
        self.set_points(seg_points, pcd_mode=2, vis_mode='add', mode='xyzrgb')

    def _draw_instances_3d(self,
                           data_input: dict,
                           instances: InstanceData,
                           input_meta: dict,
                           vis_task: str,
                           show_pcd_rgb: bool = False,
                           palette: Optional[List[tuple]] = None) -> dict:
        """Draw 3D instances of GT or prediction.

        Args:
            data_input (dict): The input dict to draw.
            instances (:obj:`InstanceData`): Data structure for instance-level
                annotations or predictions.
            input_meta (dict): Meta information.
            vis_task (str): Visualization task, it includes: 'lidar_det',
                'multi-modality_det', 'mono_det'.
            show_pcd_rgb (bool): Whether to show RGB point cloud.
            palette (List[tuple], optional): Palette information corresponding
                to the category. Defaults to None.

        Returns:
            dict: The drawn point cloud and image whose channel is RGB.
        """

        # Only visualize when there is at least one instance
        if not len(instances) > 0:
            return None

        bboxes_3d = instances.bboxes_3d  # BaseInstance3DBoxes
        labels_3d = instances.labels_3d

        data_3d = dict()

        if vis_task in ['lidar_det', 'multi-modality_det']:
            assert 'points' in data_input
            points = data_input['points']
            check_type('points', points, (np.ndarray, Tensor))
            points = tensor2ndarray(points)

            if not isinstance(bboxes_3d, DepthInstance3DBoxes):
                points, bboxes_3d_depth = to_depth_mode(points, bboxes_3d)
            else:
                bboxes_3d_depth = bboxes_3d.clone()

            if 'axis_align_matrix' in input_meta:
                points = DepthPoints(points, points_dim=points.shape[1])
                rot_mat = input_meta['axis_align_matrix'][:3, :3]
                trans_vec = input_meta['axis_align_matrix'][:3, -1]
                points.rotate(rot_mat.T)
                points.translate(trans_vec)
                points = tensor2ndarray(points.tensor)

            max_label = int(max(labels_3d) if len(labels_3d) > 0 else 0)
            bbox_color = palette if self.bbox_color is None \
                else self.bbox_color
            bbox_palette = get_palette(bbox_color, max_label + 1)
            colors = [bbox_palette[label] for label in labels_3d]

            self.set_points(
                points, pcd_mode=2, mode='xyzrgb' if show_pcd_rgb else 'xyz')
            self.draw_bboxes_3d(bboxes_3d_depth, bbox_color=colors)

            data_3d['bboxes_3d'] = tensor2ndarray(bboxes_3d_depth.tensor)
            data_3d['points'] = points

        if vis_task in ['mono_det', 'multi-modality_det']:
            assert 'img' in data_input
            img = data_input['img']
            if isinstance(img, list) or (isinstance(img, (np.ndarray, Tensor))
                                         and len(img.shape) == 4):
                # show multi-view images
                img_size = img[0].shape[:2] if isinstance(
                    img, list) else img.shape[-2:]  # noqa: E501
                img_col = self.multi_imgs_col
                img_row = math.ceil(len(img) / img_col)
                composed_img = np.zeros(
                    (img_size[0] * img_row, img_size[1] * img_col, 3),
                    dtype=np.uint8)
                for i, single_img in enumerate(img):
                    # Note that we should keep the same order of elements both
                    # in `img` and `input_meta`
                    if isinstance(single_img, Tensor):
                        single_img = single_img.permute(1, 2, 0).numpy()
                        single_img = single_img[..., [2, 1, 0]]  # bgr to rgb
                    self.set_image(single_img)
                    single_img_meta = dict()
                    for key, meta in input_meta.items():
                        if isinstance(meta,
                                      (Sequence, np.ndarray,
                                       Tensor)) and len(meta) == len(img):
                            single_img_meta[key] = meta[i]
                        else:
                            single_img_meta[key] = meta

                    max_label = int(
                        max(labels_3d) if len(labels_3d) > 0 else 0)
                    bbox_color = palette if self.bbox_color is None \
                        else self.bbox_color
                    bbox_palette = get_palette(bbox_color, max_label + 1)
                    colors = [bbox_palette[label] for label in labels_3d]

                    self.draw_proj_bboxes_3d(
                        bboxes_3d,
                        single_img_meta,
                        img_size=single_img.shape[:2][::-1],
                        edge_colors=colors)
                    if vis_task == 'mono_det' and hasattr(
                            instances, 'centers_2d'):
                        centers_2d = instances.centers_2d
                        self.draw_points(centers_2d)
                    composed_img[(i // img_col) *
                                 img_size[0]:(i // img_col + 1) * img_size[0],
                                 (i % img_col) *
                                 img_size[1]:(i % img_col + 1) *
                                 img_size[1]] = self.get_image()
                data_3d['img'] = composed_img
            else:
                # show single-view image
                # TODO: Solve the problem: some line segments of 3d bboxes are
                # out of image by a large margin
                if isinstance(data_input['img'], Tensor):
                    img = img.permute(1, 2, 0).numpy()
                    img = img[..., [2, 1, 0]]  # bgr to rgb
                self.set_image(img)

                max_label = int(max(labels_3d) if len(labels_3d) > 0 else 0)
                bbox_color = palette if self.bbox_color is None \
                    else self.bbox_color
                bbox_palette = get_palette(bbox_color, max_label + 1)
                colors = [bbox_palette[label] for label in labels_3d]

                self.draw_proj_bboxes_3d(
                    bboxes_3d, input_meta, edge_colors=colors)
                if vis_task == 'mono_det' and hasattr(instances, 'centers_2d'):
                    centers_2d = instances.centers_2d
                    self.draw_points(centers_2d)
                drawn_img = self.get_image()
                data_3d['img'] = drawn_img

        return data_3d

    def _draw_pts_sem_seg(self,
                          points: Union[Tensor, np.ndarray],
                          pts_seg: PointData,
                          palette: Optional[List[tuple]] = None,
                          keep_index: Optional[int] = None) -> None:
        """Draw 3D semantic mask of GT or prediction.

        Args:
            points (Tensor or np.ndarray): The input point cloud to draw.
            pts_seg (:obj:`PointData`): Data structure for pixel-level
                annotations or predictions.
            palette (List[tuple], optional): Palette information corresponding
                to the category. Defaults to None.
            ignore_index (int, optional): Ignore category. Defaults to None.
        """
        check_type('points', points, (np.ndarray, Tensor))

        points = tensor2ndarray(points)
        pts_sem_seg = tensor2ndarray(pts_seg.pts_semantic_mask)
        palette = np.array(palette)

        if keep_index is not None:
            keep_index = tensor2ndarray(keep_index)
            points = points[keep_index]
            pts_sem_seg = pts_sem_seg[keep_index]

        pts_color = palette[pts_sem_seg]
        seg_color = np.concatenate([points[:, :3], pts_color], axis=1)

        self.draw_seg_mask(seg_color)

    @master_only
    def show(self,
             save_path: Optional[str] = None,
             drawn_img_3d: Optional[np.ndarray] = None,
             drawn_img: Optional[np.ndarray] = None,
             win_name: str = 'image',
             wait_time: int = -1,
             continue_key: str = 'right',
             vis_task: str = 'lidar_det') -> None:
        """Show the drawn point cloud/image.

        Args:
            save_path (str, optional): Path to save open3d visualized results.
                Defaults to None.
            drawn_img_3d (np.ndarray, optional): The image to show. If
                drawn_img_3d is not None, it will show the image got by
                Visualizer. Defaults to None.
            drawn_img (np.ndarray, optional): The image to show. If drawn_img
                is not None, it will show the image got by Visualizer.
                Defaults to None.
            win_name (str): The image title. Defaults to 'image'.
            wait_time (int): Delay in milliseconds. 0 is the special value that
                means "forever". Defaults to 0.
            continue_key (str): The key for users to continue. Defaults to ' '.
        """

        # In order to show multi-modal results at the same time, we show image
        # firstly and then show point cloud since the running of
        # Open3D will block the process
        if hasattr(self, '_image'):
            if drawn_img is None and drawn_img_3d is None:
                # use the image got by Visualizer.get_image()
                if vis_task == 'multi-modality_det':
                    import matplotlib.pyplot as plt
                    is_inline = 'inline' in plt.get_backend()
                    img = self.get_image() if drawn_img is None else drawn_img
                    self._init_manager(win_name)
                    fig = self.manager.canvas.figure
                    # remove white edges by set subplot margin
                    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
                    fig.clear()
                    ax = fig.add_subplot()
                    ax.axis(False)
                    ax.imshow(img)
                    self.manager.canvas.draw()
                    if is_inline:
                        return fig
                    else:
                        fig.show()
                    self.manager.canvas.flush_events()
                else:
                    super().show(drawn_img_3d, win_name, wait_time,
                                 continue_key)
            else:
                if vis_task == 'multi-modality_det':
                    import matplotlib.pyplot as plt
                    is_inline = 'inline' in plt.get_backend()
                    img = drawn_img if drawn_img_3d is None else drawn_img_3d
                    self._init_manager(win_name)
                    fig = self.manager.canvas.figure
                    # remove white edges by set subplot margin
                    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
                    fig.clear()
                    ax = fig.add_subplot()
                    ax.axis(False)
                    ax.imshow(img)
                    self.manager.canvas.draw()
                    if is_inline:
                        return fig
                    else:
                        fig.show()
                    self.manager.canvas.flush_events()
                else:
                    if drawn_img_3d is not None:
                        super().show(drawn_img_3d, win_name, wait_time,
                                     continue_key)
                    if drawn_img is not None:
                        super().show(drawn_img, win_name, wait_time,
                                     continue_key)

        if hasattr(self, 'o3d_vis'):
            if hasattr(self, 'view_port'):
                self.view_control.convert_from_pinhole_camera_parameters(
                    self.view_port)
            self.flag_exit = not self.o3d_vis.poll_events()
            self.o3d_vis.update_renderer()
            # if not hasattr(self, 'view_control'):
            #     self.o3d_vis.create_window()
            #     self.view_control = self.o3d_vis.get_view_control()
            self.view_port = \
                self.view_control.convert_to_pinhole_camera_parameters()  # noqa: E501
            if wait_time != -1:
                self.last_time = time.time()
                while time.time(
                ) - self.last_time < wait_time and self.o3d_vis.poll_events():
                    self.o3d_vis.update_renderer()
                    self.view_port = \
                        self.view_control.convert_to_pinhole_camera_parameters()  # noqa: E501
                while self.flag_pause and self.o3d_vis.poll_events():
                    self.o3d_vis.update_renderer()
                    self.view_port = \
                        self.view_control.convert_to_pinhole_camera_parameters()  # noqa: E501

            else:
                while not self.flag_next and self.o3d_vis.poll_events():
                    self.o3d_vis.update_renderer()
                    self.view_port = \
                        self.view_control.convert_to_pinhole_camera_parameters()  # noqa: E501
                self.flag_next = False
            self.o3d_vis.clear_geometries()
            try:
                del self.pcd
            except (KeyError, AttributeError):
                pass
            if save_path is not None:
                if not (save_path.endswith('.png')
                        or save_path.endswith('.jpg')):
                    save_path += '.png'
                self.o3d_vis.capture_screen_image(save_path)
            if self.flag_exit:
                self.o3d_vis.destroy_window()
                self.o3d_vis.close()
                self._clear_o3d_vis()
                sys.exit(0)

    def escape_callback(self, vis):
        self.o3d_vis.clear_geometries()
        self.o3d_vis.destroy_window()
        self.o3d_vis.close()
        self._clear_o3d_vis()
        sys.exit(0)

    def space_action_callback(self, vis, action, mods):
        if action == 1:
            if self.flag_pause:
                print_log(
                    'Playback continued, press [SPACE] to pause.',
                    logger='current')
            else:
                print_log(
                    'Playback paused, press [SPACE] to continue.',
                    logger='current')
            self.flag_pause = not self.flag_pause
        return True

    def right_callback(self, vis):
        self.flag_next = True
        return False

    # TODO: Support Visualize the 3D results from image and point cloud
    # respectively
    @master_only
    def add_datasample(self,
                       name: str,
                       data_input: dict,
                       data_sample: Optional[Det3DDataSample] = None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       show: bool = False,
                       wait_time: float = 0,
                       out_file: Optional[str] = None,
                       o3d_save_path: Optional[str] = None,
                       vis_task: str = 'mono_det',
                       pred_score_thr: float = 0.3,
                       step: int = 0,
                       show_pcd_rgb: bool = False) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are displayed
          in a stitched image where the left image is the ground truth and the
          right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and the images
          will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be saved to
          ``out_file``. It is usually used when the display is not available.

        Args:
            name (str): The image identifier.
            data_input (dict): It should include the point clouds or image
                to draw.
            data_sample (:obj:`Det3DDataSample`, optional): Prediction
                Det3DDataSample. Defaults to None.
            draw_gt (bool): Whether to draw GT Det3DDataSample.
                Defaults to True.
            draw_pred (bool): Whether to draw Prediction Det3DDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn point clouds and image.
                Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str, optional): Path to output file. Defaults to None.
            o3d_save_path (str, optional): Path to save open3d visualized
                results. Defaults to None.
            vis_task (str): Visualization task. Defaults to 'mono_det'.
            pred_score_thr (float): The threshold to visualize the bboxes
                and masks. Defaults to 0.3.
            step (int): Global step value to record. Defaults to 0.
            show_pcd_rgb (bool): Whether to show RGB point cloud. Defaults to
                False.
        """
        assert vis_task in (
            'mono_det', 'multi-view_det', 'lidar_det', 'lidar_seg',
            'multi-modality_det'), f'got unexpected vis_task {vis_task}.'
        classes = self.dataset_meta.get('classes', None)
        # For object detection datasets, no palette is saved
        palette = self.dataset_meta.get('palette', None)
        ignore_index = self.dataset_meta.get('ignore_index', None)
        if vis_task == 'lidar_seg' and ignore_index is not None and 'pts_semantic_mask' in data_sample.gt_pts_seg:  # noqa: E501
            keep_index = data_sample.gt_pts_seg.pts_semantic_mask != ignore_index  # noqa: E501
        else:
            keep_index = None

        gt_data_3d = None
        pred_data_3d = None
        gt_img_data = None
        pred_img_data = None

        if not hasattr(self, 'o3d_vis') and vis_task in [
                'multi-view_det', 'lidar_det', 'lidar_seg',
                'multi-modality_det'
        ]:
            self.o3d_vis = self._initialize_o3d_vis(show=show)

        if draw_gt and data_sample is not None:
            if 'gt_instances_3d' in data_sample:
                gt_data_3d = self._draw_instances_3d(
                    data_input, data_sample.gt_instances_3d,
                    data_sample.metainfo, vis_task, show_pcd_rgb, palette)
            if 'gt_instances' in data_sample:
                if len(data_sample.gt_instances) > 0:
                    assert 'img' in data_input
                    img = data_input['img']
                    if isinstance(data_input['img'], Tensor):
                        img = data_input['img'].permute(1, 2, 0).numpy()
                        img = img[..., [2, 1, 0]]  # bgr to rgb
                    gt_img_data = self._draw_instances(
                        img, data_sample.gt_instances, classes, palette)
            if 'gt_pts_seg' in data_sample and vis_task == 'lidar_seg':
                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing semantic ' \
                                            'segmentation results.'
                assert 'points' in data_input
                self._draw_pts_sem_seg(data_input['points'],
                                       data_sample.gt_pts_seg, palette,
                                       keep_index)

        if draw_pred and data_sample is not None:
            if 'pred_instances_3d' in data_sample:
                pred_instances_3d = data_sample.pred_instances_3d
                # .cpu can not be used for BaseInstance3DBoxes
                # so we need to use .to('cpu')
                pred_instances_3d = pred_instances_3d[
                    pred_instances_3d.scores_3d > pred_score_thr].to('cpu')
                pred_data_3d = self._draw_instances_3d(data_input,
                                                       pred_instances_3d,
                                                       data_sample.metainfo,
                                                       vis_task, show_pcd_rgb,
                                                       palette)
            if 'pred_instances' in data_sample:
                if 'img' in data_input and len(data_sample.pred_instances) > 0:
                    pred_instances = data_sample.pred_instances
                    pred_instances = pred_instances[
                        pred_instances.scores > pred_score_thr].cpu()
                    img = data_input['img']
                    if isinstance(data_input['img'], Tensor):
                        img = data_input['img'].permute(1, 2, 0).numpy()
                        img = img[..., [2, 1, 0]]  # bgr to rgb
                    pred_img_data = self._draw_instances(
                        img, pred_instances, classes, palette)
            if 'pred_pts_seg' in data_sample and vis_task == 'lidar_seg':
                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing semantic ' \
                                            'segmentation results.'
                assert 'points' in data_input
                self._draw_pts_sem_seg(data_input['points'],
                                       data_sample.pred_pts_seg, palette,
                                       keep_index)

        # monocular 3d object detection image
        if vis_task in ['mono_det', 'multi-modality_det']:
            if gt_data_3d is not None and pred_data_3d is not None:
                drawn_img_3d = np.concatenate(
                    (gt_data_3d['img'], pred_data_3d['img']), axis=1)
            elif gt_data_3d is not None:
                drawn_img_3d = gt_data_3d['img']
            elif pred_data_3d is not None:
                drawn_img_3d = pred_data_3d['img']
            else:  # both instances of gt and pred are empty
                drawn_img_3d = None
        else:
            drawn_img_3d = None

        # 2d object detection image
        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        elif pred_img_data is not None:
            drawn_img = pred_img_data
        else:
            drawn_img = None

        if show:
            self.show(
                o3d_save_path,
                drawn_img_3d,
                drawn_img,
                win_name=name,
                wait_time=wait_time,
                vis_task=vis_task)

        if out_file is not None:
            # check the suffix of the name of image file
            if not (out_file.endswith('.png') or out_file.endswith('.jpg')):
                out_file = f'{out_file}.png'
            if drawn_img_3d is not None:
                mmcv.imwrite(drawn_img_3d[..., ::-1], out_file)
            if drawn_img is not None:
                mmcv.imwrite(drawn_img[..., ::-1],
                             out_file[:-4] + '_2d' + out_file[-4:])
        else:
            self.add_image(name, drawn_img_3d, step)
