# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import mmcv
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from mmdet.visualization import DetLocalVisualizer
from mmengine.dist import master_only
from mmengine.structures import InstanceData
from mmengine.visualization.utils import check_type, tensor2ndarray
from torch import Tensor

from mmdet3d.registry import VISUALIZERS
from mmdet3d.structures import (BaseInstance3DBoxes, CameraInstance3DBoxes,
                                Coord3DMode, DepthInstance3DBoxes,
                                Det3DDataSample, LiDARInstance3DBoxes,
                                PointData, points_cam2img)
from mmdet3d.structures.bbox_3d.box_3d_mode import Box3DMode
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
        points (numpy.array, shape=[N, 3+C]): points to visualize.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to None.
        pcd_mode (int): The point cloud mode (coordinates):
            0 represents LiDAR, 1 represents CAMERA, 2
            represents Depth. Defaults to 0.
        vis_backends (list, optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
        bbox_color (str, tuple(int), optional): Color of bbox lines.
            The tuple of color should be in BGR order. Defaults to None.
        text_color (str, tuple(int), optional): Color of texts.
            The tuple of color should be in BGR order.
            Defaults to (200, 200, 200).
        mask_color (str, tuple(int), optional): Color of masks.
            The tuple of color should be in BGR order.
            Defaults to None.
        line_width (int, float): The linewidth of lines.
            Defaults to 3.
        frame_cfg (dict): The coordinate frame config while Open3D
            visualization initialization.
            Defaults to dict(size=1, origin=[0, 0, 0]).
        alpha (int, float): The transparency of bboxes or mask.
                Defaults to 0.8.

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from mmengine.structures import InstanceData
        >>> from mmdet3d.structures import Det3DDataSample
        >>> from mmdet3d.visualization import Det3DLocalVisualizer

        >>> det3d_local_visualizer = Det3DLocalVisualizer()
        >>> image = np.random.randint(0, 256,
        ...                     size=(10, 12, 3)).astype('uint8')
        >>> points = np.random.rand((1000, ))
        >>> gt_instances_3d = InstanceData()
        >>> gt_instances_3d.bboxes_3d = BaseInstance3DBoxes(torch.rand((5, 7)))
        >>> gt_instances_3d.labels_3d = torch.randint(0, 2, (5,))
        >>> gt_det3d_data_sample = Det3DDataSample()
        >>> gt_det3d_data_sample.gt_instances_3d = gt_instances_3d
        >>> data_input = dict(img=image, points=points)
        >>> det3d_local_visualizer.add_datasample('3D Scene', data_input,
        ...                         gt_det3d_data_sample)
    """

    def __init__(self,
                 name: str = 'visualizer',
                 points: Optional[np.ndarray] = None,
                 image: Optional[np.ndarray] = None,
                 pcd_mode: int = 0,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 bbox_color: Optional[Union[str, Tuple[int]]] = None,
                 text_color: Optional[Union[str,
                                            Tuple[int]]] = (200, 200, 200),
                 mask_color: Optional[Union[str, Tuple[int]]] = None,
                 line_width: Union[int, float] = 3,
                 frame_cfg: dict = dict(size=1, origin=[0, 0, 0]),
                 alpha: float = 0.8):
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
        self.pts_seg_num = 0

    def _clear_o3d_vis(self) -> None:
        """Clear open3d vis."""

        if hasattr(self, 'o3d_vis'):
            del self.o3d_vis
            del self.pcd
            del self.points_colors

    def _initialize_o3d_vis(self, frame_cfg) -> Visualizer:
        """Initialize open3d vis according to frame_cfg.

        Args:
            frame_cfg (dict): The config to create coordinate frame
                in open3d vis.

        Returns:
            :obj:`o3d.visualization.Visualizer`: Created open3d vis.
        """
        if o3d is None or geometry is None:
            raise ImportError(
                'Please run "pip install open3d" to install open3d first.')
        o3d_vis = o3d.visualization.Visualizer()
        o3d_vis.create_window()
        # create coordinate frame
        mesh_frame = geometry.TriangleMesh.create_coordinate_frame(**frame_cfg)
        o3d_vis.add_geometry(mesh_frame)
        return o3d_vis

    @master_only
    def set_points(self,
                   points: np.ndarray,
                   pcd_mode: int = 0,
                   vis_mode: str = 'replace',
                   frame_cfg: dict = dict(size=1, origin=[0, 0, 0]),
                   points_color: Tuple = (0.5, 0.5, 0.5),
                   points_size: int = 2,
                   mode: str = 'xyz') -> None:
        """Set the points to draw.

        Args:
            points (numpy.array, shape=[N, 3+C]):
                points to visualize.
            pcd_mode (int): The point cloud mode (coordinates):
                0 represents LiDAR, 1 represents CAMERA, 2
                represents Depth. Defaults to 0.
            vis_mode (str): The visualization mode in Open3D:
                'replace': Replace the existing point cloud with
                    input point cloud.
                'add': Add input point cloud into existing point
                    cloud.
                Defaults to 'replace'.
            frame_cfg (dict): The coordinate frame config while Open3D
                visualization initialization.
                Defaults to dict(size=1, origin=[0, 0, 0]).
            point_color (tuple[float], optional): the color of points.
                Default: (0.5, 0.5, 0.5).
            points_size (int, optional): the size of points to show
                on visualizer. Default: 2.
            mode (str, optional):  indicate type of the input points,
                available mode ['xyz', 'xyzrgb']. Default: 'xyz'.
        """
        assert points is not None
        assert vis_mode in ('replace', 'add')
        check_type('points', points, np.ndarray)

        if not hasattr(self, 'o3d_vis'):
            self.o3d_vis = self._initialize_o3d_vis(frame_cfg)

        # for now we convert points into depth mode for visualization
        if pcd_mode != Coord3DMode.DEPTH:
            points = Coord3DMode.convert(points, pcd_mode, Coord3DMode.DEPTH)

        if hasattr(self, 'pcd') and vis_mode != 'add':
            self.o3d_vis.remove_geometry(self.pcd)

        # set points size in Open3D
        self.o3d_vis.get_render_option().point_size = points_size

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

        pcd.colors = o3d.utility.Vector3dVector(points_colors)
        self.o3d_vis.add_geometry(pcd)
        self.pcd = pcd
        self.points_colors = points_colors

    # TODO: assign 3D Box color according to pred / GT labels
    # We draw GT / pred bboxes on the same point cloud scenes
    # for better detection performance comparison
    def draw_bboxes_3d(self,
                       bboxes_3d: BaseInstance3DBoxes,
                       bbox_color=(0, 1, 0),
                       points_in_box_color=(1, 0, 0),
                       rot_axis=2,
                       center_mode='lidar_bottom',
                       mode='xyz'):
        """Draw bbox on visualizer and change the color of points inside
        bbox3d.

        Args:
            bboxes_3d (:obj:`BaseInstance3DBoxes`, shape=[M, 7]):
                3d bbox (x, y, z, x_size, y_size, z_size, yaw) to visualize.
            bbox_color (tuple[float], optional): the color of 3D bboxes.
                Default: (0, 1, 0).
            points_in_box_color (tuple[float], optional):
                the color of points inside 3D bboxes. Default: (1, 0, 0).
            rot_axis (int, optional): rotation axis of 3D bboxes.
                Default: 2.
            center_mode (bool, optional): Indicates the center of bbox is
                bottom center or gravity center. available mode
                ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.
            mode (str, optional):  Indicates type of input points,
                available mode ['xyz', 'xyzrgb']. Default: 'xyz'.
        """
        # Before visualizing the 3D Boxes in point cloud scene
        # we need to convert the boxes to Depth mode
        check_type('bboxes', bboxes_3d, BaseInstance3DBoxes)

        if not isinstance(bboxes_3d, DepthInstance3DBoxes):
            bboxes_3d = bboxes_3d.convert_to(Box3DMode.DEPTH)

        # convert bboxes to numpy dtype
        bboxes_3d = tensor2ndarray(bboxes_3d.tensor)

        in_box_color = np.array(points_in_box_color)

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
            line_set.paint_uniform_color(bbox_color)
            # draw bboxes on visualizer
            self.o3d_vis.add_geometry(line_set)

            # change the color of points which are in box
            if self.pcd is not None and mode == 'xyz':
                indices = box3d.get_point_indices_within_bounding_box(
                    self.pcd.points)
                self.points_colors[indices] = in_box_color

        # update points colors
        if self.pcd is not None:
            self.pcd.colors = o3d.utility.Vector3dVector(self.points_colors)
            self.o3d_vis.update_geometry(self.pcd)

    def set_bev_image(self,
                      bev_image: Optional[np.ndarray] = None,
                      bev_shape: Optional[int] = 900) -> None:
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
                        edge_colors: Union[str, tuple, List[str],
                                           List[tuple]] = 'o',
                        line_styles: Union[str, List[str]] = '-',
                        line_widths: Union[Union[int, float],
                                           List[Union[int, float]]] = 1,
                        face_colors: Union[str, tuple, List[str],
                                           List[tuple]] = 'none',
                        alpha: Union[int, float] = 1):
        """Draw projected 3D boxes on the image.

        Args:
            bboxes_3d (:obj:`BaseInstance3DBoxes`, shape=[M, 7]):
                3d bbox (x, y, z, x_size, y_size, z_size, yaw) to visualize.
            scale (dict): Value to scale the bev bboxes for better
                visualization. Defaults to 15.
            edge_colors (Union[str, tuple, List[str], List[tuple]]): The
                colors of bboxes. ``colors`` can have the same length with
                lines or just single value. If ``colors`` is single value, all
                the lines will have the same colors. Refer to `matplotlib.
                colors` for full list of formats that are accepted.
                Defaults to 'o'.
            line_styles (Union[str, List[str]]): The linestyle
                of lines. ``line_styles`` can have the same length with
                texts or just single value. If ``line_styles`` is single
                value, all the lines will have the same linestyle.
                Reference to
                https://matplotlib.org/stable/api/collections_api.html?highlight=collection#matplotlib.collections.AsteriskPolygonCollection.set_linestyle
                for more details. Defaults to '-'.
            line_widths (Union[Union[int, float], List[Union[int, float]]]):
                The linewidth of lines. ``line_widths`` can have
                the same length with lines or just single value.
                If ``line_widths`` is single value, all the lines will
                have the same linewidth. Defaults to 2.
            face_colors (Union[str, tuple, List[str], List[tuple]]):
                The face colors. Default to 'none'.
            alpha (Union[int, float]): The transparency of bboxes.
                Defaults to 1.
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
    def draw_points_on_image(
            self,
            points: Union[np.ndarray, Tensor],
            pts2img: np.ndarray,
            sizes: Optional[Union[np.ndarray, Tensor, int]] = 10) -> None:
        """Draw projected points on the image.

        Args:
            positions (Union[np.ndarray, torch.Tensor]): Positions to draw.
            pts2imgs (np,ndarray): The transformatino matrix from the
                coordinate of point cloud to image plane.
            sizes (Optional[Union[np.ndarray, torch.Tensor, int]]): The
                marker size. Default to 10.
        """
        check_type('points', points, (np.ndarray, Tensor))
        points = tensor2ndarray(points)
        assert self._image is not None, 'Please set image using `set_image`'
        projected_points = points_cam2img(points, pts2img, with_depth=True)
        depths = projected_points[:, 2]
        colors = (depths % 20) / 20
        # use colormap to obtain the render color
        color_map = plt.get_cmap('jet')
        self.ax_save.scatter(
            projected_points[:, 0],
            projected_points[:, 1],
            c=colors,
            cmap=color_map,
            s=sizes,
            alpha=0.5,
            edgecolors='none')

    # TODO: set bbox color according to palette
    @master_only
    def draw_proj_bboxes_3d(self,
                            bboxes_3d: BaseInstance3DBoxes,
                            input_meta: dict,
                            edge_colors: Union[str, tuple, List[str],
                                               List[tuple]] = 'royalblue',
                            line_styles: Union[str, List[str]] = '-',
                            line_widths: Union[Union[int, float],
                                               List[Union[int, float]]] = 2,
                            face_colors: Union[str, tuple, List[str],
                                               List[tuple]] = 'royalblue',
                            alpha: Union[int, float] = 0.4):
        """Draw projected 3D boxes on the image.

        Args:
            bbox3d (:obj:`BaseInstance3DBoxes`, shape=[M, 7]):
                3d bbox (x, y, z, x_size, y_size, z_size, yaw) to visualize.
            input_meta (dict): Input meta information.
            edge_colors (Union[str, tuple, List[str], List[tuple]]): The
                colors of bboxes. ``colors`` can have the same length with
                lines or just single value. If ``colors`` is single value, all
                the lines will have the same colors. Refer to `matplotlib.
                colors` for full list of formats that are accepted.
                Defaults to 'royalblue'.
            line_styles (Union[str, List[str]]): The linestyle
                of lines. ``line_styles`` can have the same length with
                texts or just single value. If ``line_styles`` is single
                value, all the lines will have the same linestyle.
                Reference to
                https://matplotlib.org/stable/api/collections_api.html?highlight=collection#matplotlib.collections.AsteriskPolygonCollection.set_linestyle
                for more details. Defaults to '-'.
            line_widths (Union[Union[int, float], List[Union[int, float]]]):
                The linewidth of lines. ``line_widths`` can have
                the same length with lines or just single value.
                If ``line_widths`` is single value, all the lines will
                have the same linewidth. Defaults to 2.
            face_colors (Union[str, tuple, List[str], List[tuple]]):
                The face colors. Default to 'royalblue'.
            alpha (Union[int, float]): The transparency of bboxes.
                Defaults to 0.4.
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

        corners_2d = proj_bbox3d_to_img(bboxes_3d, input_meta)

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
            edgecolors=edge_colors,
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
            face_colors=face_colors)

    @master_only
    def draw_seg_mask(self, seg_mask_colors: np.array):
        """Add segmentation mask to visualizer via per-point colorization.

        Args:
            seg_mask_colors (numpy.array, shape=[N, 6]):
                The segmentation mask whose first 3 dims are point coordinates
                and last 3 dims are converted colors.
        """
        # we can't draw the colors on existing points
        # in case gt and pred mask would overlap
        # instead we set a large offset along x-axis for each seg mask
        self.pts_seg_num += 1
        offset = (np.array(self.pcd.points).max(0) -
                  np.array(self.pcd.points).min(0))[0] * 1.2 * self.pts_seg_num
        mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
            size=1, origin=[offset, 0, 0])  # create coordinate frame for seg
        self.o3d_vis.add_geometry(mesh_frame)
        seg_points = copy.deepcopy(seg_mask_colors)
        seg_points[:, 0] += offset
        self.set_points(seg_points, pcd_mode=2, vis_mode='add', mode='xyzrgb')

    def _draw_instances_3d(self, data_input: dict, instances: InstanceData,
                           input_meta: dict, vis_task: str,
                           palette: Optional[List[tuple]]):
        """Draw 3D instances of GT or prediction.

        Args:
            data_input (dict): The input dict to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
            metainfo (dict): Meta information.
            vis_task (str): Visualiztion task, it includes:
                'lidar_det', 'multi-modality_det', 'mono_det'.

        Returns:
            dict: the drawn point cloud and image which channel is RGB.
        """

        bboxes_3d = instances.bboxes_3d  # BaseInstance3DBoxes

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

            self.set_points(points, pcd_mode=2)
            self.draw_bboxes_3d(bboxes_3d_depth)

            data_3d['bboxes_3d'] = tensor2ndarray(bboxes_3d_depth.tensor)
            data_3d['points'] = points

        if vis_task in ['mono_det', 'multi-modality_det']:
            assert 'img' in data_input
            img = data_input['img']
            if isinstance(data_input['img'], Tensor):
                img = img.permute(1, 2, 0).numpy()
                img = img[..., [2, 1, 0]]  # bgr to rgb
            self.set_image(img)
            self.draw_proj_bboxes_3d(bboxes_3d, input_meta)
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
                          ignore_index: Optional[int] = None):
        """Draw 3D semantic mask of GT or prediction.

        Args:
            points (Tensor | np.ndarray): The input point
                cloud to draw.
            pts_seg (:obj:`PointData`): Data structure for
                pixel-level annotations or predictions.
            palette (List[tuple], optional): Palette information
                corresponding to the category. Defaults to None.
            ignore_index (int, optional): Ignore category.
                Defaults to None.

        Returns:
            dict: the drawn points with color.
        """
        check_type('points', points, (np.ndarray, Tensor))

        points = tensor2ndarray(points)
        pts_sem_seg = tensor2ndarray(pts_seg.pts_semantic_mask)
        palette = np.array(palette)

        if ignore_index is not None:
            points = points[pts_sem_seg != ignore_index]
            pts_sem_seg = pts_sem_seg[pts_sem_seg != ignore_index]

        pts_color = palette[pts_sem_seg]
        seg_color = np.concatenate([points[:, :3], pts_color], axis=1)

        self.set_points(points, pcd_mode=2, vis_mode='add')
        self.draw_seg_mask(seg_color)

    @master_only
    def show(self,
             save_path: Optional[str] = None,
             drawn_img_3d: Optional[np.ndarray] = None,
             drawn_img: Optional[np.ndarray] = None,
             win_name: str = 'image',
             wait_time: int = 0,
             continue_key=' ') -> None:
        """Show the drawn point cloud/image.

        Args:
            save_path (str, optional): Path to save open3d visualized results.
                Default: None.
            drawn_img (np.ndarray, optional): The image to show. If drawn_img
                is None, it will show the image got by Visualizer. Defaults
                to None.
            win_name (str):  The image title. Defaults to 'image'.
            wait_time (int): Delay in milliseconds. 0 is the special
                value that means "forever". Defaults to 0.
            continue_key (str): The key for users to continue. Defaults to
                the space key.
        """
        if hasattr(self, 'o3d_vis'):
            self.o3d_vis.run()
            if save_path is not None:
                self.o3d_vis.capture_screen_image(save_path)
            self.o3d_vis.destroy_window()
            self._clear_o3d_vis()

        if hasattr(self, '_image'):
            if drawn_img_3d is None:
                super().show(drawn_img_3d, win_name, wait_time, continue_key)
            super().show(drawn_img, win_name, wait_time, continue_key)

    # TODO: Support Visualize the 3D results from image and point cloud
    # respectively
    @master_only
    def add_datasample(self,
                       name: str,
                       data_input: dict,
                       data_sample: Optional['Det3DDataSample'] = None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       show: bool = False,
                       wait_time: float = 0,
                       out_file: Optional[str] = None,
                       o3d_save_path: Optional[str] = None,
                       vis_task: str = 'mono_det',
                       pred_score_thr: float = 0.3,
                       step: int = 0) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be saved to
        ``out_file``. It is usually used when the display is not available.

        Args:
            name (str): The image identifier.
            data_input (dict): It should include the point clouds or image
                to draw.
            data_sample (:obj:`Det3DDataSample`, optional): Prediction
                Det3DDataSample. Defaults to None.
            draw_gt (bool): Whether to draw GT Det3DDataSample.
                Default to True.
            draw_pred (bool): Whether to draw Prediction Det3DDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn point clouds and
                image. Default to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str): Path to output file. Defaults to None.
            o3d_save_path (str, optional): Path to save open3d visualized
                results Default: None.
            vis-task (str): Visualization task. Defaults to 'mono_det'.
            pred_score_thr (float): The threshold to visualize the bboxes
                and masks. Defaults to 0.3.
            step (int): Global step value to record. Defaults to 0.
        """
        classes = self.dataset_meta.get('CLASSES', None)
        # For object detection datasets, no PALETTE is saved
        palette = self.dataset_meta.get('PALETTE', None)
        ignore_index = self.dataset_meta.get('ignore_index', None)

        gt_data_3d = None
        pred_data_3d = None
        gt_img_data = None
        pred_img_data = None

        if draw_gt and data_sample is not None:
            if 'gt_instances_3d' in data_sample:
                gt_data_3d = self._draw_instances_3d(
                    data_input, data_sample.gt_instances_3d,
                    data_sample.metainfo, vis_task, palette)
            if 'gt_instances' in data_sample:
                if len(data_sample.gt_instances) > 0:
                    assert 'img' in data_input
                    if isinstance(data_input['img'], Tensor):
                        img = data_input['img'].permute(1, 2, 0).numpy()
                        img = img[..., [2, 1, 0]]  # bgr to rgb
                    gt_img_data = self._draw_instances(
                        img, data_sample.gt_instances, classes, palette)
            if 'gt_pts_seg' in data_sample and vis_task == 'seg':
                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing panoptic ' \
                                            'segmentation results.'
                assert 'points' in data_input
                self._draw_pts_sem_seg(data_input['points'],
                                       data_sample.pred_pts_seg, palette,
                                       ignore_index)

        if draw_pred and data_sample is not None:
            if 'pred_instances_3d' in data_sample:
                pred_instances_3d = data_sample.pred_instances_3d
                # .cpu can not be used for BaseInstancesBoxes3D
                # so we need to use .to('cpu')
                pred_instances_3d = pred_instances_3d[
                    pred_instances_3d.scores_3d > pred_score_thr].to('cpu')
                pred_data_3d = self._draw_instances_3d(data_input,
                                                       pred_instances_3d,
                                                       data_sample.metainfo,
                                                       vis_task, palette)
            if 'pred_instances' in data_sample:
                if 'img' in data_input and len(data_sample.pred_instances) > 0:
                    pred_instances = data_sample.pred_instances
                    pred_instances = pred_instances_3d[
                        pred_instances.scores > pred_score_thr].cpu()
                    if isinstance(data_input['img'], Tensor):
                        img = data_input['img'].permute(1, 2, 0).numpy()
                        img = img[..., [2, 1, 0]]  # bgr to rgb
                    pred_img_data = self._draw_instances(
                        img, pred_instances, classes, palette)
            if 'pred_pts_seg' in data_sample and vis_task == 'lidar_seg':
                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing panoptic ' \
                                            'segmentation results.'
                assert 'points' in data_input
                self._draw_pts_sem_seg(data_input['points'],
                                       data_sample.pred_pts_seg, palette,
                                       ignore_index)

        # monocular 3d object detection image
        if vis_task in ['mono_det', 'multi-modality_det']:
            if gt_data_3d is not None and pred_data_3d is not None:
                drawn_img_3d = np.concatenate(
                    (gt_data_3d['img'], pred_data_3d['img']), axis=1)
            elif gt_data_3d is not None:
                drawn_img_3d = gt_data_3d['img']
            elif pred_data_3d is not None:
                drawn_img_3d = pred_data_3d['img']
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
                wait_time=wait_time)

        if out_file is not None:
            if drawn_img_3d is not None:
                mmcv.imwrite(drawn_img_3d[..., ::-1], out_file)
            if drawn_img is not None:
                mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img_3d, step)
