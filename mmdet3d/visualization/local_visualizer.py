# Copyright (c) OpenMMLab. All rights reserved.
import copy
from os import path as osp
from typing import Dict, List, Optional, Tuple, Union

import mmcv
import numpy as np
from mmengine.dist import master_only
from torch import Tensor

from mmdet.visualization import DetLocalVisualizer

try:
    import open3d as o3d
    from open3d import geometry
except ImportError:
    raise ImportError(
        'Please run "pip install open3d" to install open3d first.')

from mmengine.structures import InstanceData
from mmengine.visualization.utils import check_type, tensor2ndarray

from mmdet3d.registry import VISUALIZERS
from mmdet3d.structures import (BaseInstance3DBoxes, CameraInstance3DBoxes,
                                Coord3DMode, DepthInstance3DBoxes,
                                Det3DDataSample, LiDARInstance3DBoxes,
                                PointData)
from .vis_utils import (proj_camera_bbox3d_to_img, proj_depth_bbox3d_to_img,
                        proj_lidar_bbox3d_to_img, to_depth_mode, write_obj,
                        write_oriented_bbox)


@VISUALIZERS.register_module()
class Det3DLocalVisualizer(DetLocalVisualizer):
    """MMDetection3D Local Visualizer.

    - 3D detection and segmentation drawing methods

      - draw_bboxes_3d: draw 3D bounding boxes on point clouds
      - draw_proj_bboxes_3d: draw projected 3D bounding boxes on image
      - draw_seg_mask: draw segmentation mask via per-point colorization

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to None.
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
        vis_cfg (dict): The coordinate frame config while Open3D
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
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 bbox_color: Optional[Union[str, Tuple[int]]] = None,
                 text_color: Optional[Union[str,
                                            Tuple[int]]] = (200, 200, 200),
                 mask_color: Optional[Union[str, Tuple[int]]] = None,
                 line_width: Union[int, float] = 3,
                 vis_cfg: dict = dict(size=1, origin=[0, 0, 0]),
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
        self.o3d_vis = self._initialize_o3d_vis(vis_cfg)
        self.seg_num = 0

    def _initialize_o3d_vis(self, vis_cfg) -> tuple:
        """Build open3d vis according to vis_cfg.

        Args:
            vis_cfg (dict): The config to build open3d vis.

        Returns:
             tuple: build open3d vis.
        """
        # init open3d visualizer
        o3d_vis = o3d.visualization.Visualizer()
        o3d_vis.create_window()
        # create coordinate frame
        mesh_frame = geometry.TriangleMesh.create_coordinate_frame(**vis_cfg)
        o3d_vis.add_geometry(mesh_frame)

        return o3d_vis

    @master_only
    def set_points(self,
                   points: np.ndarray,
                   pcd_mode: int = 0,
                   vis_task: str = 'det',
                   points_color: Tuple = (0.5, 0.5, 0.5),
                   points_size: int = 2,
                   mode: str = 'xyz') -> None:
        """Set the points to draw.

        Args:
            points (numpy.array, shape=[N, 3+C]):
                points to visualize.
            pcd_mode (int): The point cloud mode (coordinates):
                0 represents LiDAR, 1 represents CAMERA, 2
                represents Depth.
            vis_task (str): Visualiztion task, it includes:
                'det', 'multi_modality-det', 'mono-det', 'seg'.
            point_color (tuple[float], optional): the color of points.
                Default: (0.5, 0.5, 0.5).
            points_size (int, optional): the size of points to show
                on visualizer. Default: 2.
            mode (str, optional):  indicate type of the input points,
                available mode ['xyz', 'xyzrgb']. Default: 'xyz'.
        """
        assert points is not None
        check_type('points', points, np.ndarray)

        # for now we convert points into depth mode for visualization
        if pcd_mode != Coord3DMode.DEPTH:
            points = Coord3DMode.convert(points, pcd_mode, Coord3DMode.DEPTH)

        if hasattr(self, 'pcd') and vis_task != 'seg':
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
                       bboxes_3d: DepthInstance3DBoxes,
                       bbox_color=(0, 1, 0),
                       points_in_box_color=(1, 0, 0),
                       rot_axis=2,
                       center_mode='lidar_bottom',
                       mode='xyz'):
        """Draw bbox on visualizer and change the color of points inside
        bbox3d.

        Args:
            bboxes_3d (:obj:`DepthInstance3DBoxes`, shape=[M, 7]):
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
        check_type('bboxes', bboxes_3d, (DepthInstance3DBoxes))

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

    # TODO: set bbox color according to palette
    def draw_proj_bboxes_3d(self,
                            bboxes_3d: BaseInstance3DBoxes,
                            input_meta: dict,
                            bbox_color: Tuple[float] = 'b',
                            line_styles: Union[str, List[str]] = '-',
                            line_widths: Union[Union[int, float],
                                               List[Union[int, float]]] = 1):
        """Draw projected 3D boxes on the image.

        Args:
            bbox3d (:obj:`BaseInstance3DBoxes`, shape=[M, 7]):
                3d bbox (x, y, z, x_size, y_size, z_size, yaw) to visualize.
            input_meta (dict): Input meta information.
            bbox_color (tuple[float], optional): the color of bbox.
                Default: (0, 1, 0).
            line_styles (Union[str, List[str]]): The linestyle
                of lines. ``line_styles`` can have the same length with
                texts or just single value. If ``line_styles`` is single
                value, all the lines will have the same linestyle.
            line_widths (Union[Union[int, float], List[Union[int, float]]]):
                The linewidth of lines. ``line_widths`` can have
                the same length with lines or just single value.
                If ``line_widths`` is single value, all the lines will
                have the same linewidth. Defaults to 2.
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

        # (num_bboxes_3d, 8, 2)
        proj_bboxes_3d = proj_bbox3d_to_img(bboxes_3d, input_meta)
        num_bboxes_3d = proj_bboxes_3d.shape[0]

        line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                        (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))

        # TODO: assign each projected 3d bboxes color
        # according to pred / gt class.
        for i in range(num_bboxes_3d):
            x_datas = []
            y_datas = []
            corners = proj_bboxes_3d[i].astype(np.int)  # (8, 2)
            for start, end in line_indices:
                x_datas.append([corners[start][0], corners[end][0]])
                y_datas.append([corners[start][1], corners[end][1]])
            x_datas = np.array(x_datas)
            y_datas = np.array(y_datas)
            self.draw_lines(x_datas, y_datas, bbox_color, line_styles,
                            line_widths)

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
        self.seg_num += 1
        offset = (np.array(self.pcd.points).max(0) -
                  np.array(self.pcd.points).min(0))[0] * 1.2 * self.seg_num
        mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
            size=1, origin=[offset, 0, 0])  # create coordinate frame for seg
        self.o3d_vis.add_geometry(mesh_frame)
        seg_points = copy.deepcopy(seg_mask_colors)
        seg_points[:, 0] += offset
        self.set_points(seg_points, vis_task='seg', pcd_mode=2, mode='xyzrgb')

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
                'det', 'multi_modality-det', 'mono-det'.

        Returns:
            dict: the drawn point cloud and image which channel is RGB.
        """

        bboxes_3d = instances.bboxes_3d  # BaseInstance3DBoxes

        data_3d = dict()

        if vis_task in ['det', 'multi_modality-det']:
            assert 'points' in data_input
            points = data_input['points']
            check_type('points', points, (np.ndarray, Tensor))
            points = tensor2ndarray(points)

            if not isinstance(bboxes_3d, DepthInstance3DBoxes):
                points, bboxes_3d_depth = to_depth_mode(points, bboxes_3d)
            else:
                bboxes_3d_depth = bboxes_3d.clone()

            self.set_points(points, pcd_mode=2, vis_task=vis_task)
            self.draw_bboxes_3d(bboxes_3d_depth)

            data_3d['bboxes_3d'] = tensor2ndarray(bboxes_3d_depth.tensor)
            data_3d['points'] = points

        if vis_task in ['mono-det', 'multi_modality-det']:
            assert 'img' in data_input
            img = data_input['img']
            if isinstance(data_input['img'], Tensor):
                img = img.permute(1, 2, 0).numpy()
                img = img[..., [2, 1, 0]]  # bgr to rgb
            self.set_image(img)
            self.draw_proj_bboxes_3d(bboxes_3d, input_meta)
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

        self.set_points(points, pcd_mode=2, vis_task='seg')
        self.draw_seg_mask(seg_color)

        seg_data_3d = dict(points=points, seg_color=seg_color)
        return seg_data_3d

    @master_only
    def show(self,
             vis_task: str = None,
             out_file: str = None,
             drawn_img_3d: Optional[np.ndarray] = None,
             drawn_img: Optional[np.ndarray] = None,
             win_name: str = 'image',
             wait_time: int = 0,
             continue_key=' ') -> None:
        """Show the drawn image.

        Args:
            vis_task (str): Visualiztion task, it includes:
                'det', 'multi_modality-det', 'mono-det', 'seg'.
            out_file (str): Output file path.
            drawn_img (np.ndarray, optional): The image to show. If drawn_img
                is None, it will show the image got by Visualizer. Defaults
                to None.
            win_name (str):  The image title. Defaults to 'image'.
            wait_time (int): Delay in milliseconds. 0 is the special
                value that means "forever". Defaults to 0.
            continue_key (str): The key for users to continue. Defaults to
                the space key.
        """
        if vis_task in ['det', 'multi_modality-det', 'seg']:
            self.o3d_vis.run()
            if out_file is not None:
                self.o3d_vis.capture_screen_image(out_file + '.png')
            self.o3d_vis.destroy_window()

        if vis_task in ['mono-det', 'multi_modality-det']:
            super().show(drawn_img_3d, win_name, wait_time, continue_key)

        if drawn_img is not None:
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
                       vis_task: str = 'mono-det',
                       pred_score_thr: float = 0.3,
                       step: int = 0) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn point cloud or
        image will be saved to ``out_file``. t is usually used when
        the display is not available.

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
            vis-task (str): Visualization task. Defaults to 'mono-det'.
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
        gt_seg_data_3d = None
        pred_seg_data_3d = None
        gt_img_data = None
        pred_img_data = None

        if draw_gt and data_sample is not None:
            if 'gt_instances_3d' in data_sample:
                gt_data_3d = self._draw_instances_3d(
                    data_input, data_sample.gt_instances_3d,
                    data_sample.metainfo, vis_task, palette)
            if 'gt_instances' in data_sample:
                assert 'img' in data_input
                if isinstance(data_input['img'], Tensor):
                    img = data_input['img'].permute(1, 2, 0).numpy()
                    img = img[..., [2, 1, 0]]  # bgr to rgb
                gt_img_data = self._draw_instances(img,
                                                   data_sample.gt_instances,
                                                   classes, palette)
            if 'gt_pts_seg' in data_sample:
                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing panoptic ' \
                                            'segmentation results.'
                assert 'points' in data_input
                gt_seg_data_3d = \
                    self._draw_pts_sem_seg(data_input['points'],
                                           data_sample.pred_pts_seg,
                                           palette, ignore_index)

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
            if 'pred_pts_seg' in data_sample:
                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing panoptic ' \
                                            'segmentation results.'
                assert 'points' in data_input
                pred_seg_data_3d = \
                    self._draw_pts_sem_seg(data_input['points'],
                                           data_sample.pred_pts_seg,
                                           palette, ignore_index)

        # monocular 3d object detection image
        if vis_task in ['mono-det', 'multi_modality-det']:
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
                vis_task,
                out_file,
                drawn_img_3d,
                drawn_img,
                win_name=name,
                wait_time=wait_time)

        if out_file is not None:
            if drawn_img_3d is not None:
                mmcv.imwrite(drawn_img_3d[..., ::-1], out_file + '.jpg')
            if drawn_img is not None:
                mmcv.imwrite(drawn_img[..., ::-1], out_file + '.jpg')
            if gt_data_3d is not None:
                write_obj(gt_data_3d['points'],
                          osp.join(out_file, 'points.obj'))
                write_oriented_bbox(gt_data_3d['bboxes_3d'],
                                    osp.join(out_file, 'gt_bbox.obj'))
            if pred_data_3d is not None:
                if 'points' in pred_data_3d:
                    write_obj(pred_data_3d['points'],
                              osp.join(out_file, 'points.obj'))
                    write_oriented_bbox(pred_data_3d['bboxes_3d'],
                                        osp.join(out_file, 'pred_bbox.obj'))
            if gt_seg_data_3d is not None:
                write_obj(gt_seg_data_3d['points'],
                          osp.join(out_file, 'points.obj'))
                write_obj(gt_seg_data_3d['seg_color'],
                          osp.join(out_file, 'gt_seg.obj'))
            if pred_seg_data_3d is not None:
                write_obj(pred_seg_data_3d['points'],
                          osp.join(out_file, 'points.obj'))
                write_obj(pred_seg_data_3d['seg_color'],
                          osp.join(out_file, 'pred_seg.obj'))
        else:
            self.add_image(name, drawn_img_3d, step)
