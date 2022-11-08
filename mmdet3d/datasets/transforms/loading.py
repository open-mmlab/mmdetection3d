# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional, Union

import mmcv
import mmengine
import numpy as np
from mmcv.transforms import LoadImageFromFile
from mmcv.transforms.base import BaseTransform
from mmdet.datasets.transforms import LoadAnnotations

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.points import BasePoints, get_points_type


@TRANSFORMS.register_module()
class LoadMultiViewImageFromFiles(BaseTransform):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmengine.fileio.FileClient` for details.
            Defaults to dict(backend='disk').
        num_views (int): Number of view in a frame. Defaults to 5.
        num_ref_frames (int): Number of frame in loading. Defaults to -1.
        test_mode (bool): Whether is test mode in loading. Defaults to False.
        set_default_scale (bool): Whether to set default scale.
            Defaults to True.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'unchanged',
                 file_client_args: dict = dict(backend='disk'),
                 num_views: int = 5,
                 num_ref_frames: int = -1,
                 test_mode: bool = False,
                 set_default_scale: bool = True) -> None:
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.num_views = num_views
        # num_ref_frames is used for multi-sweep loading
        self.num_ref_frames = num_ref_frames
        # when test_mode=False, we randomly select previous frames
        # otherwise, select the earliest one
        self.test_mode = test_mode
        self.set_default_scale = set_default_scale

    def transform(self, results: dict) -> Optional[dict]:
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
            Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        # TODO: consider split the multi-sweep part out of this pipeline
        # Derive the mask and transform for loading of multi-sweep data
        if self.num_ref_frames > 0:
            # init choice with the current frame
            init_choice = np.array([0], dtype=np.int64)
            num_frames = len(results['img_filename']) // self.num_views - 1
            if num_frames == 0:  # no previous frame, then copy cur frames
                choices = np.random.choice(
                    1, self.num_ref_frames, replace=True)
            elif num_frames >= self.num_ref_frames:
                # NOTE: suppose the info is saved following the order
                # from latest to earlier frames
                if self.test_mode:
                    choices = np.arange(num_frames - self.num_ref_frames,
                                        num_frames) + 1
                # NOTE: +1 is for selecting previous frames
                else:
                    choices = np.random.choice(
                        num_frames, self.num_ref_frames, replace=False) + 1
            elif num_frames > 0 and num_frames < self.num_ref_frames:
                if self.test_mode:
                    base_choices = np.arange(num_frames) + 1
                    random_choices = np.random.choice(
                        num_frames,
                        self.num_ref_frames - num_frames,
                        replace=True) + 1
                    choices = np.concatenate([base_choices, random_choices])
                else:
                    choices = np.random.choice(
                        num_frames, self.num_ref_frames, replace=True) + 1
            else:
                raise NotImplementedError
            choices = np.concatenate([init_choice, choices])
            select_filename = []
            for choice in choices:
                select_filename += results['img_filename'][choice *
                                                           self.num_views:
                                                           (choice + 1) *
                                                           self.num_views]
            results['img_filename'] = select_filename
            for key in ['cam2img', 'lidar2cam']:
                if key in results:
                    select_results = []
                    for choice in choices:
                        select_results += results[key][choice *
                                                       self.num_views:(choice +
                                                                       1) *
                                                       self.num_views]
                    results[key] = select_results
            for key in ['ego2global']:
                if key in results:
                    select_results = []
                    for choice in choices:
                        select_results += [results[key][choice]]
                    results[key] = select_results
            # Transform lidar2cam to
            # [cur_lidar]2[prev_img] and [cur_lidar]2[prev_cam]
            for key in ['lidar2cam']:
                if key in results:
                    # only change matrices of previous frames
                    for choice_idx in range(1, len(choices)):
                        pad_prev_ego2global = np.eye(4)
                        prev_ego2global = results['ego2global'][choice_idx]
                        pad_prev_ego2global[:prev_ego2global.
                                            shape[0], :prev_ego2global.
                                            shape[1]] = prev_ego2global
                        pad_cur_ego2global = np.eye(4)
                        cur_ego2global = results['ego2global'][0]
                        pad_cur_ego2global[:cur_ego2global.
                                           shape[0], :cur_ego2global.
                                           shape[1]] = cur_ego2global
                        cur2prev = np.linalg.inv(pad_prev_ego2global).dot(
                            pad_cur_ego2global)
                        for result_idx in range(choice_idx * self.num_views,
                                                (choice_idx + 1) *
                                                self.num_views):
                            results[key][result_idx] = \
                                results[key][result_idx].dot(cur2prev)
        # Support multi-view images with different shapes
        # TODO: record the origin shape and padded shape
        filename, cam2img, lidar2cam = [], [], []
        for _, cam_item in results['images'].items():
            filename.append(cam_item['img_path'])
            cam2img.append(cam_item['cam2img'])
            lidar2cam.append(cam_item['lidar2cam'])
        results['filename'] = filename
        results['cam2img'] = cam2img
        results['lidar2cam'] = lidar2cam

        results['ori_cam2img'] = copy.deepcopy(results['cam2img'])

        if self.file_client is None:
            self.file_client = mmengine.FileClient(**self.file_client_args)

        # img is of shape (h, w, c, num_views)
        # h and w can be different for different views
        img_bytes = [self.file_client.get(name) for name in filename]
        imgs = [
            mmcv.imfrombytes(img_byte, flag=self.color_type)
            for img_byte in img_bytes
        ]
        # handle the image with different shape
        img_shapes = np.stack([img.shape for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        assert img_shape_min[-1] == img_shape_max[-1]
        if not np.all(img_shape_max == img_shape_min):
            pad_shape = img_shape_max[:2]
        else:
            pad_shape = None
        if pad_shape is not None:
            imgs = [
                mmcv.impad(img, shape=pad_shape, pad_val=0) for img in imgs
            ]
        img = np.stack(imgs, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        if self.set_default_scale:
            results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['num_views'] = self.num_views
        results['num_ref_frames'] = self.num_ref_frames
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}', "
        repr_str += f'num_views={self.num_views}, '
        repr_str += f'num_ref_frames={self.num_ref_frames}, '
        repr_str += f'test_mode={self.test_mode})'
        return repr_str


@TRANSFORMS.register_module()
class LoadImageFromFileMono3D(LoadImageFromFile):
    """Load an image from file in monocular 3D object detection. Compared to 2D
    detection, additional camera parameters need to be loaded.

    Args:
        kwargs (dict): Arguments are the same as those in
            :class:`LoadImageFromFile`.
    """

    def transform(self, results: dict) -> dict:
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        # TODO: load different camera image from data info,
        # for kitti dataset, we load 'CAM2' image.
        # for nuscenes dataset, we load 'CAM_FRONT' image.

        if 'CAM2' in results['images']:
            filename = results['images']['CAM2']['img_path']
            results['cam2img'] = results['images']['CAM2']['cam2img']
        elif len(list(results['images'].keys())) == 1:
            camera_type = list(results['images'].keys())[0]
            filename = results['images'][camera_type]['img_path']
            results['cam2img'] = results['images'][camera_type]['cam2img']
        else:
            raise NotImplementedError(
                'Currently we only support load image from kitti and'
                'nuscenes datasets')

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]

        return results


@TRANSFORMS.register_module()
class LoadPointsFromMultiSweeps(BaseTransform):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmengine.fileio.FileClient` for details.
            Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points. Defaults to False.
        test_mode (bool): If `test_mode=True`, it will not randomly sample
            sweeps but select the nearest N frames. Defaults to False.
    """

    def __init__(self,
                 sweeps_num: int = 10,
                 load_dim: int = 5,
                 use_dim: List[int] = [0, 1, 2, 4],
                 file_client_args: dict = dict(backend='disk'),
                 pad_empty_sweeps: bool = False,
                 remove_close: bool = False,
                 test_mode: bool = False) -> None:
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode

    def _load_points(self, pts_filename: str) -> np.ndarray:
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmengine.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmengine.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self,
                      points: Union[np.ndarray, BasePoints],
                      radius: float = 1.0) -> Union[np.ndarray, BasePoints]:
        """Remove point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray | :obj:`BasePoints`: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def transform(self, results: dict) -> dict:
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data.
            Updated key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point
                  cloud arrays.
        """
        points = results['points']
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        if 'lidar_sweeps' not in results:
            if self.pad_empty_sweeps:
                for i in range(self.sweeps_num):
                    if self.remove_close:
                        sweep_points_list.append(self._remove_close(points))
                    else:
                        sweep_points_list.append(points)
        else:
            if len(results['lidar_sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['lidar_sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['lidar_sweeps']),
                    self.sweeps_num,
                    replace=False)
            for idx in choices:
                sweep = results['lidar_sweeps'][idx]
                points_sweep = self._load_points(
                    sweep['lidar_points']['lidar_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                # bc-breaking: Timestamp has divided 1e6 in pkl infos.
                sweep_ts = sweep['timestamp']
                lidar2sensor = np.array(sweep['lidar_points']['lidar2sensor'])
                points_sweep[:, :
                             3] = points_sweep[:, :3] @ lidar2sensor[:3, :3]
                points_sweep[:, :3] -= lidar2sensor[:3, 3]
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results['points'] = points
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@TRANSFORMS.register_module()
class PointSegClassMapping(BaseTransform):
    """Map original semantic class to valid category ids.

    Required Keys:

    - seg_label_mapping (np.ndarray)
    - pts_semantic_mask (np.ndarray)

    Added Keys:

    - points (np.float32)

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).
    """

    def transform(self, results: dict) -> dict:
        """Call function to map original semantic class to valid category ids.

        Args:
            results (dict): Result dict containing point semantic masks.

        Returns:
            dict: The result dict containing the mapped category ids.
            Updated key and value are described below.

                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        assert 'pts_semantic_mask' in results
        pts_semantic_mask = results['pts_semantic_mask']

        assert 'seg_label_mapping' in results
        label_mapping = results['seg_label_mapping']
        converted_pts_sem_mask = label_mapping[pts_semantic_mask]

        results['pts_semantic_mask'] = converted_pts_sem_mask

        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in results:
            assert 'pts_semantic_mask' in results['eval_ann_info']
            results['eval_ann_info']['pts_semantic_mask'] = \
                converted_pts_sem_mask

        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class NormalizePointsColor(BaseTransform):
    """Normalize color of points.

    Args:
        color_mean (list[float]): Mean color of the point cloud.
    """

    def __init__(self, color_mean: List[float]) -> None:
        self.color_mean = color_mean

    def transform(self, input_dict: dict) -> dict:
        """Call function to normalize color of points.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the normalized points.
            Updated key and value are described below.

                - points (:obj:`BasePoints`): Points after color normalization.
        """
        points = input_dict['points']
        assert points.attribute_dims is not None and \
               'color' in points.attribute_dims.keys(), \
               'Expect points have color attribute'
        if self.color_mean is not None:
            points.color = points.color - \
                           points.color.new_tensor(self.color_mean)
        points.color = points.color / 255.0
        input_dict['points'] = points
        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(color_mean={self.color_mean})'
        return repr_str


@TRANSFORMS.register_module()
class LoadPointsFromFile(BaseTransform):
    """Load Points From File.

    Required Keys:

    - lidar_points (dict)

        - lidar_path (str)

    Added Keys:

    - points (np.float32)

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:

            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points. Defaults to 6.
        use_dim (list[int] | int): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmengine.fileio.FileClient` for details.
            Defaults to dict(backend='disk').
    """

    def __init__(
        self,
        coord_type: str,
        load_dim: int = 6,
        use_dim: Union[int, List[int]] = [0, 1, 2],
        shift_height: bool = False,
        use_color: bool = False,
        file_client_args: dict = dict(backend='disk')
    ) -> None:
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filename: str) -> np.ndarray:
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmengine.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmengine.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def transform(self, results: dict) -> dict:
        """Method to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
            Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_file_path = results['lidar_points']['lidar_path']
        points = self._load_points(pts_file_path)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str


@TRANSFORMS.register_module()
class LoadPointsFromDict(LoadPointsFromFile):
    """Load Points From Dict."""

    def transform(self, results: dict) -> dict:
        assert 'points' in results
        return results


@TRANSFORMS.register_module()
class LoadAnnotations3D(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Required Keys:

    - ann_info (dict)

        - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes` |
          :obj:`DepthInstance3DBoxes` | :obj:`CameraInstance3DBoxes`):
          3D ground truth bboxes. Only when `with_bbox_3d` is True
        - gt_labels_3d (np.int64): Labels of ground truths.
          Only when `with_label_3d` is True.
        - gt_bboxes (np.float32): 2D ground truth bboxes.
          Only when `with_bbox` is True.
        - gt_labels (np.ndarray): Labels of ground truths.
          Only when `with_label` is True.
        - depths (np.ndarray): Only when
          `with_bbox_depth` is True.
        - centers_2d (np.ndarray): Only when
          `with_bbox_depth` is True.
        - attr_labels (np.ndarray): Attribute labels of instances.
          Only when `with_attr_label` is True.

    - pts_instance_mask_path (str): Path of instance mask file.
      Only when `with_mask_3d` is True.
    - pts_semantic_mask_path (str): Path of semantic mask file.
      Only when `with_seg_3d` is True.

    Added Keys:

    - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes` |
      :obj:`DepthInstance3DBoxes` | :obj:`CameraInstance3DBoxes`):
      3D ground truth bboxes. Only when `with_bbox_3d` is True
    - gt_labels_3d (np.int64): Labels of ground truths.
      Only when `with_label_3d` is True.
    - gt_bboxes (np.float32): 2D ground truth bboxes.
      Only when `with_bbox` is True.
    - gt_labels (np.int64): Labels of ground truths.
      Only when `with_label` is True.
    - depths (np.float32): Only when
      `with_bbox_depth` is True.
    - centers_2d (np.ndarray): Only when
      `with_bbox_depth` is True.
    - attr_labels (np.int64): Attribute labels of instances.
      Only when `with_attr_label` is True.
    - pts_instance_mask (np.int64): Instance mask of each point.
      Only when `with_mask_3d` is True.
    - pts_semantic_mask (np.int64): Semantic mask of each point.
      Only when `with_seg_3d` is True.

    Args:
        with_bbox_3d (bool): Whether to load 3D boxes. Defaults to True.
        with_label_3d (bool): Whether to load 3D labels. Defaults to True.
        with_attr_label (bool): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool): Whether to load 3D instance masks for points.
            Defaults to False.
        with_seg_3d (bool): Whether to load 3D semantic masks for points.
            Defaults to False.
        with_bbox (bool): Whether to load 2D boxes. Defaults to False.
        with_label (bool): Whether to load 2D labels. Defaults to False.
        with_mask (bool): Whether to load 2D instance masks. Defaults to False.
        with_seg (bool): Whether to load 2D semantic masks. Defaults to False.
        with_bbox_depth (bool): Whether to load 2.5D boxes. Defaults to False.
        poly2mask (bool): Whether to convert polygon annotations to bitmasks.
            Defaults to True.
        seg_3d_dtype (dtype): Dtype of 3D semantic masks. Defaults to int64.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmengine.fileio.FileClient` for details.
            Defaults to dict(backend='disk').
    """

    def __init__(
        self,
        with_bbox_3d: bool = True,
        with_label_3d: bool = True,
        with_attr_label: bool = False,
        with_mask_3d: bool = False,
        with_seg_3d: bool = False,
        with_bbox: bool = False,
        with_label: bool = False,
        with_mask: bool = False,
        with_seg: bool = False,
        with_bbox_depth: bool = False,
        poly2mask: bool = True,
        seg_3d_dtype: np.dtype = np.int64,
        file_client_args: dict = dict(backend='disk')
    ) -> None:
        super().__init__(
            with_bbox=with_bbox,
            with_label=with_label,
            with_mask=with_mask,
            with_seg=with_seg,
            poly2mask=poly2mask,
            file_client_args=file_client_args)
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.seg_3d_dtype = seg_3d_dtype

    def _load_bboxes_3d(self, results: dict) -> dict:
        """Private function to move the 3D bounding box annotation from
        `ann_info` field to the root of `results`.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """

        results['gt_bboxes_3d'] = results['ann_info']['gt_bboxes_3d']
        return results

    def _load_bboxes_depth(self, results: dict) -> dict:
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """

        results['depths'] = results['ann_info']['depths']
        results['centers_2d'] = results['ann_info']['centers_2d']
        return results

    def _load_labels_3d(self, results: dict) -> dict:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """

        results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']
        return results

    def _load_attr_labels(self, results: dict) -> dict:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['attr_labels'] = results['ann_info']['attr_labels']
        return results

    def _load_masks_3d(self, results: dict) -> dict:
        """Private function to load 3D mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        pts_instance_mask_path = results['pts_instance_mask_path']

        if self.file_client is None:
            self.file_client = mmengine.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_instance_mask_path)
            pts_instance_mask = np.frombuffer(mask_bytes, dtype=np.int64)
        except ConnectionError:
            mmengine.check_file_exist(pts_instance_mask_path)
            pts_instance_mask = np.fromfile(
                pts_instance_mask_path, dtype=np.int64)

        results['pts_instance_mask'] = pts_instance_mask
        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in results:
            results['eval_ann_info']['pts_instance_mask'] = pts_instance_mask
        return results

    def _load_semantic_seg_3d(self, results: dict) -> dict:
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        pts_semantic_mask_path = results['pts_semantic_mask_path']

        if self.file_client is None:
            self.file_client = mmengine.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_semantic_mask_path)
            # add .copy() to fix read-only bug
            pts_semantic_mask = np.frombuffer(
                mask_bytes, dtype=self.seg_3d_dtype).copy()
        except ConnectionError:
            mmengine.check_file_exist(pts_semantic_mask_path)
            pts_semantic_mask = np.fromfile(
                pts_semantic_mask_path, dtype=np.int64)

        results['pts_semantic_mask'] = pts_semantic_mask
        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in results:
            results['eval_ann_info']['pts_semantic_mask'] = pts_semantic_mask
        return results

    def _load_bboxes(self, results: dict) -> None:
        """Private function to load bounding box annotations.

        The only difference is it remove the proceess for
        `ignore_flag`

        Args:
            results (dict): Result dict from :obj:`mmcv.BaseDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """

        results['gt_bboxes'] = results['ann_info']['gt_bboxes']

    def _load_labels(self, results: dict) -> None:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj :obj:`mmcv.BaseDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """
        results['gt_bboxes_labels'] = results['ann_info']['gt_bboxes_labels']

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
            semantic segmentation annotations.
        """
        results = super().transform(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)
        if self.with_mask_3d:
            results = self._load_masks_3d(results)
        if self.with_seg_3d:
            results = self._load_semantic_seg_3d(results)

        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_bbox_3d={self.with_bbox_3d}, '
        repr_str += f'{indent_str}with_label_3d={self.with_label_3d}, '
        repr_str += f'{indent_str}with_attr_label={self.with_attr_label}, '
        repr_str += f'{indent_str}with_mask_3d={self.with_mask_3d}, '
        repr_str += f'{indent_str}with_seg_3d={self.with_seg_3d}, '
        repr_str += f'{indent_str}with_bbox={self.with_bbox}, '
        repr_str += f'{indent_str}with_label={self.with_label}, '
        repr_str += f'{indent_str}with_mask={self.with_mask}, '
        repr_str += f'{indent_str}with_seg={self.with_seg}, '
        repr_str += f'{indent_str}with_bbox_depth={self.with_bbox_depth}, '
        repr_str += f'{indent_str}poly2mask={self.poly2mask})'
        return repr_str
