# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np

from mmdet3d.core.bbox import points_cam2img
from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
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
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class LoadImageFromFileMono3D(LoadImageFromFile):
    """Load an image from file in monocular 3D object detection. Compared to 2D
    detection, additional camera parameters need to be loaded.

    Args:
        kwargs (dict): Arguments are the same as those in
            :class:`LoadImageFromFile`.
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        super().__call__(results)
        results['cam2img'] = results['img_info']['cam_intrinsic']
        return results


@PIPELINES.register_module()
class LoadPointsFromMultiSweeps(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int, optional): Number of sweeps. Defaults to 10.
        load_dim (int, optional): Dimension number of the loaded points.
            Defaults to 5.
        use_dim (list[int], optional): Which dimension to use.
            Defaults to [0, 1, 2, 4].
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool, optional): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool, optional): Whether to remove close points.
            Defaults to False.
        test_mode (bool, optional): If `test_mode=True`, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 test_mode=False):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float, optional): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
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

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data.
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point
                    cloud arrays.
        """
        points = results['points']
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@PIPELINES.register_module()
class PointSegClassMapping(object):
    """Map original semantic class to valid category ids.

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).

    Args:
        valid_cat_ids (tuple[int]): A tuple of valid category.
        max_cat_id (int, optional): The max possible cat_id in input
            segmentation mask. Defaults to 40.
    """

    def __init__(self, valid_cat_ids, max_cat_id=40):
        assert max_cat_id >= np.max(valid_cat_ids), \
            'max_cat_id should be greater than maximum id in valid_cat_ids'

        self.valid_cat_ids = valid_cat_ids
        self.max_cat_id = int(max_cat_id)

        # build cat_id to class index mapping
        neg_cls = len(valid_cat_ids)
        self.cat_id2class = np.ones(
            self.max_cat_id + 1, dtype=np.int) * neg_cls
        for cls_idx, cat_id in enumerate(valid_cat_ids):
            self.cat_id2class[cat_id] = cls_idx

    def __call__(self, results):
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

        converted_pts_sem_mask = self.cat_id2class[pts_semantic_mask]

        results['pts_semantic_mask'] = converted_pts_sem_mask
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(valid_cat_ids={self.valid_cat_ids}, '
        repr_str += f'max_cat_id={self.max_cat_id})'
        return repr_str


@PIPELINES.register_module()
class NormalizePointsColor(object):
    """Normalize color of points.

    Args:
        color_mean (list[float]): Mean color of the point cloud.
    """

    def __init__(self, color_mean):
        self.color_mean = color_mean

    def __call__(self, results):
        """Call function to normalize color of points.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the normalized points.
                Updated key and value are described below.

                - points (:obj:`BasePoints`): Points after color normalization.
        """
        points = results['points']
        assert points.attribute_dims is not None and \
            'color' in points.attribute_dims.keys(), \
            'Expect points have color attribute'
        if self.color_mean is not None:
            points.color = points.color - \
                points.color.new_tensor(self.color_mean)
        points.color = points.color / 255.0
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(color_mean={self.color_mean})'
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromFile(object):
    """Load Points From File.

    Load points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int, optional): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int], optional): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool, optional): Whether to use shifted height.
            Defaults to False.
        use_color (bool, optional): Whether to use color features.
            Defaults to False.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk')):
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

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
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

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromDict(LoadPointsFromFile):
    """Load Points From Dict."""

    def __call__(self, results):
        assert 'points' in results
        return results


@PIPELINES.register_module()
class LoadAnnotations3D(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """

    def __init__(self,
                 with_bbox_3d=True,
                 with_label_3d=True,
                 with_attr_label=False,
                 with_mask_3d=False,
                 with_seg_3d=False,
                 with_bbox=False,
                 with_label=False,
                 with_mask=False,
                 with_seg=False,
                 with_bbox_depth=False,
                 poly2mask=True,
                 seg_3d_dtype='int',
                 file_client_args=dict(backend='disk')):
        super().__init__(
            with_bbox,
            with_label,
            with_mask,
            with_seg,
            poly2mask,
            file_client_args=file_client_args)
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.seg_3d_dtype = seg_3d_dtype

    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        results['gt_bboxes_3d'] = results['ann_info']['gt_bboxes_3d']
        results['bbox3d_fields'].append('gt_bboxes_3d')
        return results

    def _load_bboxes_depth(self, results):
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """
        results['centers2d'] = results['ann_info']['centers2d']
        results['depths'] = results['ann_info']['depths']
        return results

    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']
        return results

    def _load_attr_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['attr_labels'] = results['ann_info']['attr_labels']
        return results

    def _load_masks_3d(self, results):
        """Private function to load 3D mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        pts_instance_mask_path = results['ann_info']['pts_instance_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_instance_mask_path)
            pts_instance_mask = np.frombuffer(mask_bytes, dtype=np.int)
        except ConnectionError:
            mmcv.check_file_exist(pts_instance_mask_path)
            pts_instance_mask = np.fromfile(
                pts_instance_mask_path, dtype=np.long)

        results['pts_instance_mask'] = pts_instance_mask
        results['pts_mask_fields'].append('pts_instance_mask')
        return results

    def _load_semantic_seg_3d(self, results):
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        pts_semantic_mask_path = results['ann_info']['pts_semantic_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_semantic_mask_path)
            # add .copy() to fix read-only bug
            pts_semantic_mask = np.frombuffer(
                mask_bytes, dtype=self.seg_3d_dtype).copy()
        except ConnectionError:
            mmcv.check_file_exist(pts_semantic_mask_path)
            pts_semantic_mask = np.fromfile(
                pts_semantic_mask_path, dtype=np.long)

        results['pts_semantic_mask'] = pts_semantic_mask
        results['pts_seg_fields'].append('pts_semantic_mask')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
            if results is None:
                return None
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
            if results is None:
                return None
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)
        if self.with_mask_3d:
            results = self._load_masks_3d(results)
        if self.with_seg_3d:
            results = self._load_semantic_seg_3d(results)

        return results

    def __repr__(self):
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


@PIPELINES.register_module()
class GenerateEgdeIndices(object):
    """Generate edge indices for monocular 3d object deteciton.

    Args:
        pad_mode (str): the pad mode of image.
            Default: default
    """

    def __init__(self, pad_mode='default'):
        self.pad_mode = pad_mode

    def __call__(self, results):
        """Call function to generate edge indices for image.
        Args:
            results (dict): Result dict containing data.

        Returns:
            dict: The result dict containing the edge indices
                and edge length. Updated key and value are
                described below.

                - edge_indices (np.ndarray): edge indices of image boundary
                    after padding.
                - edge_len (np.ndarray): the length of edge indices.
        """
        # Now only support the padding mode:
        # padding = (0, 0, shape[1] - img.shape[1], shape[0] - img.shape[0])
        img_h, img_w = results['ori_shape']
        if self.pad_mode == 'default':
            x_min = 0
            y_min = 0
            x_max, y_max = img_w - 1, img_h - 1
        step = 1

        # boundary idxs
        edge_indices = []

        # left
        y = np.arange(y_min, y_max, step)
        x = np.ones(len(y)) * x_min

        edge_indices_edge = np.stack((x, y), axis=1)
        edge_indices_edge[:, 0] = np.clip(edge_indices_edge[:, 0], x_min,
                                          x_max)
        edge_indices_edge[:, 1] = np.clip(edge_indices_edge[:, 1], y_min,
                                          y_max)
        edge_indices.append(edge_indices_edge)

        # bottom
        x = np.arange(x_min, x_max, step)
        y = np.ones(len(x)) * y_max

        edge_indices_edge = np.stack((x, y), axis=1)
        edge_indices_edge[:, 0] = np.clip(edge_indices_edge[:, 0], x_min,
                                          x_max)
        edge_indices_edge[:, 1] = np.clip(edge_indices_edge[:, 1], y_min,
                                          y_max)
        edge_indices.append(edge_indices_edge)

        # right
        y = np.arange(y_max, y_min, -step)
        x = np.ones(len(y)) * x_max

        edge_indices_edge = np.stack((x, y), axis=1)
        edge_indices_edge[:, 0] = np.clip(edge_indices_edge[:, 0], x_min,
                                          x_max)
        edge_indices_edge[:, 1] = np.clip(edge_indices_edge[:, 1], y_min,
                                          y_max)
        edge_indices.append(edge_indices_edge)

        # top
        x = np.arange(x_max, x_min - 1, -step)
        y = np.ones(len(x)) * y_min

        edge_indices_edge = np.stack((x, y), axis=1)
        edge_indices_edge[:, 0] = np.clip(edge_indices_edge[:, 0], x_min,
                                          x_max)
        edge_indices_edge[:, 1] = np.clip(edge_indices_edge[:, 1], y_min,
                                          y_max)
        edge_indices.append(edge_indices_edge)

        edge_indices = np.concatenate([index for index in edge_indices],
                                      axis=0)
        results['edge_indices'] = edge_indices
        # length is different to different images
        results['edge_len'] = edge_indices.shape[0] - 1

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(pad_mode={self.pad_mode})'
        return repr_str


@PIPELINES.register_module()
class GenerateKeypoints(object):
    """Generate keypoints for monocular 3d object deteciton.

    Args:
        use_local_coords (list[float]): Whether to use local coordinate of
            keypooints. Default: True.
    """

    def __init__(self, use_local_coords=True):
        self.use_local_coords = use_local_coords

    def __call__(self, results):
        """Call function to generate keypoints and its visible mask.
        Args:
            results (dict): Result dict containing data.

        Returns:
            dict: The result dict containing the generated keypoints and
                its mask. Updated key and value are described below.

                - keypoints2d (np.ndarray): generated keypoints with
                    visible mask.
                - keypoints_depth_mask (np.ndarray): mask of depth
                    constructed by keypoints.
        """
        img_h, img_w = results['ori_shape']
        gt_bboxes_3d = results['gt_bboxes_3d']
        # shape (N, 8, 3)
        corners3d = np.array(gt_bboxes_3d.corners)
        top_centers3d = corners3d[:, [0, 1, 4, 5], :].mean(axis=1)
        bot_centers3d = corners3d[:, [2, 3, 6, 7], :].mean(axis=1)
        # (N, 2, 3)
        top_bot_centers3d = np.stack((top_centers3d, bot_centers3d), axis=1)
        keypoints3d = np.concatenate((corners3d, top_bot_centers3d),
                                     axis=1)  # (N, 10, 3)
        # (N, 10, 2)
        keypoints2d = points_cam2img(keypoints3d, results['cam_intrinsic'])

        # keypoints mask: keypoints must be inside
        # the image and in front of the camera
        keypoints_x_visible = (keypoints2d[..., 0] >= 0) & (
            keypoints2d[..., 0] <= img_w - 1)
        keypoints_y_visible = (keypoints2d[..., 1] >= 0) & (
            keypoints2d[..., 1] <= img_h - 1)
        keypoints_z_visible = (keypoints3d[..., -1] > 0)

        # xyz visible
        # (N, 1O)
        keypoints_visible = keypoints_x_visible & \
            keypoints_y_visible & keypoints_z_visible
        # center, diag-02, diag-13
        # (N, 3)
        keypoints_depth_valid = np.stack(
            (keypoints_visible[:, [8, 9]].all(axis=1),
             keypoints_visible[:, [0, 3, 5, 6]].all(axis=1),
             keypoints_visible[:, [1, 2, 4, 7]].all(axis=1)),
            axis=1)

        if not self.use_local_coords:
            keypoints2d = np.concatenate(
                (keypoints2d, keypoints_visible[:, np.newaxis]), axis=1)
        else:
            if 'target_centers2d' in results:
                keypoints2d = np.concatenate(
                    (keypoints2d - results['target_centers2d'].reshape(1, -1),
                     keypoints_visible[:, np.newaxis]),
                    axis=1)
            else:
                keypoints2d = np.concatenate(
                    (keypoints2d - results['centers2d'].reshape(1, -1),
                     keypoints_visible[:, np.newaxis]),
                    axis=1)

        results['keypoints2d'] = keypoints2d
        results['keypoints_depth_mask'] = keypoints_depth_valid

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(use_local_coords={self.use_local_coords})'
        return repr_str


@PIPELINES.register_module()
class TruncationHandle(object):
    """Pipeline to handle truncated objects.

    First, it judges whether the projected 3d center is inside
    the image or not. If it's outside the image, it will choose
    to discard or retain it.
    Once the outside center is saved, it cannot be directly used
    in gt heatmap generation, the outside center needs to be
    clamped to the image, so two mode:
    1. instersecttion 2. vertical_line (not implement)
    By the transition, the outside center can be clamped into image
    boundary. However, the clamped center can not be promised inside
    the corresponding 2d bbox (on the boundary of 2d bbox).

    Args:
        keep_outside_objs (bool): Whether to keep the objects
            outside the image.
        proj_center_mode (str, optional): the mode of dealing with
            label of the outside object's projected center.
            Default: 'intersection'.
    """

    def __init__(self, keep_outside_objs, proj_center_mode='intersection'):
        self.keep_outside_objs = keep_outside_objs
        self.proj_center_mode = proj_center_mode

    def __call__(self, results):
        """Call function to handle truncation, generate target centers2d and
        offset.

        Args:
            results (dict): Result dict containing data.

        Returns:
            dict: The result dict containing target centers2d and
                offset to centers2d, Updated key and value are
                described below.

                - target_centers2d (np.ndarray): Target centers2d
                    for objects outside image.
                - offset2d (np.ndarray): Offsets between target
                    centers2d and real centers2d.
        """

        centers2d = results['centers2d']  # (N, 2)
        target_centers2d = centers2d.copy()
        inside_index = (centers2d[:, 0] > 0) & \
            (centers2d[:, 0] < self.img_scale[0]) & \
            (centers2d[:, 1] > 0) & \
            (centers2d[:, 1] < self.img_scale[1])

        if self.keep_outside_objs is True:
            outside_index = not inside_index
            gt_bboxes = results['gt_bboxes']  # (N, 4)
            centers = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2  # (N, 2)
            # The num of projected center2d should be same as
            # the number of 2d gt bboxes
            assert len(centers2d) == len(centers)
            outside_centers2d = centers2d[outside_index]
            match_centers = centers[outside_index]

            if self.proj_center_mode == 'intersection':
                target_outside_centers2d = self._approx_centers2d(
                    outside_centers2d, match_centers, results['ori_shape'])
            else:
                raise NotImplementedError

            target_centers2d[outside_index] = target_outside_centers2d
            # translate target_centers2d from float to int,
            # used for generating heatmap.
            target_centers2d = target_centers2d.round().astype(np.int)
            # (n, 2) np.float
            offsets2d = centers2d - target_centers2d
            results['target_centers2d'] = target_centers2d
            results['offsets2d'] = offsets2d

        else:
            # keep only the center2d inside objects
            results['centers2d'] = centers2d[inside_index]
            for key in results.get('bbox_fields', []):
                if key in ['gt_bboxes']:
                    results[key] = results[key][inside_index]
                    if 'gt_labels' in results:
                        results['gt_labels'] = results['gt_labels'][
                            inside_index]
                    if 'gt_masks' in results:
                        raise NotImplementedError(
                            'Truncation only supports bbox.')

            for key in results.get('bbox3d_fields', []):
                if key in ['gt_bboxes_3d']:
                    results[key].tensor = results[key].tensor[inside_index]
                    if 'gt_labels_3d' in results:
                        results['gt_labels_3d'] = results['gt_labels_3d'][
                            inside_index]

        return results

    def _approx_centers2d(centers2d, centers, img_scale):
        """
        Args:
            centers2d (np.ndarray): Projected 3D centers onto 2D images.
            centers (np.ndarray): Centers of 2d gt bboxes.
            img_scale (tuple): Image original shape.

        Returns:
            np.ndarray: Target centers2d for real centers2d.

        """
        img_h, img_w = img_scale[0], img_scale[1]

        target_center2d_list = []

        for i in range(centers2d.shape[0]):
            # y = ax + b
            # get a line
            center2d = centers2d[i]
            center = centers[i]
            a, b = np.polyfit([center2d[0], center[0]],
                              [center2d[1], center[1]], 1)
            valid_intersects = []
            # valid_edge = []

            left_y = b
            if (0 <= left_y <= img_h - 1):
                valid_intersects.append(np.array([0, left_y]))
                # valid_edge.append(0)

            right_y = (img_w - 1) * a + b
            if (0 <= right_y <= img_h - 1):
                valid_intersects.append(np.array([img_w - 1, right_y]))
                # valid_edge.append(1)

            top_x = -b / a
            if (0 <= top_x <= img_w - 1):
                valid_intersects.append(np.array([top_x, 0]))
                # valid_edge.append(2)

            bottom_x = (img_h - 1 - b) / a
            if (0 <= bottom_x <= img_w - 1):
                valid_intersects.append(np.array([bottom_x, img_h - 1]))
                # valid_edge.append(3)

            valid_intersects = np.stack(valid_intersects)
            # 找到距离proj center 最近的交点
            min_idx = np.argmin(
                np.linalg.norm(
                    valid_intersects - center2d.reshape(1, 2), axis=1))
            target_center2d_list.append(valid_intersects[min_idx])

        target_centers2d = np.stack(target_center2d_list)  # (n, 2)

        return target_centers2d

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(keep_outside_objs={self.keep_outside_objs})'
        repr_str += f'(proj_center_mode={self.proj_center_mode})'
        return repr_str
