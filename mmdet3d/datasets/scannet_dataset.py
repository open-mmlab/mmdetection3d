import mmcv
import numpy as np
import torch
from os import path as osp

from mmdet3d.core import show_result, show_seg_result
from mmdet3d.core.bbox import DepthInstance3DBoxes
from mmdet.datasets import DATASETS
from ..core.bbox import get_box_type
from .custom_3d import Custom3DDataset
from .pipelines import Compose


@DATASETS.register_module()
class ScanNetDataset(Custom3DDataset):
    r"""ScanNet Dataset for Detection Task.

    This class serves as the API for experiments on the ScanNet Dataset.

    Please refer to the `github repo <https://github.com/ScanNet/ScanNet>`_
    for data downloading.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'Depth' in this dataset. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """
    CLASSES = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
               'garbagebin')

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='Depth',
                 filter_empty_gt=True,
                 test_mode=False):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`DepthInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - pts_instance_mask_path (str): Path of instance masks.
                - pts_semantic_mask_path (str): Path of semantic masks.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]
        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(
                np.float32)  # k, 6
            gt_labels_3d = info['annos']['class'].astype(np.long)
        else:
            gt_bboxes_3d = np.zeros((0, 6), dtype=np.float32)
            gt_labels_3d = np.zeros((0, ), dtype=np.long)

        # to target box structure
        gt_bboxes_3d = DepthInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            with_yaw=False,
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        pts_instance_mask_path = osp.join(self.data_root,
                                          info['pts_instance_mask_path'])
        pts_semantic_mask_path = osp.join(self.data_root,
                                          info['pts_semantic_mask_path'])

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            pts_instance_mask_path=pts_instance_mask_path,
            pts_semantic_mask_path=pts_semantic_mask_path)
        return anns_results

    def show(self, results, out_dir, show=True):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        for i, result in enumerate(results):
            data_info = self.data_infos[i]
            pts_path = data_info['pts_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = np.fromfile(
                osp.join(self.data_root, pts_path),
                dtype=np.float32).reshape(-1, 6)
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor
            pred_bboxes = result['boxes_3d'].tensor.numpy()
            show_result(points, gt_bboxes, pred_bboxes, out_dir, file_name,
                        show)


@DATASETS.register_module()
class ScanNetSegDataset(Custom3DDataset):
    r"""ScanNet Dataset for Semantic Segmentation Task.

    This class serves as the API for experiments on the ScanNet Dataset.

    Please refer to the `github repo <https://github.com/ScanNet/ScanNet>`_
    for data downloading.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        palette (list[list[int]], optional): The palette of segmentation map.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        ignore_index (int, optional): The label index to be ignored, e.g. \
            unannotated points. If None is given, set to len(self.CLASSES).
            Defaults to None.
        num_points (int, optional): Number of points in each batch.
            Defaults to 8192.
        room_idxs (np.ndarray | str, optional): Precomputed index to load data.
            Map data sampling idx to room idx.
            Defaults to None.
        label_weight (np.ndarray | str, optional): Precomputed weight to \
            balance loss calculation. If None is given, compute from data.
            Defaults to None.
    """
    CLASSES = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
               'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
               'curtain', 'refrigerator', 'showercurtrain', 'toilet', 'sink',
               'bathtub', 'otherfurniture')

    VALID_CLASS_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28,
                       33, 34, 36, 39)  # class_ids used for training

    # all possible class_ids in loaded segmentation mask
    ALL_CLASS_IDS = tuple(range(41))

    # official color for visualization
    PALETTE = [
        [174, 199, 232],
        [152, 223, 138],
        [31, 119, 180],
        [255, 187, 120],
        [188, 189, 34],
        [140, 86, 75],
        [255, 152, 150],
        [214, 39, 40],
        [197, 176, 213],
        [148, 103, 189],
        [196, 156, 148],
        [23, 190, 207],
        [247, 182, 210],
        [219, 219, 141],
        [255, 127, 14],
        [158, 218, 229],
        [44, 160, 44],
        [112, 128, 144],
        [227, 119, 194],
        [82, 84, 163],
    ]

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 palette=None,
                 modality=None,
                 test_mode=False,
                 ignore_index=None,
                 num_points=8192,
                 room_idxs=None,
                 label_weight=None,
                 label_weight_func=None):

        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.filter_empty_gt = False
        self.box_type_3d, self.box_mode_3d = get_box_type('depth')
        self.label_weight_func = label_weight_func

        self.data_infos = self.load_annotations(self.ann_file)

        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        self.num_points = num_points
        self.ignore_index = len(self.CLASSES) if \
            ignore_index is None else ignore_index

        self.CLASSES, self.PALETTE = \
            self.get_classes_and_palette(classes, palette)
        self.room_idxs, self.label_weight = \
            self._get_room_idxs_and_label_weight(room_idxs, label_weight)

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - pts_semantic_mask_path (str): Path of semantic masks.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]

        pts_semantic_mask_path = osp.join(self.data_root,
                                          info['pts_semantic_mask_path'])

        anns_results = dict(pts_semantic_mask_path=pts_semantic_mask_path)
        return anns_results

    def show(self, results, out_dir, show=True):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        for i, result in enumerate(results):
            data_info = self.data_infos[i]
            pts_path = data_info['pts_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = np.fromfile(
                osp.join(self.data_root, pts_path),
                dtype=np.float32).reshape(-1, 6)
            sem_mask_path = data_info['pts_semantic_mask_path']
            gt_sem_mask = self._convert_to_train_label(
                osp.join(self.data_root, sem_mask_path))
            pred_sem_mask = result['semantic_mask'].numpy()
            show_seg_result(points, gt_sem_mask,
                            pred_sem_mask, out_dir, file_name,
                            np.array(self.PALETTE), self.ignore_index, show)

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        This function is taken from MMSegmentation.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
                Defaults to None.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Defaults to None.
        """
        if classes is None:
            self.custom_classes = False
            self.label_map = {
                cls_id: self.ignore_index
                for cls_id in self.ALL_CLASS_IDS
            }
            self.label_map.update(
                {cls_id: i
                 for i, cls_id in enumerate(self.VALID_CLASS_IDS)})
            self.label2cat = {
                i: cat_name
                for i, cat_name in enumerate(self.CLASSES)
            }
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(class_names).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # update valid_class_ids
            self.VALID_CLASS_IDS = [
                self.VALID_CLASS_IDS[self.CLASSES.index(cls_name)]
                for cls_name in class_names
            ]

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {
                cls_id: self.ignore_index
                for cls_id in self.ALL_CLASS_IDS
            }
            self.label_map.update(
                {cls_id: i
                 for i, cls_id in enumerate(self.VALID_CLASS_IDS)})
            self.label2cat = {
                i: cat_name
                for i, cat_name in enumerate(class_names)
            }

        palette = [
            self.PALETTE[self.CLASSES.index(cls_name)]
            for cls_name in class_names
        ]

        return class_names, palette

    def _get_room_idxs_and_label_weight(self, room_idxs, label_weight):
        """Compute room_idxs for data sampling and label weight for loss \
        calculation.

        We sample more times for rooms with more points. Label_weight is
        inversely proportional to number of class points.
        """
        if self.test_mode:
            return np.array(range(len(self.data_infos))), None

        if room_idxs is not None:  # precomputed
            if isinstance(room_idxs, str):
                room_idxs = np.load(room_idxs).astype(np.int32)
            if isinstance(label_weight, str):
                label_weight = np.load(label_weight).astype(np.float32)
            return room_idxs, label_weight

        num_classes = len(self.CLASSES)
        num_point_all = []
        label_weight = np.zeros((num_classes + 1, ))  # ignore_index
        for data_info in self.data_infos:
            label = self._convert_to_train_label(
                osp.join(self.data_root, data_info['pts_semantic_mask_path']))
            num_point_all.append(label.shape[0])
            class_count, _ = np.histogram(label, range(num_classes + 2))
            label_weight += class_count

        # repeat room_idx for num_room_point // num_sample_point times
        sample_prob = np.array(num_point_all) / float(np.sum(num_point_all))
        num_iter = int(np.sum(num_point_all) / float(self.num_points))
        room_idxs = []
        for idx in range(len(self.data_infos)):
            room_idxs.extend([idx] * round(sample_prob[idx] * num_iter))

        # calculate label weight, adopted from PointNet++
        if self.label_weight_func is None:
            label_weight = np.ones(num_classes, dtype=np.float32)
        else:
            label_weight = label_weight[:-1].astype(np.float32)
            label_weight = label_weight / label_weight.sum()
            label_weight = self.label_weight_func(label_weight)

        return np.array(room_idxs), label_weight

    def _convert_to_train_label(self, mask):
        """Convert class_id in loaded segmentation mask to label."""
        if isinstance(mask, str):
            if mask.endswith('npy'):
                mask = np.load(mask)
            else:
                mask = np.fromfile(mask, dtype=np.long)
        mask_copy = mask.copy()
        for class_id, label in self.label_map.items():
            mask_copy[mask == class_id] = label

        return mask_copy

    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 show=False,
                 out_dir=None):
        """Evaluate.

        Evaluation in semantic segmentation protocol.

        Args:
            results (list[dict]): List of results.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Defaults to None.
            show (bool, optional): Whether to visualize.
                Defaults to False.
            out_dir (str, optional): Path to save the visualization results.
                Defaults to None.

        Returns:
            dict: Evaluation results.
        """
        from mmdet3d.core.evaluation import seg_eval
        assert isinstance(
            results, list), f'Expect results to be list, got {type(results)}.'
        assert len(results) > 0, 'Expect length of results > 0.'
        assert len(results) == len(self.data_infos)
        assert isinstance(
            results[0], dict
        ), f'Expect elements in results to be dict, got {type(results[0])}.'
        pred_sem_masks = [result['semantic_mask'] for result in results]
        gt_sem_masks = [
            torch.from_numpy(
                self._convert_to_train_label(
                    osp.join(self.data_root,
                             data_info['pts_semantic_mask_path'])))
            for data_info in self.data_infos
        ]
        ret_dict = seg_eval(
            gt_sem_masks,
            pred_sem_masks,
            self.label2cat,
            len(self.CLASSES),
            self.ignore_index,
            logger=logger)
        if show:
            self.show(pred_sem_masks, out_dir)

        return ret_dict

    def __len__(self):
        """Return the length of room_idxs.

        Returns:
            int: Length of data infos.
        """
        return len(self.room_idxs)

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        room_idx = self.room_idxs[idx]  # map to room idx
        if self.test_mode:
            return self.prepare_test_data(room_idx)
        while True:
            data = self.prepare_train_data(room_idx)
            if data is None:
                idx = self._rand_another(idx)
                room_idx = self.room_idxs[idx]  # map to room idx
                continue
            return data
