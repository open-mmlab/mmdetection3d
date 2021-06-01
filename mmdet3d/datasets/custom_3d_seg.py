import mmcv
import numpy as np
import tempfile
import warnings
from os import path as osp
from torch.utils.data import Dataset

from mmdet.datasets import DATASETS
from mmseg.datasets import DATASETS as SEG_DATASETS
from .pipelines import Compose
from .utils import extract_result_dict, get_loading_pipeline


@DATASETS.register_module()
@SEG_DATASETS.register_module()
class Custom3DSegDataset(Dataset):
    """Customized 3D dataset for semantic segmentation task.

    This is the base dataset of ScanNet and S3DIS dataset.

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
            unannotated points. If None is given, set to len(self.CLASSES) to
            be consistent with PointSegClassMapping function in pipeline.
            Defaults to None.
        scene_idxs (np.ndarray | str, optional): Precomputed index to load
            data. For scenes with many points, we may sample it several times.
            Defaults to None.
    """
    # names of all classes data used for the task
    CLASSES = None

    # class_ids used for training
    VALID_CLASS_IDS = None

    # all possible class_ids in loaded segmentation mask
    ALL_CLASS_IDS = None

    # official color for visualization
    PALETTE = None

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 palette=None,
                 modality=None,
                 test_mode=False,
                 ignore_index=None,
                 scene_idxs=None):
        super().__init__()
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality

        self.data_infos = self.load_annotations(self.ann_file)

        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        self.ignore_index = len(self.CLASSES) if \
            ignore_index is None else ignore_index

        self.scene_idxs = self.get_scene_idxs(scene_idxs)
        self.CLASSES, self.PALETTE = \
            self.get_classes_and_palette(classes, palette)

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        return mmcv.load(ann_file)

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - file_name (str): Filename of point clouds.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        sample_idx = info['point_cloud']['lidar_idx']
        pts_filename = osp.join(self.data_root, info['pts_path'])

        input_dict = dict(
            pts_filename=pts_filename,
            sample_idx=sample_idx,
            file_name=pts_filename)

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
        return input_dict

    def pre_pipeline(self, results):
        """Initialization before data preparation.

        Args:
            results (dict): Dict before data preprocessing.

                - img_fields (list): Image fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
        """
        results['img_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['bbox3d_fields'] = []

    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

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
            # map id in the loaded mask to label used for training
            self.label_map = {
                cls_id: self.ignore_index
                for cls_id in self.ALL_CLASS_IDS
            }
            self.label_map.update(
                {cls_id: i
                 for i, cls_id in enumerate(self.VALID_CLASS_IDS)})
            # map label to category name
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

        # modify palette for visualization
        palette = [
            self.PALETTE[self.CLASSES.index(cls_name)]
            for cls_name in class_names
        ]

        return class_names, palette

    def get_scene_idxs(self, scene_idxs):
        """Compute scene_idxs for data sampling.

        We sample more times for scenes with more points.
        """
        if self.test_mode:
            # when testing, we load one whole scene every time
            return np.arange(len(self.data_infos)).astype(np.int32)

        # we may need to re-sample different scenes according to scene_idxs
        # this is necessary for indoor scene segmentation such as ScanNet
        if scene_idxs is None:
            scene_idxs = np.arange(len(self.data_infos))
        if isinstance(scene_idxs, str):
            scene_idxs = np.load(scene_idxs)
        else:
            scene_idxs = np.array(scene_idxs)

        return scene_idxs.astype(np.int32)

    def format_results(self,
                       outputs,
                       pklfile_prefix=None,
                       submission_prefix=None):
        """Format the results to pkl file.

        Args:
            outputs (list[dict]): Testing results of the dataset.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (outputs, tmp_dir), outputs is the detection results, \
                tmp_dir is the temporal directory created for saving json \
                files when ``jsonfile_prefix`` is not specified.
        """
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
            out = f'{pklfile_prefix}.pkl'
        mmcv.dump(outputs, out)
        return outputs, tmp_dir

    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
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
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

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

        load_pipeline = self._get_pipeline(pipeline)
        pred_sem_masks = [result['semantic_mask'] for result in results]
        gt_sem_masks = [
            self._extract_data(
                i, load_pipeline, 'pts_semantic_mask', load_annos=True)
            for i in range(len(self.data_infos))
        ]
        ret_dict = seg_eval(
            gt_sem_masks,
            pred_sem_masks,
            self.label2cat,
            self.ignore_index,
            logger=logger)

        if show:
            self.show(pred_sem_masks, out_dir, pipeline=pipeline)

        return ret_dict

    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        raise NotImplementedError('_build_default_pipeline is not implemented '
                                  f'for dataset {self.__class__.__name__}')

    def _get_pipeline(self, pipeline):
        """Get data loading pipeline in self.show/evaluate function.

        Args:
            pipeline (list[dict] | None): Input pipeline. If None is given, \
                get from self.pipeline.
        """
        if pipeline is None:
            if not hasattr(self, 'pipeline') or self.pipeline is None:
                warnings.warn(
                    'Use default pipeline for data loading, this may cause '
                    'errors when data is on ceph')
                return self._build_default_pipeline()
            loading_pipeline = get_loading_pipeline(self.pipeline.transforms)
            return Compose(loading_pipeline)
        return Compose(pipeline)

    def _extract_data(self, index, pipeline, key, load_annos=False):
        """Load data using input pipeline and extract data according to key.

        Args:
            index (int): Index for accessing the target data.
            pipeline (:obj:`Compose`): Composed data loading pipeline.
            key (str | list[str]): One single or a list of data key.
            load_annos (bool): Whether to load data annotations.
                If True, need to set self.test_mode as False before loading.

        Returns:
            np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor]:
                A single or a list of loaded data.
        """
        assert pipeline is not None, 'data loading pipeline is not provided'
        # when we want to load ground-truth via pipeline (e.g. bbox, seg mask)
        # we need to set self.test_mode as False so that we have 'annos'
        if load_annos:
            original_test_mode = self.test_mode
            self.test_mode = False
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = pipeline(input_dict)

        # extract data items according to keys
        if isinstance(key, str):
            data = extract_result_dict(example, key)
        else:
            data = [extract_result_dict(example, k) for k in key]
        if load_annos:
            self.test_mode = original_test_mode

        return data

    def __len__(self):
        """Return the length of scene_idxs.

        Returns:
            int: Length of data infos.
        """
        return len(self.scene_idxs)

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        In indoor scene segmentation task, each scene contains millions of
        points. However, we only sample less than 10k points within a patch
        each time. Therefore, we use `scene_idxs` to re-sample different rooms.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        scene_idx = self.scene_idxs[idx]  # map to scene idx
        if self.test_mode:
            return self.prepare_test_data(scene_idx)
        while True:
            data = self.prepare_train_data(scene_idx)
            if data is None:
                idx = self._rand_another(idx)
                scene_idx = self.scene_idxs[idx]  # map to scene idx
                continue
            return data

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
