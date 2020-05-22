import mmcv
import numpy as np
from torch.utils.data import Dataset

from mmdet.datasets import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class Custom3DDataset(Dataset):

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 modality=None,
                 test_mode=False):
        super().__init__()
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality

        self.CLASSES = self.get_classes(classes)
        self.data_infos = self.load_annotations(self.ann_file)

        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

    def load_annotations(self, ann_file):
        return mmcv.load(ann_file)

    def get_data_info(self, index):
        info = self.data_infos[index]
        sample_idx = info['point_cloud']['lidar_idx']
        pts_filename = self._get_pts_filename(sample_idx)

        input_dict = dict(pts_filename=pts_filename)

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if len(annos['gt_bboxes_3d']) == 0:
                return None
        return input_dict

    def pre_pipeline(self, results):
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []

    def prepare_train_data(self, index):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if example is None or len(example['gt_bboxes_3d']._data) == 0:
            return None
        return example

    def prepare_test_data(self, index):
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Return:
            list[str]: return the list of class names
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names

    def _generate_annotations(self, output):
        """Generate annotations.

        Transform results of the model to the form of the evaluation.

        Args:
            output (list): The output of the model.
        """
        result = []
        bs = len(output)
        for i in range(bs):
            pred_list_i = list()
            pred_boxes = output[i]
            box3d_depth = pred_boxes['box3d_lidar']
            if box3d_depth is not None:
                label_preds = pred_boxes['label_preds']
                scores = pred_boxes['scores']
                label_preds = label_preds.detach().cpu().numpy()
                for j in range(box3d_depth.shape[0]):
                    bbox_lidar = box3d_depth[j]  # [7] in lidar
                    bbox_lidar_bottom = bbox_lidar.copy()
                    pred_list_i.append(
                        (label_preds[j], bbox_lidar_bottom, scores[j]))
                result.append(pred_list_i)
            else:
                result.append(pred_list_i)

        return result

    def format_results(self, outputs):
        results = []
        for output in outputs:
            result = self._generate_annotations(output)
            results.append(result)
        return results

    def evaluate(self, results, metric=None):
        """Evaluate.

        Evaluation in indoor protocol.

        Args:
            results (list): List of result.
            metric (list[float]): AP IoU thresholds.
        """
        results = self.format_results(results)
        from mmdet3d.core.evaluation import indoor_eval
        assert len(metric) > 0
        gt_annos = [info['annos'] for info in self.data_infos]
        label2cat = {i: cat_id for i, cat_id in enumerate(self.CLASSES)}
        ret_dict = indoor_eval(gt_annos, results, metric, label2cat)
        return ret_dict

    def __len__(self):
        return len(self.data_infos)

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        In 3D datasets, they are all the same, thus are all zeros

        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
